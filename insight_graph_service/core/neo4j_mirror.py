"""Optional Neo4j mirror of the graph snapshot (docs/06_graph_and_storage.md §6.6).

Follows lac's publisher pattern (constraints + parameterised UNWIND/MERGE) with
the SIG schema. The local journal/state is the source of truth and every publish makes
the mirror equal to the snapshot: properties are replaced, not merged. A full sync (the
first publish of a process, and the one after a failed publish) writes every row and deletes
the nodes of the SIG labels and relationships of the SIG types that the snapshot does not
hold, as well as relationships between SIG nodes of a type the schema no longer has; every
other publish sends only what changed since the last one written. SIG owns these labels in
``NEO4J_DATABASE``: one workspace per database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insight_graph_service.core.settings import Settings

CYPHER_DIR = Path(__file__).parent / "cypher"
LABELS = ("Pattern", "Attractor", "Region", "TemporalNode", "Dimension", "Metric", "Dataset", "Batch")
EDGE_ENDPOINTS = {
    "SPECIALIZES": ("Pattern", "Pattern"),
    "SIBLING": ("Pattern", "Pattern"),
    "CONTRASTS": ("Pattern", "Pattern"),
    "CO_LOCATED": ("Pattern", "Pattern"),
    "IN_REGION": ("Pattern", "Region"),
    "PARENT_OF": ("Region", "Region"),
    "HAS_SCOPE": ("Pattern", "Dimension"),
    "TARGETS": ("Pattern", "Metric"),
    "ACTIVATES": ("Pattern", "Attractor"),
    "RELATED_TO": ("Attractor", "Attractor"),
    "CO_OCCURS": ("Attractor", "Attractor"),
    "DISCOVERED_IN": ("Pattern", "Batch"),
    "OF_DATASET": ("Batch", "Dataset"),
    "PEAKED_AT": ("Pattern", "TemporalNode"),
    "DROPPED_AT": ("Pattern", "TemporalNode"),
    "STARTS_AT": ("Pattern", "TemporalNode"),
    "ENDS_AT": ("Pattern", "TemporalNode"),
    "WITHIN": ("TemporalNode", "TemporalNode"),
}
STALE_TYPES_QUERY = (  # relationships between SIG nodes of a type EDGE_ENDPOINTS no longer lists
    "MATCH (s)-[r]->(t) WHERE NOT type(r) IN $types AND any(l IN labels(s) WHERE l IN $labels) AND any(l IN labels(t) WHERE l IN $labels) DELETE r"
)


def load_cypher(name: str) -> str:
    return (CYPHER_DIR / f"{name}.cypher").read_text(encoding="utf-8").strip()


def flatten_props(props: dict[str, Any]) -> dict[str, Any]:
    """Neo4j properties must be primitives or homogeneous primitive lists; nest -> JSON string."""
    out: dict[str, Any] = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value) and len({type(v) for v in value}) <= 1:
            out[key] = value
        else:
            out[f"{key}_json"] = json.dumps(value, ensure_ascii=False, default=str)
    return out


def snapshot_rows(snapshot: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    nodes: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for n in snapshot["nodes"]:
        if n["kind"] in nodes:
            nodes[n["kind"]].append({"id": n["id"], "props": {"label": n["label"], **flatten_props(n["props"])}})
    edges: dict[str, list[dict[str, Any]]] = {t: [] for t in EDGE_ENDPOINTS}
    for e in snapshot["edges"]:
        edges[e["type"]].append({"source": e["source"], "target": e["target"], "weight": float(e["weight"]), "props": flatten_props(e["props"])})
    return nodes, edges


def edge_query(edge_type: str) -> str:
    """Relationship types cannot be parameters; they come from the fixed EDGE_ENDPOINTS enum."""
    src, dst = EDGE_ENDPOINTS[edge_type]
    return (
        f"UNWIND $rows AS row\n"
        f"MATCH (s:{src} {{id: row.source}})\n"
        f"MATCH (t:{dst} {{id: row.target}})\n"
        f"MERGE (s)-[r:{edge_type}]->(t)\n"
        f"SET r = row.props, r.weight = row.weight"
    )


def node_query(label: str) -> str:
    assert label in LABELS
    return f"UNWIND $rows AS row\nMERGE (n:{label} {{id: row.id}})\nSET n = row.props, n.id = row.id"


def stale_node_query(label: str) -> str:
    """Delete the label's nodes that the snapshot does not hold (with their relationships)."""
    assert label in LABELS
    return f"MATCH (n:{label}) WHERE NOT n.id IN $ids DETACH DELETE n"


def stale_edge_query(edge_type: str) -> str:
    """Delete the type's relationships whose (source, target) pair the snapshot does not hold."""
    src, dst = EDGE_ENDPOINTS[edge_type]
    return f"MATCH (s:{src})-[r:{edge_type}]->(t:{dst}) WHERE NOT (s.id + '|' + t.id) IN $keys DELETE r"


def gone_edge_query(edge_type: str) -> str:
    """Delete the type's relationships between the listed (source, target) pairs."""
    src, dst = EDGE_ENDPOINTS[edge_type]
    return f"UNWIND $rows AS row\nMATCH (s:{src} {{id: row.source}})-[r:{edge_type}]->(t:{dst} {{id: row.target}})\nDELETE r"


def gone_node_query(label: str) -> str:
    """Delete the label's listed nodes with their relationships."""
    assert label in LABELS
    return f"MATCH (n:{label}) WHERE n.id IN $ids DETACH DELETE n"


def row_hashes(rows: dict[str, list[dict[str, Any]]], key: tuple[str, ...]) -> dict[tuple[str, ...], int]:
    """``(label or type, *key)`` -> a hash of the row as written: what the next diff compares against."""
    return {(kind, *(r[k] for k in key)): hash(json.dumps(r, sort_keys=True, default=str)) for kind, kind_rows in rows.items() for r in kind_rows}


class Mirror:
    """The Neo4j mirror as this process last wrote it: ``written`` holds the row hashes of its last successful publish,
    None before the first one and after a failed one."""

    def __init__(self) -> None:
        self.written: tuple[dict, dict] | None = None

    def publish(self, snapshot: dict[str, Any], settings: Settings, driver: Any = None) -> dict[str, Any]:
        """Make ``NEO4J_DATABASE`` equal to the snapshot (an empty one clears it): a full sync while ``written`` is None,
        else only the rows that differ from it and explicit deletions; returns counts. ``driver`` is injectable for tests."""
        previous, self.written = self.written, None
        nodes, edges = snapshot_rows(snapshot)
        now = (row_hashes(nodes, ("id",)), row_hashes(edges, ("source", "target")))
        out = {
            "nodes": {k: len(v) for k, v in nodes.items() if v},
            "edges": {k: len(v) for k, v in edges.items() if v},
            "database": settings.neo4j_database,
            "sync": "full" if previous is None else "diff",
        }
        if previous is not None:
            nodes = {kind: [r for r in rows if previous[0].get((kind, r["id"])) != now[0][(kind, r["id"])]] for kind, rows in nodes.items()}
            edges = {kind: [r for r in rows if previous[1].get(k := (kind, r["source"], r["target"])) != now[1][k]] for kind, rows in edges.items()}
            gone_nodes = [k for k in previous[0] if k not in now[0]]
            gone_edges = [k for k in previous[1] if k not in now[1]]
            out["deleted"] = len(gone_nodes) + len(gone_edges)
        out["written"] = sum(map(len, nodes.values())) + sum(map(len, edges.values()))
        if previous is not None and not out["written"] and not out["deleted"]:
            self.written = now
            return out
        own = driver is None
        if own:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(  # a mirror that is down fails fast: the sync runs under the write lock
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                connection_timeout=5.0,
                connection_acquisition_timeout=10.0,
                max_transaction_retry_time=5.0,
            )
        try:
            if previous is None:
                try:  # multi-database editions: create the SIG database on first use
                    with driver.session(database="system") as session:
                        session.run(f"CREATE DATABASE {settings.neo4j_database} IF NOT EXISTS WAIT")
                except Exception:
                    pass  # community edition / insufficient rights: use the database as configured
            batch = settings.neo4j_load_batch_size
            with driver.session(database=settings.neo4j_database) as session:
                if previous is None:  # a diff follows a full sync of this process, which made them
                    for stmt in (s.strip() for s in load_cypher("ensure_constraints").split(";")):
                        if stmt:
                            session.run(stmt)

                def write(tx) -> None:
                    for label, rows in nodes.items():
                        for i in range(0, len(rows), batch):
                            tx.run(node_query(label), rows=rows[i : i + batch])
                    for etype, rows in edges.items():
                        for i in range(0, len(rows), batch):
                            tx.run(edge_query(etype), rows=rows[i : i + batch])
                    if previous is None:
                        for etype in EDGE_ENDPOINTS:
                            tx.run(stale_edge_query(etype), keys=[f"{k[1]}|{k[2]}" for k in now[1] if k[0] == etype])
                        tx.run(STALE_TYPES_QUERY, types=list(EDGE_ENDPOINTS), labels=list(LABELS))
                        for label in LABELS:
                            tx.run(stale_node_query(label), ids=[k[1] for k in now[0] if k[0] == label])
                        return
                    for etype in EDGE_ENDPOINTS:
                        rows = [{"source": k[1], "target": k[2]} for k in gone_edges if k[0] == etype]
                        for i in range(0, len(rows), batch):
                            tx.run(gone_edge_query(etype), rows=rows[i : i + batch])
                    for label in LABELS:
                        ids = [k[1] for k in gone_nodes if k[0] == label]
                        for i in range(0, len(ids), batch):
                            tx.run(gone_node_query(label), ids=ids[i : i + batch])

                session.execute_write(write)
        finally:
            if own:
                driver.close()
        self.written = now
        return out
