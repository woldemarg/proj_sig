"""Optional Neo4j mirror of the graph snapshot (docs/06_graph_and_storage.md §6.6).

Follows lac's publisher pattern (constraints + parameterised UNWIND/MERGE) with
the SIG schema. The local journal/state is the source of truth and every publish makes
the mirror equal to the snapshot: properties are replaced, not merged, and nodes of the
six SIG labels and relationships of the eleven SIG types that the snapshot no longer holds
are deleted (after a reset, a rebuild or a recomputed topology). SIG owns
these labels in ``NEO4J_DATABASE``: one workspace per database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insight_graph_service.core.settings import Settings

CYPHER_DIR = Path(__file__).parent / "cypher"
LABELS = ("Pattern", "Attractor", "Dimension", "Metric", "Dataset", "Batch")
EDGE_ENDPOINTS = {
    "SPECIALIZES": ("Pattern", "Pattern"),
    "GENERALIZES": ("Pattern", "Pattern"),
    "SIBLING": ("Pattern", "Pattern"),
    "CONTRASTS": ("Pattern", "Pattern"),
    "HAS_SCOPE": ("Pattern", "Dimension"),
    "TARGETS": ("Pattern", "Metric"),
    "ACTIVATES": ("Pattern", "Attractor"),
    "RELATED_TO": ("Attractor", "Attractor"),
    "CO_OCCURS": ("Attractor", "Attractor"),
    "DISCOVERED_IN": ("Pattern", "Batch"),
    "OF_DATASET": ("Batch", "Dataset"),
}


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


def publish_snapshot(snapshot: dict[str, Any], settings: Settings, driver: Any = None) -> dict[str, Any]:
    """Make ``NEO4J_DATABASE`` equal to the snapshot (an empty snapshot clears it); returns counts.

    ``driver`` is injectable for tests."""
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
        try:  # multi-database editions: create the SIG database on first use
            with driver.session(database="system") as session:
                session.run(f"CREATE DATABASE {settings.neo4j_database} IF NOT EXISTS WAIT")
        except Exception:
            pass  # community edition / insufficient rights: use the database as configured
        nodes, edges = snapshot_rows(snapshot)
        batch = settings.neo4j_load_batch_size
        with driver.session(database=settings.neo4j_database) as session:
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
                for etype, rows in edges.items():
                    tx.run(stale_edge_query(etype), keys=[f"{r['source']}|{r['target']}" for r in rows])
                for label, rows in nodes.items():
                    tx.run(stale_node_query(label), ids=[r["id"] for r in rows])

            session.execute_write(write)
        return {
            "nodes": {k: len(v) for k, v in nodes.items()},
            "edges": {k: len(v) for k, v in edges.items() if v},
            "database": settings.neo4j_database,
        }
    finally:
        if own:
            driver.close()
