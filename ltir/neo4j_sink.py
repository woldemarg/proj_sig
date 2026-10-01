"""Optional Neo4j mirror of the graph snapshot (SDD 09 §Neo4j).

Follows lac's publisher pattern (constraints + parameterised UNWIND/MERGE) with
the SIG schema. The local journal/state is the source of truth; publishing is
idempotent (MERGE on unique ids) and RELATED_TO is replaced wholesale so stale
mutual-kNN edges do not linger (lac's accepted MERGE-only trade-off is fixed here).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ltir.config import Config

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
        f"SET r += row.props, r.weight = row.weight"
    )


def node_query(label: str) -> str:
    assert label in LABELS
    return f"UNWIND $rows AS row\nMERGE (n:{label} {{id: row.id}})\nSET n += row.props"


def publish_snapshot(snapshot: dict[str, Any], config: Config, driver: Any = None) -> dict[str, Any]:
    """MERGE the snapshot into ``NEO4J_DATABASE``; returns counts. ``driver`` is injectable for tests."""
    own = driver is None
    if own:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
    try:
        try:  # multi-database editions: create the SIG database on first use
            with driver.session(database="system") as session:
                session.run(f"CREATE DATABASE {config.neo4j_database} IF NOT EXISTS WAIT")
        except Exception:
            pass  # community edition / insufficient rights: use the database as configured
        nodes, edges = snapshot_rows(snapshot)
        batch = config.neo4j_load_batch_size
        with driver.session(database=config.neo4j_database) as session:
            for stmt in (s.strip() for s in load_cypher("ensure_constraints").split(";")):
                if stmt:
                    session.run(stmt)

            def write(tx) -> None:
                for label, rows in nodes.items():
                    for i in range(0, len(rows), batch):
                        tx.run(node_query(label), rows=rows[i : i + batch])
                tx.run(load_cypher("clear_related_to"))
                for etype, rows in edges.items():
                    for i in range(0, len(rows), batch):
                        tx.run(edge_query(etype), rows=rows[i : i + batch])

            session.execute_write(write)
        return {
            "nodes": {k: len(v) for k, v in nodes.items()},
            "edges": {k: len(v) for k, v in edges.items() if v},
            "database": config.neo4j_database,
        }
    finally:
        if own:
            driver.close()
