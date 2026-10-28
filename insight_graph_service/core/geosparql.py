"""The committed graph as RDF Turtle with GeoSPARQL geometries and a ``sig:`` vocabulary (docs/06_graph_and_storage.md §6.9)."""

from __future__ import annotations

import json
import math
from typing import Any

from graph_query_engine import DualGraph
from insight_contracts import dataset_node_id
from subgroup_miner.spatial_plane import cells_to_wkt

PREFIXES = {
    "geo": "http://www.opengis.net/ont/geosparql#",
    "prov": "http://www.w3.org/ns/prov#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "sig": "urn:sig:vocab#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}
CO_LOCATED = {"overlaps": "sig:extentsOverlap", "adjacent": "sig:extentsTouch", "same_region": "sig:sameRegion"}
VOCABULARY = [  # the sig: terms' relations; their meaning is in docs/06_graph_and_storage.md §6.9
    "sig:Insight rdfs:subClassOf prov:Entity",
    "sig:Dataset rdfs:subClassOf prov:Entity",
    *(f"{p} rdfs:subPropertyOf sig:coLocated" for p in CO_LOCATED.values()),
    "sig:inRegionByEvents rdfs:subPropertyOf sig:inRegion",
]


def iri(node_id: str) -> str:
    """A graph node's IRI: its id under ``urn:sig:``, the same for every export of the same graph."""
    return f"<urn:sig:{node_id}>"


def literal(value: Any) -> str:
    """A Turtle literal: a JSON string is a valid Turtle string; floats are typed ``xsd:double`` (``NaN``, ``INF`` as xsd spells them)."""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value) if math.isfinite(value) else "NaN" if math.isnan(value) else "INF" if value > 0 else "-INF"
        return f'"{text}"^^xsd:double'
    return json.dumps(str(value), ensure_ascii=False)


def _subject(lines: list[str], node: str, types: list[str], props: list[tuple[str, str]]) -> None:
    lines.append(f"{node} a {', '.join(types)}" + "".join(f" ;\n    {p} {o}" for p, o in props) + " .")


def build_turtle(graph: DualGraph, extents: dict[str, list[str]]) -> str:
    """The graph's datasets, insights, regions and anchors as Turtle. ``extents``: a geo insight's H3 cells, which give its
    WKT geometry. Region parents are ``sig:parentRegion``, not ``geo:sfWithin``: H3 children do not tile their parent."""
    lines = [f"@prefix {k}: <{v}> ." for k, v in PREFIXES.items()] + [""] + [f"{t} ." for t in VOCABULARY] + [""]
    uploads = set()
    for d in graph.of_kind("Dataset"):
        p = d["props"]
        props = [("rdfs:label", literal(p.get("filename") or p["dataset_id"])), ("sig:sourceReliability", literal(p["reliability"]))]
        props += [("sig:source", literal(p["source"]))] if p.get("source") else []
        props += [("sig:integrity", literal(p["integrity"]))]
        if p.get("upload_id"):
            props.append(("prov:wasDerivedFrom", iri(p["upload_id"])))
            uploads.add(p["upload_id"])
        _subject(lines, iri(d["id"]), ["sig:Dataset"], props)
    for up in sorted(uploads):
        _subject(lines, iri(up), ["prov:Entity"], [("rdfs:label", literal(f"uploaded file {up}"))])
    for n in graph.of_kind("Pattern"):
        ins, p = graph.insight(n["id"]), n["props"]
        props = [("rdfs:label", literal(n["label"])), ("sig:target", literal(ins.target)), ("sig:effectSize", literal(float(ins.effect_size)))]
        props += [("sig:support", literal(int(ins.support))), ("sig:weight", literal(float(ins.weight)))]
        props += [("sig:pAdjusted", literal(float(ins.p_adjusted)))] if ins.p_adjusted is not None else []
        props += [("sig:credibility", literal(p["credibility"]))] if p.get("credibility") is not None else []
        props += [("sig:integrity", literal(p["integrity"])), ("prov:wasDerivedFrom", iri(dataset_node_id(ins.dataset_id)))]
        types, geometry = ["sig:Insight"], None
        if n["id"] in extents:
            types.append("geo:Feature")
            props.append(("geo:hasGeometry", iri(f"{n['id']}:geometry")))
            props += [("sig:h3Cell", literal(c)) for c in extents[n["id"]]]
            geometry = literal(cells_to_wkt(extents[n["id"]])) + "^^geo:wktLiteral"
        for e in graph.out.get(n["id"], []):
            if e["type"] == "IN_REGION":  # a base pattern's region holds its events, not its cells
                props.append(("sig:inRegionByEvents" if e["props"].get("basis") == "events" else "sig:inRegion", iri(e["target"])))
            elif e["type"] == "ACTIVATES" and not e["props"].get("weak"):
                props.append(("sig:activates", iri(e["target"])))
        _subject(lines, iri(n["id"]), types, props)
        if geometry:
            _subject(lines, iri(f"{n['id']}:geometry"), ["geo:Geometry"], [("geo:asWKT", geometry)])
    for r in graph.of_kind("Region"):
        p = r["props"]
        props = [("sig:resolution", literal(int(p["resolution"]))), ("sig:h3Cell", literal(p["h3"])), ("geo:hasGeometry", iri(f"{r['id']}:geometry"))]
        props += [("sig:parentRegion", iri(e["target"])) for e in graph.out.get(r["id"], []) if e["type"] == "PARENT_OF"]
        _subject(lines, iri(r["id"]), ["sig:Region", "geo:Feature"], props)
        _subject(lines, iri(f"{r['id']}:geometry"), ["geo:Geometry"], [("geo:asWKT", literal(cells_to_wkt([p["h3"]])) + "^^geo:wktLiteral")])
    anchors = graph.of_kind("Attractor")
    if anchors:
        _subject(lines, iri("anchors"), ["skos:ConceptScheme"], [("rdfs:label", literal("latent anchors"))])
    for a in anchors:
        props = [("skos:prefLabel", literal(a["label"])), ("skos:inScheme", iri("anchors"))]
        props += [("skos:related", iri(e["target"])) for e in graph.out.get(a["id"], []) if e["type"] == "RELATED_TO"]
        _subject(lines, iri(a["id"]), ["skos:Concept"], props)
    for e in graph.edges:
        if e["type"] == "CO_LOCATED" and e["props"].get("relation") in CO_LOCATED:
            lines.append(f"{iri(e['source'])} {CO_LOCATED[e['props']['relation']]} {iri(e['target'])} .")
    return "\n".join(lines) + "\n"
