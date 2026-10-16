"""Evidence builder (docs/07_question_answering.md §7.4): traversal result -> structured, citable evidence.

The LLM receives *only* this object's ``to_prompt()`` rendering: English built from the kernel's phrases
(``insight_contracts.text``, docs/04_representation.md §4.2), with rounded numbers, p-value buckets, shifts in robust
standard deviations and prose scopes, and the data literals as the data holds them. ``summary()`` is the cited
observation lines, in Ukrainian around the untouched literals, that the narrator frames as the evidence-only answer
when no LLM answer is available. Every item has a citation key ``[P#]`` that maps back to a Pattern id, its
dataset, batch and exact EDA selector (the insight's own fields, over whatever its free-form provenance holds),
so answers are traceable to table slices.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from graph_query_engine.config import QueryConfig
from graph_query_engine.graph import DualGraph, describe_components
from graph_query_engine.question import ParsedQuery
from graph_query_engine.traversal import LATENT, PathStep, TraversalResult
from graph_query_engine.vertical import ScaleChain, picture, rank_zones, resolution, scale_chain, upload_of
from insight_contracts import SpatialProvenance, dataset_node_id, metric_node_id
from insight_contracts.text import (
    RELIABILITY,
    describe_covariance,
    describe_scope,
    describe_shift,
    describe_validation,
    format_value,
    humanize,
)

EVIDENCE_MAX_PATTERNS = 10
LOCATION_TAG = {  # an item reached only by location: the same area, not a related finding
    "same_region": " [reached by location only: same area, zones do not overlap]",
    "adjacent": " [reached by location only: zones touch or barely overlap]",
}
SCALE_TEXT = {  # the scale label of a geo item (docs/07_question_answering.md §7.4)
    "regional": "regional (holds at a coarser level of this upload)",
    "scale-dependent": "scale-dependent (reverses at a coarser level of this upload; see SCALE CHAIN)",
    "not_found_coarser": "not found at the coarser level",
}  # insufficient (no coarser level planned) stays on the item, unprinted: a one-level upload's prompt does not change
AREA_TAG = {True: " | in the asked area", False: " | outside the asked area"}  # an item of a level the zone weighed
AREA_WORDS = {True: " (у виділеній зоні)", False: " (поза виділеною зоною)"}
SCALE_WORDS = {  # the same labels in the evidence-only summary
    "regional": " (тримається і на рівні району)",
    "scale-dependent": " (на рівні району зсув протилежний)",
    "not_found_coarser": " (на рівні району окремо не підтверджено)",
}


@dataclass
class EvidenceItem:
    key: str
    pattern_id: str
    role: str  # seed | structural | transversal
    headline: str
    scope: list[str]
    target: str
    phenomenon_type: str
    statistics: dict[str, Any]
    scope_text: str  # readable scope ("category is phones and region is US")
    shift_text: list[str]  # phenomenon shifts as phrases (target + |z| >= min_component_z)
    relationship: str  # material correlation change as a phrase, or ""
    path: list[dict[str, Any]]
    path_text: str
    validation: str  # what validated it: bootstrap stability and adjusted p, or the correlation change
    attractors: list[dict[str, Any]]
    transversal_only: bool
    provenance: dict[str, Any]
    located: str = ""  # adjacent | same_region: reached only by location, scope-disjoint from the seeds
    scale: str = ""  # regional | scale-dependent | not_found_coarser | insufficient (ScaleChain.scale)
    folded_into: str = ""  # the key of the item this one restates at another level of the same upload
    also: list[str] = field(default_factory=list)  # the keys of the levels of the same upload that restate this item
    credibility: int | None = None  # 1 | 2 | 5 | 6 from independent sources (docs/06_graph_and_storage.md §6.2); None: an older snapshot
    credibility_basis: dict[str, Any] = field(default_factory=dict)  # rule, sources, patterns, comparability


@dataclass
class Evidence:
    query: str
    parsed: dict[str, Any]
    seed_patterns: list[str]
    items: list[EvidenceItem]
    attractors: list[dict[str, Any]]
    paths: list[str]
    metrics: list[dict[str, Any]]
    datasets: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)
    scale_chains: list[dict[str, Any]] = field(default_factory=list)  # ScaleChain as dicts, the items' keys added
    zones: list[dict[str, Any]] = field(default_factory=list)  # rank_zones: a question without scope over geo items
    area: dict[str, float] | None = None  # an asked map zone: item key -> place (0 outside) for the levels it weighed; None: no zone

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt(self) -> str:
        p = self.parsed
        plain = self.plain_language()
        lines = [f"QUESTION: {self.query or '(no text: the map area below is the question)'}"]
        parsed_bits = []
        if p.get("targets"):
            parsed_bits.append("target=" + ",".join(p["targets"]))
        if p.get("direction"):
            parsed_bits.append("direction=" + ("up" if p["direction"] > 0 else "down"))
        if p.get("conditions"):
            parsed_bits.append("scope=" + ",".join(p["conditions"]))
        lines.append("PARSED: " + (" | ".join(parsed_bits) or "no explicit metric/scope recognised"))
        if self.area is not None:  # findings of the levels the zone weighed; it says nothing of the others
            inside = ", ".join(f"{k} {v:.0%}" if v >= 0.005 else f"{k} <1%" for k, v in self.area.items() if v)
            outside = ", ".join(k for k, v in self.area.items() if not v)
            lines.append(
                "AREA: the question is about a map area; "
                + (
                    f"in it, with how closely each one's map cells match it (100% = the same cells): {inside}"
                    if inside
                    else "none of the findings lies in it"
                )
                + (f"; outside it: {outside}." if outside else ".")
            )
        lines.append(
            "UNITS: shifts are robust standard deviations (sd = median difference scaled by the MAD); medians compare the subgroup with the whole dataset."
        )
        lines.append("")
        lines.append("DATASETS:")
        for d in self.datasets:
            trust = f" source={json.dumps(d['source'], ensure_ascii=False)}" if d.get("source") else ""  # quoted: free text
            if trust or d.get("reliability", "F") != "F":  # an undeclared dataset adds nothing
                trust += f" reliability={d['reliability']} ({RELIABILITY.get(d['reliability'], 'unrecognised code')})"
            lines.append(f"- {d['dataset_id']} file={d['filename']} batch={d['batch_id']} rows={d.get('rows')}{trust}")
        if plain:
            lines.append("")
            lines.append("PLAIN LANGUAGE (rows are map cells built from point events; a subgroup is a set of map cells):")
            lines.extend(f"- {meaning}" for _, meaning in plain)
        lines.append("")
        lines.append("METRIC BASELINES (whole dataset):")
        for m in self.metrics:
            parts = [f"{k} {json.dumps(m[k], ensure_ascii=False)}" for k in ("indicator", "unit") if k in m]
            meaning = f" ({', '.join(parts)})" if parts else ""
            lines.append(
                f"- {self.reword(humanize(m['metric']))}{meaning}: median {format_value(m['global_median'])}, MAD {format_value(m['global_mad'])}"
            )
        if self.attractors:
            lines.append("")
            lines.append("LATENT ANCHORS VISITED (recurring phenomena learned across patterns):")
            for a in self.attractors:
                rel = ", ".join(f"{r['id']} ({self._links(r)})" for r in a["related"])
                twin = next((r for r in a["related"] if r["id"] == a.get("same_as")), {})
                same = (
                    f" (same description as {a['same_as']}; "
                    + (f"shares {_members(twin['shared'])} with it)" if twin.get("shared") else "a separate anchor with no shared members)")
                    if a.get("same_as")
                    else ""
                )
                lines.append(
                    f'- {a["id"]} "{a["description"]}"{same}: {a["n_patterns"]} patterns over {a["distinct_scopes"]} distinct scopes;'
                    f" related: {rel or 'none'}"
                )
        lines.append("")
        lines.append("EVIDENCE (statistical findings computed from the uploaded data; cite as [P#]):")
        for it in self.items:
            if it.folded_into:  # restated on its host's scale line
                continue
            s = it.statistics
            scope = "; ".join(self.reword_condition(c) for c in it.scope) if plain else it.scope_text
            area = AREA_TAG[bool(self.area[it.key])] if it.key in (self.area or {}) else ""
            lines.append(f"[{it.key}] role={it.role}{area} | scope: {scope} | support {s['support']:,} rows ({s['support_fraction']:.1%})")
            lines.append(f"     shifts: {'; '.join(self.reword(t) for t in it.shift_text) or 'no validated median shift'}")
            if it.relationship:
                lines.append(f"     relationship: {self.reword(it.relationship)} (divergence {s['emm_score']:.2f})")
            lines.append(
                f"     validation: {it.validation.replace('; ', ' | ')} | insight weight {s['weight']:.2f}"
                f" | confounders: {', '.join(self.reword(d) for d in s['drivers']) or 'none detected'}"
            )
            spatial = SpatialProvenance.from_dict(it.provenance.get("spatial"))
            if spatial and spatial.cells is not None and spatial.resolution is not None:
                compactness = f", compactness {s['compactness']:.2f}" if s.get("compactness") is not None else ""
                n_eff = f" (effective n {format_value(spatial.n_eff)})" if spatial.n_eff is not None else ""
                lines.append(
                    f"     place: {spatial.cells:,} H3 cells (resolution {spatial.resolution}){compactness}"
                    f" | significance corrected for spatial autocorrelation{n_eff}"
                )
            if it.scale in SCALE_TEXT:
                also = "".join(f"; also at {self._level(f)} [{f.key}]: {self._shift(f, it.target)}" for f in self.items if f.key in it.also)
                lines.append(f"     scale: {SCALE_TEXT[it.scale]}{also}")
            tag = " [scope-disjoint from seeds: no shared condition]" if it.transversal_only else LOCATION_TAG.get(it.located, "")
            lines.append(f"     retrieved via: {it.path_text}{tag}")
        chains, regions = [c for c in self.scale_chains if c["up"] or c["down"]], self._regions()
        if chains or regions:
            lines.append("")
            lines.append("SCALE CHAIN (one phenomenon at finer and coarser levels: a shared anchor and region; then the regions above it):")
            lines.extend(f"- {self._chain_text(c)}" for c in chains)
            lines.extend(regions)
        if self.zones:
            keys = self._keys()
            lines.append("")
            lines.append(
                "ZONES (where the evidence lies on the map: the regions of one level holding most of it; with a direction, only findings shifting that way count):"
            )
            lines.extend(
                f"- Z{n} = {z['id']} (res {z['resolution']}): contains {', '.join(keys[p] for p in z['holds'])}; score {z['score']:.2f};"
                f" {z['patterns']} patterns over {z['datasets']} datasets, {z['themes']} themes"
                for n, z in enumerate(self.zones, start=1)
            )
        if self.notes:
            lines.append("")
            lines.extend(f"NOTE: {n}" for n in self.notes)
        return "\n".join(lines)

    @staticmethod
    def _links(related: dict[str, Any]) -> str:
        """``9 shared members, 1.00; centroid 0.90``: what joins an anchor to one neighbour."""
        links, parts = related["links"], []
        if "CO_OCCURS" in links:
            parts.append(f"{_members(related['shared'], 'shared ')}, {links['CO_OCCURS']:.2f}")
        if "RELATED_TO" in links:
            parts.append(f"centroid {links['RELATED_TO']:.2f}")
        return "; ".join(parts)

    def _keys(self) -> dict[str, str]:
        return {it.pattern_id: it.key for it in self.items}

    @staticmethod
    def _level(it: EvidenceItem) -> str:
        spatial = SpatialProvenance.from_dict(it.provenance.get("spatial"))
        return f"res {spatial.resolution}" if spatial else "table"

    @staticmethod
    def _shift_of(it: EvidenceItem, metric: str) -> dict[str, Any] | None:
        return next((x for x in it.statistics["shifts"] if x["metric"] == metric), None)

    def _shift(self, it: EvidenceItem, metric: str) -> str:
        shift = self._shift_of(it, metric) or it.statistics["shifts"][0]
        return f"{self.reword(humanize(shift['metric']))} {shift['robust_z']:+.2f} sd"

    def _chain_text(self, chain: dict[str, Any]) -> str:
        """``res 7: P1 +1.60 sd -> res 5: P7 +1.10 sd -> region R:3:…``, finest first, on the host's target."""
        by_id = {it.pattern_id: it for it in self.items}
        host = by_id[chain["pattern_id"]]
        steps = [by_id[s["pattern_id"]] for s in reversed(chain["down"])] + [host] + [by_id[s["pattern_id"]] for s in chain["up"]]
        shifts = [self._shift_of(it, host.target) for it in steps]
        parts = [f"{self._level(it)}: {it.key}" + (f" {x['robust_z']:+.2f} sd" if x else "") for it, x in zip(steps, shifts, strict=True)]
        return " -> ".join(parts + [f"region {r['id']}" for r in chain["regions"][:1]])

    def _regions(self) -> list[str]:
        """The regions above the items that hold two or more of them, finest first; a coarser region holding the same items
        as a finer one is left out."""
        keys = self._keys()
        held: dict[str, tuple[dict[str, Any], list[str]]] = {}
        for chain in self.scale_chains:
            for r in chain["regions"]:
                held.setdefault(r["id"], (r, []))[1].append(keys[chain["pattern_id"]])
        shown, seen = [], set()
        for rid in sorted((rid for rid, (_, ks) in held.items() if len(ks) > 1), key=lambda rid: (-int(rid.split(":")[1]), rid)):
            if (members := frozenset(held[rid][1])) not in seen:
                seen.add(members)
                shown.append(rid)
        return [
            f"- {', '.join(held[rid][1])} lie in one region, {rid}: {held[rid][0]['patterns']} patterns over {held[rid][0]['datasets']} datasets,"
            f" {held[rid][0]['themes']} themes"
            for rid in shown
        ]

    def _glossary(self) -> dict[str, str]:
        return {k: v for d in self.datasets for k, v in d.get("glossary", {}).items()}

    def reword(self, text: str) -> str:
        """``text`` with every glossary column name (raw or humanized) replaced by its plain meaning."""
        for col, meaning in sorted(self._glossary().items(), key=lambda kv: -len(kv[0])):
            if "=" not in col:
                text = text.replace(humanize(col), meaning).replace(col, meaning)
        return text

    def reword_condition(self, condition: str) -> str:
        """``attribute=value`` in plain words: the glossary's phrase for the pair, else '<meaning> is <value>'."""
        glossary = self._glossary()
        attribute, _, value = condition.partition("=")
        return glossary.get(condition) or f"{glossary.get(attribute, humanize(attribute))} is {value}"

    def plain_language(self) -> list[tuple[str, str]]:
        """(column, meaning) for the derived columns the evidence names, from the datasets' glossaries."""
        glossary = self._glossary()
        named: list[str] = []
        for it in self.items:
            named += [c.split("=", 1)[0] for c in it.scope] + [s["metric"] for s in it.statistics["shifts"]]
            named += it.statistics["covariance"].get("pair", [])
        return [(n, glossary[n]) for n in dict.fromkeys(named) if n in glossary]

    def summary(self) -> str:
        """The observations, cited: one ``- …`` line each, in Ukrainian around untouched literals (or the
        one line saying the graph holds no such evidence). The narrator frames them as the evidence-only answer.

        The prose of the evidence (``scope_text``, ``shift_text``) is the English LLM serializer, so this
        renders from the numbers: scope as ``attribute=value`` literals, the phenomenon shifts (the first
        ``len(shift_text)`` of the record's shifts, which are sorted by |z|) as ``metric ±z sd``, or the
        correlation pair for a covariance insight.
        """
        lines = []
        for it in self.items:
            if it.folded_into:
                continue
            s = it.statistics
            if it.phenomenon_type == "covariance" and s["covariance"].get("pair"):
                a, b = s["covariance"]["pair"]
                observed = f"кореляція {a} ~ {b}: {s['covariance']['global_corr']:+.2f} загалом → {s['covariance']['local_corr']:+.2f} у підгрупі"
            else:
                observed = "; ".join(
                    f"{x['metric']} {x['robust_z']:+.2f} sd (медіана {format_value(x['local_median'])} проти {format_value(x['global_median'])})"
                    for x in s["shifts"][: len(it.shift_text) or 1][:2]
                )
            tag = " (інший сегмент: без спільної умови із запитом, знайдено через латентну тему)" if it.transversal_only else ""
            tag = tag or (" (знайдено лише за місцем: той самий район)" if it.located else "")
            tag += SCALE_WORDS.get(it.scale, "") + "".join(f" (також [{key}])" for key in it.also)
            tag += AREA_WORDS[bool(self.area[it.key])] if it.key in (self.area or {}) else ""
            lines.append(f"- {', '.join(it.scope)} | {observed} | n={s['support']:,} [{it.key}]{tag}")
        keys = self._keys()
        lines += [f"- зона Z{n}: {''.join(f'[{keys[p]}]' for p in z['holds'])}" for n, z in enumerate(self.zones, start=1)]
        if self.area is not None:
            lines.append(f"- у виділеній зоні: {''.join(f'[{k}]' for k, v in self.area.items() if v) or 'жодного свідчення'}")
        if not self.items:
            lines.append("- У графі немає відповідних свідчень.")
        return "\n".join(lines)


def _members(n: int, kind: str = "") -> str:
    return f"{n} {kind}member{'' if n == 1 else 's'}"


def _path_text(path: list[PathStep], role: str) -> str:
    if role == "seed" or not path:
        return "seed (matched the question)"
    parts = [path[0].source]
    for st in path:
        arrow = f"<-{st.edge_type}({st.weight:.2f})-" if st.reverse else f"-{st.edge_type}({st.weight:.2f})->"
        parts.append(f"{arrow} {st.target}")
    return " ".join(parts)


def _add_vertical(items: list[EvidenceItem], graph: DualGraph, item: Any) -> list[dict[str, Any]]:
    """The scale chains of the walk's geo items: each item gets its scale label, the vertical patterns join as items
    (``vertical_up`` | ``vertical_down``), and a level of the same upload that corresponds all the way is folded into
    the item it restates, so two levels of one mechanism never stand as two facts. The host is the item the walk ranked
    first; a seed is never folded (it matched the question), and a host with a finer level folded in is regional too."""
    by_id = {it.pattern_id: it for it in items}
    hosts: set[str] = set()
    chains = []
    for chain in scale_chain(graph, [it.pattern_id for it in items]):
        host = by_id[chain.pattern_id]
        if host.folded_into:
            continue
        hosts.add(chain.pattern_id)
        host.scale = chain.scale
        upload = upload_of(graph, chain.pattern_id)
        for steps in (chain.up, chain.down):
            agrees = True
            for step in steps:
                other = by_id.get(step.pattern_id)
                if other is None:
                    path_text = f"vertical {step.direction} from {by_id[step.origin].key} through {step.attractor} in {step.region}"
                    other = by_id[step.pattern_id] = item(f"P{len(items) + 1}", step.pattern_id, f"vertical_{step.direction}", [], path_text)
                    items.append(other)
                agrees = agrees and step.relation == "corresponds"
                if agrees and upload is not None and upload_of(graph, step.pattern_id) == upload:
                    host.also.append(other.key)
                    if step.direction == "down" and host.scale == "insufficient":  # a finer level holds: the group is regional
                        host.scale = "regional"
                    if step.pattern_id not in hosts and other.role != "seed":
                        other.folded_into = other.folded_into or host.key
        chains.append(_chain_dict(chain, graph))
    return chains


def _zones(query: ParsedQuery, items: list[EvidenceItem], graph: DualGraph, config: QueryConfig) -> list[dict[str, Any]]:
    """The ZONES section, only for a question without scope: its geo items that answer it, i.e. that shift an asked
    target (a phenomenon shift, as the prompt shows them) the asked way, or for a question about a relationship whose
    material correlation change involves an asked target; without a target every geo item counts."""

    def answers(pid: str) -> bool:
        ins = graph.insight(pid)
        if query.covariance:
            return config.thresholds.has_material_covariance(ins) and bool(set(ins.covariance["pair"]) & set(query.targets or ins.covariance["pair"]))
        shifts = config.thresholds.phenomenon_shifts(ins)
        return not query.targets or any(s.metric in query.targets and s.robust_z * (query.direction or s.robust_z) > 0 for s in shifts)

    if query.conditions:
        return []
    geo = [it.pattern_id for it in items if not it.folded_into and resolution(graph, it.pattern_id) is not None]
    return rank_zones(graph, geo, [pid for pid in geo if answers(pid)])


def _chain_dict(chain: ScaleChain, graph: DualGraph) -> dict[str, Any]:
    return asdict(chain) | {"regions": [{"id": r, **picture(graph, r)} for r in chain.regions]}


def build_evidence(
    query: ParsedQuery, result: TraversalResult, graph: DualGraph, config: QueryConfig, place_scores: dict[str, float] | None = None
) -> Evidence:
    items: list[EvidenceItem] = []
    chosen, forged = [], []  # the first EVIDENCE_MAX_PATTERNS intact patterns; the tampered ones ranked among them
    for r in result.patterns:
        if len(chosen) == EVIDENCE_MAX_PATTERNS:
            break
        (forged if graph.nodes[r.node_id]["props"].get("integrity") == "mismatch" else chosen).append(r)

    def item(key: str, pid: str, role: str, path: list[PathStep], path_text: str, transversal_only: bool = False, located: str = "") -> EvidenceItem:
        node = graph.nodes[pid]
        ins = graph.insight(pid)
        material = config.thresholds.has_material_covariance(ins)
        acts = [
            {"attractor": e["target"], "alignment": round(e["weight"], 3), "label": graph.nodes[e["target"]]["label"]}
            for e, _ in graph.incident(pid, ["ACTIVATES"])
            if e["source"] == pid
        ]
        return EvidenceItem(
            key=key,
            pattern_id=pid,
            role=role,
            headline=node["label"],
            scope=[c.expr for c in ins.conditions],
            target=ins.target,
            phenomenon_type=ins.phenomenon_type,
            statistics={
                "support": ins.support,
                "support_fraction": ins.support_fraction,
                "baseline": ins.baseline,
                "local": ins.local,
                "effect_size": ins.effect_size,
                "shifts": [asdict(s) for s in ins.shifts],
                "emm_score": ins.emm_score,
                "stability": ins.stability,
                "p_value": ins.p_value,
                "p_adjusted": ins.p_adjusted,
                "weight": ins.weight,
                "drivers": list(ins.drivers),
                "covariance": ins.covariance,
                "compactness": node["props"].get("compactness"),
            },
            scope_text=describe_scope(ins.conditions),
            shift_text=[describe_shift(s) for s in config.thresholds.phenomenon_shifts(ins)],
            relationship=describe_covariance(ins.covariance) if material else "",
            path=[asdict(st) for st in path],
            path_text=path_text,
            validation=describe_validation(ins, config.thresholds)
            + ("; integrity unverified" if node["props"].get("integrity") == "unverified" else ""),
            attractors=acts,
            transversal_only=transversal_only,
            located=located,
            credibility=node["props"].get("credibility"),
            credibility_basis=node["props"].get("credibility_basis", {}),
            provenance={**ins.provenance, "pattern_id": pid, "expression": ins.expression, "dataset_id": ins.dataset_id, "batch_id": ins.batch_id},
        )

    for idx, r in enumerate(chosen, start=1):
        items.append(item(f"P{idx}", r.node_id, r.route, r.path, _path_text(r.path, r.route), r.transversal_only, r.located))
    chains = _add_vertical(items, graph, item)

    attractors, described = [], {}
    for a in result.attractors:
        node = graph.nodes[a.node_id]
        related: dict[str, dict[str, Any]] = {}  # one entry per neighbour, with every edge that joins them
        for e, other in graph.incident(a.node_id, LATENT):
            entry = related.setdefault(other, {"id": other, "links": {}})
            entry["links"][e["type"]] = round(e["weight"], 3)
            if e["type"] == "CO_OCCURS":
                entry["shared"] = e["props"]["shared"]  # the weight is a capped sum: the count says how many
        description = describe_components(node["props"]["signature"]) or node["label"]
        attractors.append(
            {
                "id": a.node_id,
                "label": node["label"],
                "description": description,
                "n_patterns": node["props"]["n_patterns"],
                "distinct_scopes": node["props"]["distinct_scopes"],
                # members shared first (most first), then centroid neighbours by cosine: two scales never compared
                "related": sorted(related.values(), key=lambda r: (-r.get("shared", 0), -r["links"].get("RELATED_TO", 0.0), r["id"])),
                "signature": node["props"]["signature"],
            }
            # existing anchors never merge with each other (docs/05_latent_anchors.md §5.8)
            | ({"same_as": described[description]} if description in described else {})
        )
        described.setdefault(description, a.node_id)

    ds_ids = sorted({it.provenance["dataset_id"] for it in items})
    datasets = []
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for ds in ds_ids:
        dnode = graph.nodes.get(dataset_node_id(ds), {"props": {}})
        batch = next((it.provenance["batch_id"] for it in items if it.provenance["dataset_id"] == ds), None)
        props = dnode["props"]
        datasets.append(
            {"dataset_id": ds, "filename": props.get("filename"), "rows": props.get("rows"), "batch_id": batch}
            | {k: props[k] for k in ("source", "reliability") if k in props}
        )
        if dnode["props"].get("glossary"):
            datasets[-1]["glossary"] = dnode["props"]["glossary"]
    for it in items:  # baselines of the metrics the prompt actually mentions
        ins = graph.insight(it.pattern_id)
        named = [s.metric for s in config.thresholds.phenomenon_shifts(ins)]
        if config.thresholds.has_material_covariance(ins):
            named += ins.covariance["pair"]
        for metric in named:
            nid = metric_node_id(ins.dataset_id, metric)
            if nid in graph.nodes and (ins.dataset_id, metric) not in metrics:
                mp = graph.nodes[nid]["props"]
                metrics[(ins.dataset_id, metric)] = {
                    "metric": metric,
                    "dataset_id": ins.dataset_id,
                    "global_median": mp["global_median"],
                    "global_mad": mp["global_mad"],
                } | {k: mp[k] for k in ("indicator", "unit") if k in mp}
    notes = []
    located = [it.key for it in items if it.located]
    if located:
        notes.append(
            f"{', '.join(located)} were reached only by location (the same area of the map, no shared condition with the seeds);"
            " being in the same area does not show the findings are related."
        )
    disjoint = [it.key for it in items if it.transversal_only]
    if disjoint:
        notes.append(
            f"{', '.join(disjoint)} share no scope condition with the seeds; they were reached through "
            "latent anchors (the same phenomenon in a different part of the data)."
        )
    unconfirmed = [it.key for it in items if it.scale == "not_found_coarser"]
    if unconfirmed:
        notes.append(
            f"{', '.join(unconfirmed)} were not found at the coarser level of their upload: that is missing evidence at that level,"
            " not proof the effect is local."
        )
    notes += [f"{r.node_id} excluded: integrity check failed (its statistics no longer match the record digest)." for r in forged]
    if not items and not forged:
        notes.append("No pattern in the graph matched the question.")
    return Evidence(
        query=query.text,
        parsed=query.to_dict(),
        seed_patterns=[it.key for it in items if it.role == "seed"],
        items=items,
        attractors=attractors,
        paths=[it.path_text for it in items],
        metrics=list(metrics.values()),
        datasets=datasets,
        provenance=[{"key": it.key, **it.provenance} for it in items],
        notes=notes,
        scale_chains=chains,
        zones=_zones(query, items, graph, config),
        area=None
        if place_scores is None
        else {it.key: place_scores[it.pattern_id] for it in items if not it.folded_into and it.pattern_id in place_scores},
    )
