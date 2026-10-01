"""Evidence builder (docs/07_question_answering.md §7.4): traversal result -> structured, citable evidence.

The LLM receives *only* this object's ``to_prompt()`` rendering; ``summary()`` is the
deterministic evidence-only answer used when no LLM answer is available. Both are plain
ASCII built from the readable-text helpers in ``ltir.canonical`` (docs/04_representation.md §4.2): rounded numbers,
p-value buckets, shifts in robust standard deviations, prose scopes. Every item has a
citation key ``[P#]`` that maps back to a Pattern id, its dataset, batch and exact EDA
selector, so answers are traceable to table slices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ltir.canonical import (
    describe_covariance,
    describe_scope,
    describe_shift,
    format_p,
    format_value,
    has_material_covariance,
    humanize,
    phenomenon_shifts,
)
from ltir.config import Config
from ltir.graph import DualGraph, describe_components
from ltir.models import dataset_node_id, metric_node_id
from ltir.query import ParsedQuery
from ltir.traversal import PathStep, TraversalResult


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
    shift_text: list[str]  # phenomenon shifts as phrases (target + |z| >= MIN_COMPONENT_Z)
    relationship: str  # material correlation change as a phrase, or ""
    path: list[dict[str, Any]]
    path_text: str
    attractors: list[dict[str, Any]]
    transversal_only: bool
    provenance: dict[str, Any]


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

    @property
    def key_to_pattern(self) -> dict[str, str]:
        return {i.key: i.pattern_id for i in self.items}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt(self) -> str:
        p = self.parsed
        lines = [f"QUESTION: {self.query}"]
        parsed_bits = []
        if p.get("targets"):
            parsed_bits.append("target=" + ",".join(p["targets"]))
        if p.get("direction"):
            parsed_bits.append("direction=" + ("up" if p["direction"] > 0 else "down"))
        if p.get("conditions"):
            parsed_bits.append("scope=" + ",".join(p["conditions"]))
        lines.append("PARSED: " + (" | ".join(parsed_bits) or "no explicit metric/scope recognised"))
        lines.append(
            "UNITS: shifts are robust standard deviations (sd = median difference scaled by the MAD); medians compare the subgroup with the whole dataset."
        )
        lines.append("")
        lines.append("DATASETS:")
        for d in self.datasets:
            lines.append(f"- {d['dataset_id']} file={d['filename']} batch={d['batch_id']} rows={d.get('rows')}")
        lines.append("")
        lines.append("METRIC BASELINES (whole dataset):")
        for m in self.metrics:
            lines.append(f"- {humanize(m['metric'])}: median {format_value(m['global_median'])}, MAD {format_value(m['global_mad'])}")
        if self.attractors:
            lines.append("")
            lines.append("LATENT ANCHORS VISITED (recurring phenomena learned across patterns):")
            for a in self.attractors:
                rel = ", ".join(f"{r['id']} ({r['weight']:.2f})" for r in a["related"]) or "none"
                lines.append(
                    f'- {a["id"]} "{a["description"]}": {a["n_patterns"]} patterns over {a["distinct_scopes"]} distinct scopes; related: {rel}'
                )
        lines.append("")
        lines.append("EVIDENCE (verified statistical observations; cite as [P#]):")
        for it in self.items:
            s = it.statistics
            lines.append(f"[{it.key}] role={it.role} | scope: {it.scope_text} | support {s['support']:,} rows ({s['support_fraction']:.1%})")
            lines.append(f"     shifts: {'; '.join(it.shift_text) or 'no material median shift'}")
            if it.relationship:
                lines.append(f"     relationship: {it.relationship} (divergence {s['emm_score']:.2f})")
            lines.append(
                f"     validation: bootstrap stability {s['stability']:.2f} | adjusted p {format_p(s['p_adjusted'])} | insight weight {s['weight']:.2f}"
                f" | confounders: {', '.join(s['drivers']) or 'none detected'}"
            )
            lines.append(f"     retrieved via: {it.path_text}{' [scope-disjoint from seeds: no shared condition]' if it.transversal_only else ''}")
        if self.notes:
            lines.append("")
            lines.extend(f"NOTE: {n}" for n in self.notes)
        return "\n".join(lines)

    def summary(self) -> str:
        """Evidence-only answer: the verified observations, cited, without interpretation."""
        lines = ["Observations:"]
        for it in self.items:
            observed = "; ".join(it.shift_text[:2] or [it.relationship])
            tag = " (scope-disjoint from the seed, linked via a latent anchor)" if it.transversal_only else ""
            lines.append(f"- {it.scope_text} | {observed} | n={it.statistics['support']:,} [{it.key}]{tag}")
        if not self.items:
            lines.append("- No matching evidence in the graph.")
        lines.append("Interpretation (hypotheses): not generated (no language-model answer is available).")
        return "\n".join(lines)


def _path_text(path: list[PathStep], role: str) -> str:
    if role == "seed" or not path:
        return "seed (matched the question)"
    parts = [path[0].source]
    for st in path:
        arrow = f"<-{st.edge_type}({st.weight:.2f})-" if st.reverse else f"-{st.edge_type}({st.weight:.2f})->"
        parts.append(f"{arrow} {st.target}")
    return " ".join(parts)


def build_evidence(query: ParsedQuery, result: TraversalResult, graph: DualGraph, config: Config) -> Evidence:
    items: list[EvidenceItem] = []
    chosen = result.patterns[: config.evidence_max_patterns]
    for idx, r in enumerate(chosen, start=1):
        node = graph.nodes[r.node_id]
        ins = graph.insight(r.node_id)
        material = has_material_covariance(ins, config)
        acts = [
            {"attractor": e["target"], "alignment": round(e["weight"], 3), "label": graph.nodes[e["target"]]["label"]}
            for e, _ in graph.incident(r.node_id, ["ACTIVATES"])
            if e["source"] == r.node_id
        ]
        items.append(
            EvidenceItem(
                key=f"P{idx}",
                pattern_id=r.node_id,
                role=r.route,
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
                },
                scope_text=describe_scope(ins.conditions),
                shift_text=[describe_shift(s) for s in phenomenon_shifts(ins, config)],
                relationship=describe_covariance(ins.covariance) if material else "",
                path=[asdict(st) for st in r.path],
                path_text=_path_text(r.path, r.route),
                attractors=acts,
                transversal_only=r.transversal_only,
                provenance={**ins.provenance, "pattern_id": r.node_id},
            )
        )

    attractors = []
    for a in result.attractors:
        node = graph.nodes[a.node_id]
        related = [{"id": other, "weight": round(e["weight"], 3)} for e, other in graph.incident(a.node_id, ["RELATED_TO"])]
        attractors.append(
            {
                "id": a.node_id,
                "label": node["label"],
                "description": describe_components(node["props"]["signature"]) or node["label"],
                "n_patterns": node["props"]["n_patterns"],
                "distinct_scopes": node["props"]["distinct_scopes"],
                "related": related,
                "signature": node["props"]["signature"],
            }
        )

    ds_ids = sorted({it.provenance["dataset_id"] for it in items})
    datasets = []
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for ds in ds_ids:
        dnode = graph.nodes.get(dataset_node_id(ds), {"props": {}})
        batch = next((it.provenance["batch_id"] for it in items if it.provenance["dataset_id"] == ds), None)
        datasets.append({"dataset_id": ds, "filename": dnode["props"].get("filename"), "rows": dnode["props"].get("rows"), "batch_id": batch})
    for r in chosen:  # baselines of the metrics the prompt actually mentions
        ins = graph.insight(r.node_id)
        named = [s.metric for s in phenomenon_shifts(ins, config)]
        if has_material_covariance(ins, config):
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
                }
    notes = []
    disjoint = [it.key for it in items if it.transversal_only]
    if disjoint:
        notes.append(
            f"{', '.join(disjoint)} share no scope condition with the seeds; they were reached through "
            "latent anchors (the same phenomenon in a different part of the data)."
        )
    if not items:
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
    )
