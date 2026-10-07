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

from dataclasses import asdict, dataclass, field
from typing import Any

from graph_query_engine.config import QueryConfig
from graph_query_engine.graph import DualGraph, describe_components
from graph_query_engine.question import ParsedQuery
from graph_query_engine.traversal import LATENT, PathStep, TraversalResult
from insight_contracts import dataset_node_id, metric_node_id
from insight_contracts.text import (
    describe_covariance,
    describe_scope,
    describe_shift,
    describe_validation,
    format_value,
    humanize,
)

LINK_NOTE = {"CO_OCCURS": ", shared members"}  # how the prompt tells a co-occurrence link from a centroid neighbour


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
                rel = ", ".join(f"{r['id']} ({r['weight']:.2f}{LINK_NOTE.get(r['type'], '')})" for r in a["related"]) or "none"
                lines.append(
                    f'- {a["id"]} "{a["description"]}": {a["n_patterns"]} patterns over {a["distinct_scopes"]} distinct scopes; related: {rel}'
                )
        lines.append("")
        lines.append("EVIDENCE (verified statistical observations; cite as [P#]):")
        for it in self.items:
            s = it.statistics
            lines.append(f"[{it.key}] role={it.role} | scope: {it.scope_text} | support {s['support']:,} rows ({s['support_fraction']:.1%})")
            lines.append(f"     shifts: {'; '.join(it.shift_text) or 'no validated median shift'}")
            if it.relationship:
                lines.append(f"     relationship: {it.relationship} (divergence {s['emm_score']:.2f})")
            lines.append(
                f"     validation: {it.validation.replace('; ', ' | ')} | insight weight {s['weight']:.2f}"
                f" | confounders: {', '.join(s['drivers']) or 'none detected'}"
            )
            lines.append(f"     retrieved via: {it.path_text}{' [scope-disjoint from seeds: no shared condition]' if it.transversal_only else ''}")
        if self.notes:
            lines.append("")
            lines.extend(f"NOTE: {n}" for n in self.notes)
        return "\n".join(lines)

    def summary(self) -> str:
        """The verified observations, cited: one ``- …`` line each, in Ukrainian around untouched literals (or the
        one line saying the graph holds no such evidence). The narrator frames them as the evidence-only answer.

        The prose of the evidence (``scope_text``, ``shift_text``) is the English LLM serializer, so this
        renders from the numbers: scope as ``attribute=value`` literals, the phenomenon shifts (the first
        ``len(shift_text)`` of the record's shifts, which are sorted by |z|) as ``metric ±z sd``, or the
        correlation pair for a covariance insight.
        """
        lines = []
        for it in self.items:
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
            lines.append(f"- {', '.join(it.scope)} | {observed} | n={s['support']:,} [{it.key}]{tag}")
        if not self.items:
            lines.append("- У графі немає відповідних свідчень.")
        return "\n".join(lines)


def _path_text(path: list[PathStep], role: str) -> str:
    if role == "seed" or not path:
        return "seed (matched the question)"
    parts = [path[0].source]
    for st in path:
        arrow = f"<-{st.edge_type}({st.weight:.2f})-" if st.reverse else f"-{st.edge_type}({st.weight:.2f})->"
        parts.append(f"{arrow} {st.target}")
    return " ".join(parts)


def build_evidence(query: ParsedQuery, result: TraversalResult, graph: DualGraph, config: QueryConfig) -> Evidence:
    items: list[EvidenceItem] = []
    chosen = result.patterns[: config.evidence_max_patterns]
    for idx, r in enumerate(chosen, start=1):
        node = graph.nodes[r.node_id]
        ins = graph.insight(r.node_id)
        material = config.thresholds.has_material_covariance(ins)
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
                shift_text=[describe_shift(s) for s in config.thresholds.phenomenon_shifts(ins)],
                relationship=describe_covariance(ins.covariance) if material else "",
                path=[asdict(st) for st in r.path],
                path_text=_path_text(r.path, r.route),
                validation=describe_validation(ins, config.thresholds),
                attractors=acts,
                transversal_only=r.transversal_only,
                provenance={
                    **ins.provenance,
                    "pattern_id": r.node_id,
                    "expression": ins.expression,
                    "dataset_id": ins.dataset_id,
                    "batch_id": ins.batch_id,
                },
            )
        )

    attractors = []
    for a in result.attractors:
        node = graph.nodes[a.node_id]
        related = [{"id": other, "weight": round(e["weight"], 3), "type": e["type"]} for e, other in graph.incident(a.node_id, LATENT)]
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
