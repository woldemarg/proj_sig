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
import re
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any

from graph_query_engine.config import QueryConfig
from graph_query_engine.graph import DualGraph, band_counts, describe_components, insight_reading
from graph_query_engine.question import ParsedQuery
from graph_query_engine.traversal import LATENT, PathStep, Retrieved, TraversalResult
from graph_query_engine.vertical import ScaleChain, picture, rank_zones, restating, scale_chain
from insight_contracts import SpatialProvenance, dataset_node_id, metric_node_id, wording
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
WEAKEST_FIRST = (5, 6, 2, 1)
PLAIN_HEAD = {  # the PLAIN LANGUAGE heading: a geo item's rows are map cells, a table's are its own
    True: "rows are map cells built from point events; a subgroup is a set of map cells",
    False: "column and category meanings declared for this dataset",
}
CONTEXT_VALUE_MAX = 300  # characters of one DATASET CONTEXT value, then "…"
CONTEXT_MAX = 2000  # characters of the DATASET CONTEXT entries; those past it are counted, not shown


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
    credibility: int | None = None  # 1 | 2 | 5 | 6 (docs/06_graph_and_storage.md §6.2); None: not in the snapshot
    credibility_basis: dict[str, Any] = field(default_factory=dict)  # rule, sources, patterns, comparability
    trust: str = ""  # its trust: line, once the graph has a declared source or two datasets (docs/07_question_answering.md §7.4)
    reliability: str = ""  # its dataset's A-F code, with the trust line
    reading: dict[str, str] = field(default_factory=dict)  # {"where", "what"} in Ukrainian (graph.insight_reading)


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
    excluded: list[str] = field(default_factory=list)  # tampered patterns the walk ranked within its budget, left out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt(self) -> str:
        """The evidence as the LLM reads it: its sections in the order of docs/07_question_answering.md §7.4."""
        plain = self.plain_language()
        sections = (
            self._head(),
            self._dataset_lines(),
            self._plain_lines(plain),
            self._context_lines(),
            self._baseline_lines(),
            self._anchor_lines(),
            self._evidence_lines(plain),
            self._chain_lines(),
            self._zone_lines(),
            self._note_lines(),
        )
        return "\n".join(line for section in sections for line in section)

    def _head(self) -> list[str]:
        """QUESTION, PARSED, the asked AREA and the UNITS."""
        p = self.parsed
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
        return lines

    def _dataset_lines(self) -> list[str]:
        lines = ["", "DATASETS:"]
        for d in self.datasets:
            trust = f" source={json.dumps(d['source'], ensure_ascii=False)}" if d.get("source") else ""  # quoted: free text
            if trust or d.get("reliability", "F") != "F":  # an undeclared dataset adds nothing
                trust += f" reliability={d['reliability']} ({RELIABILITY.get(d['reliability'], 'unrecognised code')})"
            lines.append(f"- {d['dataset_id']} file={d['filename']} batch={d['batch_id']} rows={d.get('rows')}{trust}")
        return lines

    def _plain_lines(self, plain: list[tuple[str, str]]) -> list[str]:
        if not plain:
            return []
        kinds = {bool(it.provenance.get("spatial")) for it in self.items}
        head = "; ".join(PLAIN_HEAD[geo] for geo in (True, False) if geo in kinds)
        return ["", f"PLAIN LANGUAGE ({head}):", *(f"- {meaning}" for _, meaning in plain)]

    def _context_lines(self) -> list[str]:
        """DATASET CONTEXT: each metadata document's dataset description and, for every column the evidence names (seed
        items' first, across datasets; a geo cell's derivation read as its source column, ``wording.source_column``), its
        assistant context or description, both JSON-quoted (a document's instruction stays a quoted value), clipped at
        ``CONTEXT_VALUE_MAX``; the entries past ``CONTEXT_MAX`` characters are counted in ``omitted_columns``."""
        about: list[str] = []
        notes: list[tuple[bool, str]] = []  # (not a seed's, entry)
        for d in self.datasets:
            if not (doc := d.get("metadata_doc")):
                continue
            if text := doc["dataset"].get("description"):
                about.append(f"- dataset {d['dataset_id']}: {_quoted(text)}")
            for it in (it for it in self.items if it.provenance["dataset_id"] == d["dataset_id"]):
                for col in filter(None, (wording.source_column(n, doc["columns"]) for n in _named(it))):
                    if text := doc["columns"][col].get("assistant_context") or doc["columns"][col].get("description"):
                        notes.append((it.role != "seed", f"- {json.dumps(col, ensure_ascii=False)}: {_quoted(text)}"))
        columns = list(dict.fromkeys(entry for _, entry in sorted(notes, key=lambda n: n[0])))  # stable: seeds' first, each once
        if not about and not columns:
            return []
        shown, used = [], 0
        for entry in about + columns:
            if used + len(entry) + 1 > CONTEXT_MAX:
                break
            shown.append(entry)
            used += len(entry) + 1
        omitted = len(about) + len(columns) - len(shown)
        return [
            "",
            "DATASET CONTEXT (declared by the uploader; describes columns, asserts nothing about the findings):",
            *shown,
            *([f"- omitted_columns: {omitted}"] if omitted else []),
        ]

    def _baseline_lines(self) -> list[str]:
        lines = ["", "METRIC BASELINES (whole dataset):"]
        for m in self.metrics:
            parts = [f"{k} {json.dumps(m[k], ensure_ascii=False)}" for k in ("indicator", "unit") if k in m]
            meaning = f" ({', '.join(parts)})" if parts else ""
            name = humanize(m["metric"]) if parts else self.reword(humanize(m["metric"]), m["dataset_id"])  # the indicator already says it
            lines.append(f"- {name}{meaning}: median {format_value(m['global_median'])}, MAD {format_value(m['global_mad'])}")
        return lines

    def _anchor_lines(self) -> list[str]:
        if not self.attractors:
            return []
        lines = ["", "LATENT ANCHORS VISITED (recurring phenomena learned across patterns):"]
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
        return lines

    def _evidence_lines(self, plain: list[tuple[str, str]]) -> list[str]:
        trusted = any(it.trust for it in self.items)
        lines = [
            "",
            f"EVIDENCE (statistical findings computed from the uploaded data;{' trust is given per item;' if trusted else ''} cite as [P#]):",
        ]
        for it in self.items:
            if not it.folded_into:  # a folded item is restated on its host's scale line
                lines += self._item_lines(it, plain)
        return lines

    def _item_lines(self, it: EvidenceItem, plain: list[tuple[str, str]]) -> list[str]:
        s, ds = it.statistics, it.provenance["dataset_id"]
        scope = "; ".join(self.reword_condition(c, ds) for c in it.scope) if plain else it.scope_text
        area = AREA_TAG[bool(self.area[it.key])] if it.key in (self.area or {}) else ""
        lines = [f"[{it.key}] role={it.role}{area} | scope: {scope} | support {s['support']:,} rows ({s['support_fraction']:.1%})"]
        lines.append(f"     shifts: {'; '.join(self.reword(t, ds) for t in it.shift_text) or 'no validated median shift'}")
        if it.relationship:
            lines.append(f"     relationship: {self.reword(it.relationship, ds)} (divergence {s['emm_score']:.2f})")
        lines.append(
            f"     validation: {it.validation.replace('; ', ' | ')} | insight weight {s['weight']:.2f}"
            f" | confounders: {', '.join(self.reword(d, ds) for d in s['drivers']) or 'none detected'}"
        )
        if it.trust:
            lines.append(f"     trust: {it.trust}")
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
        return lines

    def _chain_lines(self) -> list[str]:
        chains, regions = [c for c in self.scale_chains if c["up"] or c["down"]], self._regions()
        if not (chains or regions):
            return []
        head = "SCALE CHAIN (one phenomenon at finer and coarser levels: a shared anchor and region; then the regions above it):"
        return ["", head, *(f"- {self._chain_text(c)}" for c in chains), *regions]

    def _zone_lines(self) -> list[str]:
        if not self.zones:
            return []
        keys = self._keys()
        return [
            "",
            "ZONES (where the evidence lies on the map: the regions of one level holding most of it; with a direction, only findings shifting that way count):",
            *(
                f"- Z{n} = {z['id']} (res {z['resolution']}): contains {', '.join(keys[p] for p in z['holds'])}; score {z['score']:.2f};"
                f" {z['patterns']} patterns over {z['datasets']} datasets, {z['themes']} themes{self._weakest(z)}"
                for n, z in enumerate(self.zones, start=1)
            ),
        ]

    def _note_lines(self) -> list[str]:
        return ["", *(f"NOTE: {n}" for n in self.notes)] if self.notes else []

    @staticmethod
    def _links(related: dict[str, Any]) -> str:
        """``9 shared members, 1.00; centroid 0.90``: what joins an anchor to one neighbour."""
        links, parts = related["links"], []
        if "CO_OCCURS" in links:
            parts.append(f"{_members(related['shared'], 'shared ')}, {links['CO_OCCURS']:.2f}")
        if "RELATED_TO" in links:
            parts.append(f"centroid {links['RELATED_TO']:.2f}")
        return "; ".join(parts)

    def _weakest(self, zone: dict[str, Any]) -> str:
        """``; weakest credibility 6``: the least confirmed finding of a zone, with the trust lines."""
        levels = [it.credibility for it in self.items if it.pattern_id in zone["holds"] and it.trust and it.credibility is not None]
        if not levels:
            return ""
        return f"; weakest credibility {min(levels, key=lambda lv: WEAKEST_FIRST.index(lv) if lv in WEAKEST_FIRST else -1)}"  # unknown first

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
        return f"{self.reword(humanize(shift['metric']), it.provenance['dataset_id'])} {shift['robust_z']:+.2f} sd"

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

    @cached_property
    def _glossaries(self) -> dict[str, dict[str, str]]:  # read by every reword of one rendering: built once
        """Each dataset's glossary by id, never another's: its derived English one (``glossary``) under the declared one
        (``user_glossary``); a declared source column read onto every cell column derived from it, as the readings read
        it (``wording.column_words``), and a declared ``column=value`` onto the cell's mode column."""
        out = {}
        for d in self.datasets:
            derived, user = d.get("glossary", {}), d.get("user_glossary", {})
            cells = {
                c: wording.column_words(c, user, True)
                for c in derived
                if "=" not in c and not c.endswith("_band") and wording.source_column(c, user) in user
            }
            modes = {k.replace("=", "_mode=", 1): v for k, v in user.items() if "=" in k and k.split("=", 1)[0] + "_mode" in derived}
            out[d["dataset_id"]] = derived | cells | user | modes
        return out

    @cached_property
    def _names(self) -> dict[str, tuple[re.Pattern[str], dict[str, str]]]:
        """Per dataset, one pattern over every glossary column name, raw and humanized, longest first, matched whole: never
        inside a word, and never inside a longer name its evidence uses, which stands for itself (``rate`` leaves ``return rate``)."""
        out = {}
        for ds, glossary in self._glossaries.items():
            meanings = {n: meaning for col, meaning in glossary.items() if "=" not in col for n in (col, humanize(col))}
            if meanings:
                names = {n: n for it in self.items if it.provenance["dataset_id"] == ds for c in _named(it) for n in (c, humanize(c))} | meanings
                out[ds] = re.compile(rf"(?<![\w])(?:{'|'.join(map(re.escape, sorted(names, key=len, reverse=True)))})(?![\w])"), names
        return out

    def reword(self, text: str, dataset_id: str) -> str:
        """``text`` of ``dataset_id`` with every column name of its glossary (raw or humanized) replaced by its plain meaning,
        in one pass."""
        if dataset_id not in self._names:
            return text
        pattern, names = self._names[dataset_id]
        return pattern.sub(lambda m: names[m[0]], text)

    def reword_condition(self, condition: str, dataset_id: str) -> str:
        """``attribute=value`` of ``dataset_id`` in plain words: its glossary's phrase for the pair, else '<meaning> is <value>',
        a quantile band's value with its place among the dataset's bands (a quarter only of four)."""
        glossary = self._glossaries.get(dataset_id, {})
        attribute, _, value = condition.partition("=")
        words = glossary.get(condition) or f"{glossary.get(attribute, humanize(attribute))} is {value}"
        bands = next((d.get("bands", {}).get(attribute) for d in self.datasets if d["dataset_id"] == dataset_id), None)
        if condition in glossary or not bands or not re.fullmatch(r"q\d+", value):
            return words
        return f"{words} ({f'quarter {value[1:]} of 4' if bands == 4 else f'band {value[1:]} of {bands}'})"

    def plain_language(self) -> list[tuple[str, str]]:
        """(column or declared ``column=value``, meaning) for what each item names that its dataset's glossary holds, each
        once; a derived pair's phrase (a geo cell's class or band) is on its scope line already."""
        derived = {d["dataset_id"]: d.get("glossary", {}) for d in self.datasets}
        out: dict[tuple[str, str], None] = {}
        for it in self.items:
            ds = it.provenance["dataset_id"]
            glossary = self._glossaries.get(ds, {})
            out |= {(n, glossary[n]): None for n in (*_named(it), *(c for c in it.scope if c not in derived.get(ds, {}))) if n in glossary}
        return list(out)

    def summary(self) -> str:
        """The observations, cited: one ``- …`` line each, the item's reading (``wording``) in Ukrainian around
        untouched literals, its rows and tags, or the one line saying the graph holds no such evidence. Trust, once the
        graph has it, is one line when every item shares it and a tag on each line when they differ. The narrator
        frames these lines as the evidence-only answer."""
        shown = [it for it in self.items if not it.folded_into]
        trust = {(it.reliability, it.credibility) for it in shown if it.trust}
        lines = []
        for it in shown:
            reached = "transversal_only" if it.transversal_only else "located" if it.located else ""
            tag = f" ({wording.name(reached)}: {wording.TERMS[reached][1]})" if reached else ""
            tag += f" ({wording.name('scale_' + it.scale)})" if it.scale in SCALE_TEXT else ""
            tag += "".join(f" (також [{key}])" for key in it.also)
            tag += AREA_WORDS[bool(self.area[it.key])] if it.key in (self.area or {}) else ""
            tag += f" ({_trust_words(it.reliability, it.credibility)})" if it.trust and len(trust) > 1 else ""
            n = it.statistics["support"]
            lines.append(f"- {wording.sentence(it.reading)}; {n:,} {wording.plural(n, 'рядок', 'рядки', 'рядків')} [{it.key}]{tag}")
        if len(trust) == 1:
            lines.append(f"- довіра{' всіх свідчень' if len(shown) > 1 else ''}: {_trust_words(*trust.pop())}")
        keys = self._keys()
        lines += [
            f"- регіон Z{n} (рівень {z['resolution']}): {''.join(f'[{keys[p]}]' for p in z['holds'])}" for n, z in enumerate(self.zones, start=1)
        ]
        if self.area is not None:
            lines.append(f"- у виділеній зоні: {''.join(f'[{k}]' for k, v in self.area.items() if v) or 'жодного свідчення'}")
        if self.excluded:  # what the NOTEs say of integrity, for the answer without the model
            lines.append(f"- виключено, перевірку цілісності не пройдено (статистика не збігається з дайджестом запису): {', '.join(self.excluded)}")
        lines += [
            f"- метадані датасету {d['dataset_id']} не пройшли перевірку цілісності: його джерело, надійність, показники, глосарій і документ проігноровано"
            for d in self.datasets
            if d.get("integrity") == "mismatch"
        ]
        if not self.items:
            lines.append("- У графі немає відповідних свідчень.")
        return "\n".join(lines)


def _trust_words(reliability: str, credibility: int | None) -> str:
    """An item's trust in the summary's words: its source's code and rating, and whether another source confirms it."""
    rated = wording.name(f"reliability_{reliability}") if f"reliability_{reliability}" in wording.TERMS else "невідомий код"
    source = f"джерело {reliability} ({rated})" if reliability else "джерело невідоме"
    return source + (f", {wording.name(f'credibility_{credibility}')}" if f"credibility_{credibility}" in wording.TERMS else "")


def _credibility_text(level: int, basis: dict[str, Any], keys: dict[str, str]) -> str:
    """``credibility 1 (confirmed by another source at the same place: "b"; [P3])``: the level and its basis."""
    place = basis.get("rule") == "place"
    words = {
        1: "confirmed by another source" + (" at the same place" if place else ""),
        2: "not confirmed; another dataset has " + ("events at this place" if place else "the same metric"),
        5: "contradicted by another source" + (" at the same place" if place else ""),
        6: "no independent source covers this",
    }.get(level, "unrecognised level")

    def cited(pids: list[str]) -> str:
        shown, rest = [f"[{keys[p]}]" for p in pids if p in keys], sum(p not in keys for p in pids)
        return ", ".join(shown + ([f"{rest} finding{'s' if rest > 1 else ''} not in this evidence"] if rest else []))

    detail = "; ".join(
        x
        for x in (
            ", ".join(json.dumps(s, ensure_ascii=False) for s in basis.get("sources", [])),
            cited(basis.get("patterns", [])),
            f"confirmed by {cited(basis['confirmed_by'])}" if basis.get("confirmed_by") else "",
        )
        if x
    )
    return f"credibility {level} ({words}{': ' + detail if detail else ''})"


def _trust_applies(graph: DualGraph) -> bool:
    """Whether the items carry a trust line (docs/07_question_answering.md §7.4): once the graph has a declared source or
    reliability, a dataset whose metadata failed its check, or two uploads (the levels of one upload are one); otherwise
    the evidence stays as it was without sources."""
    datasets = [d["props"] for d in graph.of_kind("Dataset")]
    uploads = {d.get("upload_id") or d["dataset_id"] for d in datasets}
    declared = any(d.get("source") or d.get("reliability", "F") != "F" or d.get("integrity") == "mismatch" for d in datasets)
    return len(uploads) >= 2 or declared


def _add_trust(items: list[EvidenceItem], graph: DualGraph) -> None:
    """Each item's trust line: source reliability | credibility | integrity (when ``_trust_applies``)."""
    keys = {it.pattern_id: it.key for it in items}
    for it in items:
        props = graph.nodes.get(dataset_node_id(it.provenance["dataset_id"]), {"props": {}})["props"]
        forged = props.get("integrity") == "mismatch"
        it.reliability = "" if forged else props.get("reliability", "F")
        mark = ("" if props.get("source") else ", undeclared") + (", metadata unverified" if props.get("integrity") == "unverified" else "")
        parts = [
            "source unknown (its dataset's metadata failed the integrity check)"
            if forged
            else f"source {it.reliability} ({RELIABILITY.get(it.reliability, 'unrecognised code')}){mark}"
        ]
        if it.credibility is not None:
            parts.append(_credibility_text(it.credibility, it.credibility_basis, keys))
        parts.append(f"integrity {graph.nodes[it.pattern_id]['props'].get('integrity', 'unverified')}")
        it.trust = " | ".join(parts)


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


def _add_vertical(items: list[EvidenceItem], graph: DualGraph, config: QueryConfig, trusted: bool) -> list[dict[str, Any]]:
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
        restates = restating(graph, chain)
        for steps in (chain.up, chain.down):
            for step in steps:
                other = by_id.get(step.pattern_id)
                if other is None:
                    path_text = f"vertical {step.direction} from {by_id[step.origin].key} through {step.attractor} in {step.region}"
                    key, role = f"P{len(items) + 1}", f"vertical_{step.direction}"
                    other = by_id[step.pattern_id] = _item(graph, config, trusted, key, step.pattern_id, role, [], path_text)
                    items.append(other)
                if step.pattern_id in restates:
                    host.also.append(other.key)
                    if step.direction == "down" and host.scale == "insufficient":  # a finer level holds: the group is regional
                        host.scale = "regional"
                    if step.pattern_id not in hosts and other.role != "seed":
                        other.folded_into = other.folded_into or host.key
        chains.append(chain_dict(chain, graph))
    return chains


def _zones(query: ParsedQuery, items: list[EvidenceItem], graph: DualGraph, config: QueryConfig) -> list[dict[str, Any]]:
    """The ZONES section, only for a question without scope: its placed items (any IN_REGION, a geo pattern's or a base
    one's) that answer it, i.e. that shift an asked target (a phenomenon shift, as the prompt shows them) the asked way,
    or for a question about a relationship whose material correlation change involves an asked target; without a target
    every placed item counts."""

    def answers(pid: str) -> bool:
        ins = graph.insight(pid)
        if query.covariance:
            return config.thresholds.has_material_covariance(ins) and bool(set(ins.covariance["pair"]) & set(query.targets or ins.covariance["pair"]))
        shifts = config.thresholds.phenomenon_shifts(ins)
        return not query.targets or any(s.metric in query.targets and s.robust_z * (query.direction or s.robust_z) > 0 for s in shifts)

    if query.conditions:
        return []
    geo = [it.pattern_id for it in items if not it.folded_into and graph.facts(it.pattern_id).regions]
    return rank_zones(graph, geo, [pid for pid in geo if answers(pid)])


def chain_dict(chain: ScaleChain, graph: DualGraph) -> dict[str, Any]:
    """A scale chain as the view carries it: each region above it with its ``picture``."""
    return asdict(chain) | {"regions": [{"id": r, **picture(graph, r)} for r in chain.regions]}


def _item(
    graph: DualGraph,
    config: QueryConfig,
    trusted: bool,
    key: str,
    pid: str,
    role: str,
    path: list[PathStep],
    path_text: str,
    transversal_only: bool = False,
    located: str = "",
) -> EvidenceItem:
    """One evidence item: the pattern's statistics, its phrases, how it was reached and its trust fields."""
    node = graph.nodes[pid]
    ins = graph.insight(pid)
    material = config.thresholds.has_material_covariance(ins)
    acts = [
        {"attractor": e["target"], "alignment": round(e["weight"], 3), "label": graph.nodes[e["target"]]["label"]}
        for e, _ in graph.incident(pid, ["ACTIVATES"])
        if e["source"] == pid
    ]
    unverified = not trusted and node["props"].get("integrity") == "unverified"  # with a trust line, the line says it
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
        validation=describe_validation(ins, config.thresholds) + ("; integrity unverified" if unverified else ""),
        attractors=acts,
        transversal_only=transversal_only,
        located=located,
        credibility=node["props"].get("credibility"),
        credibility_basis=node["props"].get("credibility_basis", {}),
        provenance={**ins.provenance, "pattern_id": pid, "expression": ins.expression, "dataset_id": ins.dataset_id, "batch_id": ins.batch_id},
        reading=insight_reading(graph, pid, config.thresholds),
    )


def _anchor_section(result: TraversalResult, graph: DualGraph) -> list[dict[str, Any]]:
    """The anchors the walk visited, each with its description and its latent neighbours (``LATENT ANCHORS VISITED``)."""
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
    return attractors


def _named(it: EvidenceItem) -> list[str]:
    """The columns an item names: its scope's attributes, its shifts' metrics, its correlation's pair."""
    return [c.split("=", 1)[0] for c in it.scope] + [s["metric"] for s in it.statistics["shifts"]] + it.statistics["covariance"].get("pair", [])


def _quoted(text: str) -> str:
    """A declared value JSON-quoted, clipped at ``CONTEXT_VALUE_MAX`` characters."""
    return json.dumps(text if len(text) <= CONTEXT_VALUE_MAX else text[: CONTEXT_VALUE_MAX - 1] + "…", ensure_ascii=False)


def _datasets(items: list[EvidenceItem], graph: DualGraph) -> list[dict[str, Any]]:
    """The items' datasets (``DATASETS``): file, rows, batch, the declared trust, the glossaries, the metadata document
    and the band count of each quantile-band column the items name."""
    datasets = []
    for ds in sorted({it.provenance["dataset_id"] for it in items}):
        dnode = graph.nodes.get(dataset_node_id(ds), {"props": {}})
        batch = next((it.provenance["batch_id"] for it in items if it.provenance["dataset_id"] == ds), None)
        props = dnode["props"]
        datasets.append(
            {"dataset_id": ds, "filename": props.get("filename"), "rows": props.get("rows"), "batch_id": batch}
            | {k: props[k] for k in ("source", "reliability", "integrity", "glossary", "user_glossary") if props.get(k)}
        )
        mine = [it for it in items if it.provenance["dataset_id"] == ds]
        if bands := band_counts(graph, ds, {c.split("=", 1)[0] for it in mine for c in it.scope}):
            datasets[-1]["bands"] = bands
        if doc := props.get("metadata_doc"):  # the description and the columns the items name, never the whole document
            named = {wording.source_column(n, doc["columns"]) for it in mine for n in _named(it)}
            datasets[-1]["metadata_doc"] = {"dataset": doc["dataset"], "columns": {c: f for c, f in doc["columns"].items() if c in named}}
    return datasets


def _baselines(items: list[EvidenceItem], graph: DualGraph, config: QueryConfig) -> list[dict[str, Any]]:
    """The whole-dataset median and MAD of every metric the prompt mentions (``METRIC BASELINES``)."""
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for it in items:
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
    return list(metrics.values())


def _notes(items: list[EvidenceItem], datasets: list[dict[str, Any]], forged: list[Retrieved], place_scores: dict[str, float] | None) -> list[str]:
    """The NOTEs: how the items were reached, what is missing, what was excluded or ignored."""
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
    if place_scores == {}:  # a zone, but no geo dataset of its levels to weigh: it places no finding in or out
        notes.append("The asked map area weighs no geo dataset (none at the levels it was asked for), so no finding is placed in or out of it.")
    notes += [f"{r.node_id} excluded: integrity check failed (its statistics no longer match the record digest)." for r in forged]
    notes += [
        f"{d['dataset_id']}: its metadata no longer matches its digest; its source, reliability, indicators, glossary and metadata document are ignored"
        " and its baselines are unchecked."
        for d in datasets
        if d.get("integrity") == "mismatch"
    ]
    if not items and not forged:
        notes.append("No pattern in the graph matched the question.")
    return notes


def build_evidence(
    query: ParsedQuery, result: TraversalResult, graph: DualGraph, config: QueryConfig, place_scores: dict[str, float] | None = None
) -> Evidence:
    """The walk's patterns as citable evidence: the items (with the vertical zoom's and the trust lines), the anchors
    visited, the datasets and their baselines, the notes, the zones or the asked area."""
    chosen, forged = result.patterns[:EVIDENCE_MAX_PATTERNS], result.excluded  # the walk ranks intact patterns only
    trusted = _trust_applies(graph)
    items = [
        _item(graph, config, trusted, f"P{idx}", r.node_id, r.route, r.path, _path_text(r.path, r.route), r.transversal_only, r.located)
        for idx, r in enumerate(chosen, start=1)
    ]
    chains = _add_vertical(items, graph, config, trusted)
    if trusted:
        _add_trust(items, graph)
    datasets = _datasets(items, graph)
    return Evidence(
        query=query.text,
        parsed=query.to_dict(),
        seed_patterns=[it.key for it in items if it.role == "seed"],
        items=items,
        attractors=_anchor_section(result, graph),
        paths=[it.path_text for it in items],
        metrics=_baselines(items, graph, config),
        datasets=datasets,
        provenance=[{"key": it.key, **it.provenance} for it in items],
        notes=_notes(items, datasets, forged, place_scores),
        scale_chains=chains,
        zones=[] if place_scores else _zones(query, items, graph, config),  # an asked area is the question's place: no second one
        area={it.key: place_scores[it.pattern_id] for it in items if not it.folded_into and it.pattern_id in place_scores} if place_scores else None,
        excluded=[r.node_id for r in forged],
    )
