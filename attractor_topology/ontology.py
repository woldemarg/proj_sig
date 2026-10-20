"""Latent ontology over ``attractor_topology.vendor.lac`` (docs/05_latent_anchors.md).

Stage order matches ``v2_orchestrator.main.run_batch`` — cold start |
assign -> EMA update -> orphans -> extract -> soft merge -> absorbed routing |
nearest fallback — with pattern vectors instead of text chunks. Signed-insight
repair (atom sign, intra-extraction merge, alignment floor) is part of
``extract_attractors``.

All vectors live in one frame: the unit composite insight vectors, exactly as the
encoder produces them and the caller stores them (lac's running-mean centering is
not used, so stored rows, centroids and query vectors stay comparable).

Statistical weight enters through **evidence-magnitude encoding**: a pattern
enters the engine as ``x = w * x_hat`` (||x_hat|| = 1, w = insight_weight).
Cosine assignment is scale invariant, so w never changes *which* attractor a
pattern matches; the EMA pull ``(1-a)c + a*w*x_hat`` and the OMP reconstruction
loss (rows scaled by w) make strong insights shape attractors more, and an
attractor's mass (its inertia and its share in the hub guard) is the sum of
its members' w, the same units as the pull.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from attractor_topology.config import TopologyConfig
from attractor_topology.encoder import l2_normalize
from attractor_topology.vendor.lac import observability
from attractor_topology.vendor.lac import ontology_engine as eng
from attractor_topology.vendor.lac.storage import ConceptStore

Activations = list[dict[str, Any]]


@dataclass
class OntologyUpdate:
    activations: Activations
    row_ids: list[int]
    metrics: dict[str, Any] = field(default_factory=dict)
    new_attractors: list[int] = field(default_factory=list)


@dataclass
class _Extraction:
    """What one ingest did to the attractor set (feeds lac ``BatchMetrics``)."""

    extracted: int = 0
    kept: int = 0
    soft_merged: int = 0
    orphaned: int = 0
    adaptive_thresh: float | None = None
    damped: int = 0  # attractors whose EMA step was damped (hub guard)
    max_step: float | None = None  # largest raw centroid move of the batch
    clamped: int = 0  # attractors pulled back by the trust region


class OntologyError(RuntimeError):
    code = "ontology_failure"


def orphan_anchors(members: dict[int, int], links: list[dict[str, Any]], anchors: list[int]) -> list[int]:
    """Anchors to drop once patterns were removed: no remaining member *and* no RELATED_TO link
    (docs/05_latent_anchors.md §5.11). A memberless anchor that is still linked stays: its centroid
    keeps its place in the topology and can receive future insights."""
    linked = {a for e in links for a in (e["source"], e["target"])}
    return [a for a in anchors if not members.get(a) and a not in linked]


def _member_weight(activation: dict[str, Any]) -> float:
    """The insight weight a membership carries; journals written before the field existed give strength / alignment."""
    w = activation.get("insight_weight")
    if w is None:
        w = abs(activation["strength"]) / max(abs(activation["alignment"]), 1e-3)
    return float(np.clip(w, 1e-3, 1.0))


def _recount(activations: Activations, index: dict[int, int], unit: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Mass per attractor (``index``: attractor id -> row), the summed member weights of ``activations``; with ``unit``
    (unit vectors by ``row_id``) also the weight-averaged pull per attractor. An activation naming an attractor the
    state lacks is a journal / state inconsistency and raises."""
    mass = np.zeros(len(index))
    pull = np.zeros((len(index), unit.shape[1])) if unit is not None and len(unit) else None
    for a in activations:
        i, w = index[int(a["attractor_id"])], _member_weight(a)
        mass[i] += w
        if pull is not None:
            pull[i] += w * unit[int(a["row_id"])]
    return mass, pull


def _mass(local: Activations, x_in: np.ndarray, n_attractors: int) -> np.ndarray:
    """The initial mass of freshly extracted attractors: the summed norms (w) of the rows each one activates."""
    members = [{"attractor_id": a["concept_id"], "insight_weight": float(np.linalg.norm(x_in[int(a["chunk_id"])]))} for a in local]
    return _recount(members, {i: i for i in range(n_attractors)})[0]


def _tag(edges: Activations, source: str) -> Activations:
    """Label activations with the lifecycle step that produced them (``reroute`` wins)."""
    return [{**e, "source": "reroute" if e.get("rerouted") else source} for e in edges]


def activation_record(
    pattern: str,
    attractor: int,
    alignment: float,
    insight_weight: float,
    source: str,
    engine_weight: float,
    batch_id: str,
    *,
    row_id: int,
    weak: bool = False,
) -> dict[str, Any]:
    """Contract of one ACTIVATES edge, as the caller records it and draws it in the graph."""
    return {
        "pattern_id": pattern,
        "attractor_id": int(attractor),
        "alignment": float(alignment),
        "strength": float(alignment * insight_weight),
        "insight_weight": float(insight_weight),
        "engine_weight": float(engine_weight),
        "source": source,  # cold_start | assign | omp | absorbed | nearest | reroute
        "weak": weak,  # rerouted below min_activation_alignment (coverage only)
        "batch_id": batch_id,
        "row_id": int(row_id),
    }


@dataclass(frozen=True)
class Attractor:
    id: int
    centroid: np.ndarray  # unit vector in the insight frame
    mass: float  # the summed insight weights of its memberships (the store's inertia)
    created_at: str
    last_updated_seq: int  # batch_seq of its last centroid update


class LatentOntology:
    def __init__(self, config: TopologyConfig, state_dir: Path) -> None:
        self.config = config
        self.state_dir = state_dir
        self.store = ConceptStore.load(state_dir)

    @property
    def attractor_ids(self) -> list[int]:
        return [int(c) for c in self.store.concept_ids]

    def attractors(self) -> list[Attractor]:
        """Every attractor's state (copies): what the graph and the read frame need from the topology."""
        st = self.store
        return [
            Attractor(int(c), st.embeddings[i].copy(), float(st.chunk_counts[i]), st.created_at[i], int(st.last_updated_batch[i]))
            for i, c in enumerate(st.concept_ids)
        ]

    def centroid(self, attractor_id: int) -> np.ndarray:
        return self.store.embeddings[self.store.concept_ids.index(attractor_id)]

    def topology(self) -> list[dict[str, Any]]:
        """Mutual k-NN RELATED_TO edges (lac ``calculate_knn_topology``)."""
        return eng.calculate_knn_topology(self.store, self.config)

    def save(self) -> None:
        self.store.save(self.state_dir)

    def recount_mass(self, activations: Activations) -> None:
        """Every anchor's mass as the summed weights of its memberships in the journal (``activations``: the whole
        journal's). The batch keeps the same sum incrementally; the caller recounts before every batch, so the store's
        inertia is the journal-derived quantity and a state saved under another unit (lac's head counts) needs no mode."""
        self.store.chunk_counts = _recount(activations, {a: i for i, a in enumerate(self.attractor_ids)})[0]

    def forget(self, kept_activations: Activations, kept_vectors: np.ndarray, lost: set[int]) -> list[int]:
        """Dataset deletion. Every surviving anchor's mass is recounted from its surviving members (``row_id``
        already renumbered into ``kept_vectors``); an anchor that ``lost`` a member is also moved to its surviving
        members' weight-averaged direction, so its inertia and position agree again, while an anchor that lost
        nobody keeps the position its history gave it. Orphan anchors are dropped; the row counter restarts at the
        rows kept. A linked anchor left without members keeps its centroid with mass 0. Returns the dropped anchor ids."""
        st = self.store
        n_rows = len(kept_vectors)
        members = Counter(int(a["attractor_id"]) for a in kept_activations)
        # no pattern left: nothing for a topology to be relative to, the knowledge base starts cold again
        dropped = orphan_anchors(members, self.topology(), self.attractor_ids) if n_rows else list(self.attractor_ids)
        st.keep_concepts([a for a in self.attractor_ids if a not in dropped])
        ids = self.attractor_ids  # a fresh list from the store on every access
        mass, pull = _recount(kept_activations, {a: i for i, a in enumerate(ids)}, l2_normalize(kept_vectors) if n_rows else None)
        for i in np.flatnonzero(mass > 0):
            if ids[i] in lost:
                st.embeddings[i] = l2_normalize(pull[i][None])[0].astype(np.float32)
        st.chunk_counts = mass
        st.next_chunk_id = n_rows
        st.clear_orphan_buffer()  # always empty between batches; the ids would be stale anyway
        return dropped

    def ingest(self, vectors: np.ndarray, weights: np.ndarray, pattern_ids: list[str], *, batch_seq: int, batch_id: str) -> OntologyUpdate:
        """Register one batch of insight vectors; returns ACTIVATES records + metrics.

        Row ids are allocated here, from ``ConceptStore.next_chunk_id``: they continue the caller's journal, so that
        counter must equal the number of rows the caller holds (``forget`` sets it to the rows kept); ``vectors[i]``
        is row ``update.row_ids[i]``."""
        start = time.perf_counter()
        st = self.store
        if st.embeddings.size and st.embeddings.shape[1] != vectors.shape[1]:
            raise OntologyError(f"vector dim {vectors.shape[1]} != ontology dim {st.embeddings.shape[1]}")
        emb_before = observability.snapshot_embeddings(st)
        n_before = len(st.concept_ids)
        damping = self._damping()  # from the shares before this batch
        x_unit = l2_normalize(vectors)
        w = np.clip(np.asarray(weights, dtype=np.float64), 1e-3, 1.0)
        x_in = (x_unit * w[:, None]).astype(np.float32)
        row_ids = list(range(st.next_chunk_id, st.next_chunk_id + len(vectors)))
        st.next_chunk_id += len(vectors)

        if st.is_empty:
            acts, ext = self._cold_start(x_in, x_unit, row_ids, batch_seq)
        else:
            acts, ext = self._stream(x_in, x_unit, row_ids, batch_seq, damping)
            ext.damped = int(np.sum(damping < 1.0))
        ext.max_step, ext.clamped = self._clamp_steps(emb_before)

        records = self._activation_records(acts, x_unit, row_ids, w, pattern_ids, batch_id)
        errors = self.check_invariants(records, pattern_ids, new_from=n_before)
        if errors:
            raise OntologyError("; ".join(errors))

        metrics = self._record_metrics(ext, len(vectors), emb_before, batch_seq, time.perf_counter() - start)
        return OntologyUpdate(
            activations=records,
            row_ids=row_ids,
            metrics={**asdict(metrics), "activation_count": len(records)},
            new_attractors=[int(c) for c in st.concept_ids[n_before:]],
        )

    def _cold_start(self, x_in: np.ndarray, x_unit: np.ndarray, row_ids: list[int], batch_seq: int) -> tuple[Activations, _Extraction]:
        """First batch: extract the initial attractor set from the batch itself."""
        cents, local = eng.extract_attractors(x_in, x_unit, self.config)
        gids = self.store.append_concepts(cents, chunk_counts=_mass(local, x_in, len(cents)), batch_id=batch_seq)
        acts = _tag(eng.remap_activation_edges(local, chunk_id_map=row_ids, concept_id_map=gids), "cold_start")
        return acts, _Extraction(extracted=len(cents), kept=len(cents))

    def _damping(self) -> np.ndarray:
        """Per-attractor EMA multiplier d_j = min(1, tau / share_j), share_j = its mass over all mass: an
        over-represented attractor keeps its members but moves less (docs/05_latent_anchors.md §5.6); all ones before any share exists."""
        st = self.store
        total = st.chunk_counts.sum()
        if st.is_empty or total == 0:
            return np.ones(len(st.concept_ids))
        tau = observability.density_threshold(len(st.concept_ids), self.config)
        share = st.chunk_counts.astype(np.float64) / total
        return np.minimum(1.0, tau / np.maximum(share, 1e-12))

    def _clamp_steps(self, before: np.ndarray | None) -> tuple[float | None, int]:
        """Trust region: a pre-existing centroid that moved more than max_centroid_step in this
        batch is pulled back onto that radius; returns (largest raw move, attractors clamped)."""
        if before is None or not len(before):
            return None, 0
        st, limit = self.store, self.config.max_centroid_step
        delta = st.embeddings[: len(before)].astype(np.float64) - before.astype(np.float64)
        steps = np.linalg.norm(delta, axis=1)
        over = np.flatnonzero(steps > limit)
        for i in over:
            moved = before[i] + delta[i] * (limit / steps[i])
            st.embeddings[i] = (moved / np.linalg.norm(moved)).astype(np.float32)
        return float(steps.max()), len(over)

    def _stream(
        self, x_in: np.ndarray, x_unit: np.ndarray, row_ids: list[int], batch_seq: int, damping: np.ndarray
    ) -> tuple[Activations, _Extraction]:
        """Later batches: assign (EMA update), buffer orphans, then extract or wire them."""
        st, cfg = self.store, self.config
        ext = _Extraction(adaptive_thresh=eng.compute_adaptive_threshold(st, cfg))
        assigned, orphan_embs, orphan_ids = eng.assign_and_update(x_in, row_ids, st, cfg, ext.adaptive_thresh, batch_seq, damping)
        acts = _tag(assigned, "assign")
        ext.orphaned = len(orphan_ids)
        if orphan_embs:
            st.push_orphans(np.stack(orphan_embs), orphan_ids)
        if st.should_extract_orphans(cfg, len(orphan_ids)):
            buf, buf_ids = st.get_orphan_buffer()
            new_c, local = eng.extract_attractors(buf, l2_normalize(buf), cfg)
            kept, absorptions = eng.soft_merge_orphans(new_c, st, cfg)
            ext.extracted, ext.soft_merged, ext.kept = len(new_c), len(absorptions), len(kept)
            acts += _tag(eng.route_absorbed_activations(local, absorptions, buf_ids, buf, st, cfg, batch_seq, damping), "absorbed")
            if len(kept):
                mass = _mass(local, buf, len(new_c))
                gids = st.append_concepts(kept, chunk_counts=[mass[i] for i in range(len(new_c)) if i not in absorptions], batch_id=batch_seq)
                mapping = eng.build_kept_local_to_global(len(new_c), absorptions, gids)
                acts += _tag(
                    eng.remap_activation_edges(local, chunk_id_map=buf_ids, concept_id_map=mapping, skip_absorbed=absorptions),
                    "omp",
                )
            st.clear_orphan_buffer()
        elif orphan_ids:
            acts += _tag(eng.assign_orphans_nearest(np.stack(orphan_embs), orphan_ids, st, cfg, batch_seq, damping), "nearest")
            st.clear_orphan_buffer()
        return acts, ext

    def _record_metrics(
        self, ext: _Extraction, ingested: int, emb_before: np.ndarray | None, batch_seq: int, elapsed_s: float
    ) -> observability.BatchMetrics:
        """lac ``BatchMetrics`` + health warnings, appended to ``state/ontology_metrics.csv``."""
        st = self.store
        edges = self.topology()
        dirty = set(st.dirty_concept_indices)
        st.clear_dirty()
        metrics = observability.BatchMetrics(
            batch_id=batch_seq,
            elapsed_s=elapsed_s,
            ingested=ingested,
            assigned_instant=ingested - ext.orphaned,
            orphaned=ext.orphaned,
            orphan_rate=ext.orphaned / max(ingested, 1),
            total_concepts=len(st.concept_ids),
            new_extracted=ext.extracted,
            new_kept=ext.kept,
            soft_merged=ext.soft_merged,
            extraction_yield=(ext.kept / ext.extracted) if ext.extracted else None,
            related_to_edges=len(edges),
            avg_degree=observability.avg_related_to_degree(len(edges), len(st.concept_ids)),
            max_concept_density_pct=observability.max_concept_density_pct(st),
            centroid_drift=observability.mean_centroid_drift(emb_before, st.embeddings, dirty),
            adaptive_thresh=ext.adaptive_thresh,
            density_threshold=observability.density_threshold(len(st.concept_ids), self.config),
            damped_attractors=ext.damped,
            max_centroid_step=ext.max_step,
            clamped_attractors=ext.clamped,
        )
        observability.apply_health_warnings(metrics, self.config)
        observability.MetricsRecorder(self.state_dir / "ontology_metrics.csv").append(metrics)
        return metrics

    def _activation_records(
        self,
        acts: Activations,
        x_unit: np.ndarray,
        row_ids: list[int],
        weights: np.ndarray,
        pattern_ids: list[str],
        batch_id: str,
    ) -> Activations:
        """One record per (pattern, attractor), alignment measured against the final centroid."""
        local_of = {rid: i for i, rid in enumerate(row_ids)}
        best: dict[tuple[str, int], dict[str, Any]] = {}
        for act in acts:
            row_id = int(act["chunk_id"])
            i = local_of[row_id]
            cid = int(act["concept_id"])
            alignment = float(x_unit[i].astype(np.float64) @ self.centroid(cid).astype(np.float64))
            rec = activation_record(
                pattern_ids[i],
                cid,
                alignment,
                float(weights[i]),
                act["source"],
                float(act["weight"]),
                batch_id,
                row_id=row_id,
                weak=bool(act.get("weak", False)),
            )
            key = (rec["pattern_id"], cid)
            if key not in best or rec["alignment"] > best[key]["alignment"]:
                best[key] = rec
        return list(best.values())

    def check_invariants(self, records: Activations, pattern_ids: list[str], new_from: int = 0) -> list[str]:
        """Every pattern activates something; no *new* anchor is empty (an older one may be: a linked
        anchor survives the deletion of its last member, ``forget``); centroids are unit-norm."""
        errors: list[str] = []
        activated = {r["pattern_id"] for r in records}
        missing = [p for p in pattern_ids if p not in activated]
        if missing:
            errors.append(f"{len(missing)} patterns lack ACTIVATES (e.g. {missing[:3]})")
        new_counts = self.store.chunk_counts[new_from:]
        if len(new_counts) and float(np.min(new_counts)) <= 0.0:
            errors.append("attractor with mass 0")
        if self.store.embeddings.size:
            norms = np.linalg.norm(self.store.embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                errors.append("attractor centroids not unit-norm")
        return errors
