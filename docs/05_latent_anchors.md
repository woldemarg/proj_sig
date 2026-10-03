# 5. Latent anchors — the self-organising ontology

> **In one paragraph.** Insight vectors stream batch by batch into the dynamic ontology of the vendored lac engine: a store of unit-norm centroids ("attractors", called latent anchors here and themes in the UI). A new insight joins the anchors it aligns with and pulls their centroids towards itself (an EMA whose step shrinks as an anchor matures); insights that fit nowhere become orphans, from which sparse dictionary learning (OMP) extracts new anchors, merging any that duplicate existing ones. Anchors are linked to their mutual nearest neighbours. Because the phenomenon block dominates the vector, an anchor gathers insights that *behave* alike — "discount up, margin down" — wherever in the data they occur. That is the bridge transversal retrieval walks across.

**Code** `ltir/analysis/ontology.py` (`LatentOntology`), `ltir/engines/lac/` (`storage.py`, `ontology_engine.py`, `observability.py`), `ltir/analysis/graph.py` (anchor descriptions) · **Tests** `tests/test_ontology.py` · **Previous** [4. Representation](04_representation.md) · **Next** [6. Graph and storage](06_graph_and_storage.md)

---

## 5.1 The idea

The structural lattice can only relate subgroups that share conditions. Anchors relate subgroups that share a **phenomenon**: they are centroids in the insight-vector space, and the vector of [4.3](04_representation.md#43-the-tripartite-vector) gives two thirds of its similarity to the signed phenomenon. The ontology is *living* rather than a one-off clustering: it is built incrementally, keeps its anchors stable across batches, absorbs new evidence in proportion to its strength, and grows new anchors only for phenomena it has not seen. lac was designed for text chunks; SIG feeds it insight vectors and repairs the places where that difference matters (signed atoms, small batches, evidence weights).

> **Running example.** On the demo the ontology forms four anchors from 28 insights:
>
> | Anchor | Members | Signature (top entries) | Planted mechanism |
> |---|---|---|---|
> | `A-0 "delivery days ↑ · return rate ↑"` | 10 | delivery days +0.91, return rate +0.66 | delay → returns |
> | `A-1 "discount ↑ · margin ↓"` | 9 | discount +1.00, margin −0.49 | discount erosion (US∧phones, APAC∧tablets, EU∧tablets∧retail, …) |
> | `A-2 "margin ↑"` | 7 | margin +1.00 | EU laptops uplift and its stronger online specialisation |
> | `A-3 "corr(discount~margin) weakens · margin ↓"` | 2 | correlation −0.53, margin −0.47 | the two one-off phenomena (EU∧phones correlation break, EU∧laptops∧retail contrast) |
>
> `P-bc4657a04746` (phones ∧ US) activates `A-1` with alignment 0.98.

## 5.2 One batch through the ontology

`LatentOntology.ingest(vectors, weights, pattern_ids, batch_seq=, batch_id=) → OntologyUpdate(activations, row_ids, metrics, new_attractors)`. The stage order follows lac's own batch loop (which is not vendored):

```text
x_unit = l2(vectors)                        the single frame: no centering, the same rows as the journal and the queries
x_in   = w · x_unit                         evidence-magnitude encoding, w = insight_weight clipped to [1e-3, 1]  (5.5)
d_j    = min(1, τ_density / share_j)        per-anchor damping from the shares before this batch            (5.6)
store empty?  ── yes ─► extract_attractors(x_in, x_unit) ─► append_concepts                    source = cold_start
              └─ no ──► τ = compute_adaptive_threshold
                        assign_and_update(x_in, τ, d)  → activations (damped EMA) + orphans    source = assign
                        orphans ≥ 2 ─► extract_attractors(orphans) ─► soft_merge_orphans
                                       route_absorbed_activations (EMA into existing anchors)  source = absorbed
                                       append_concepts + remap_activation_edges                source = omp
                        orphans = 1 ─► assign_orphans_nearest                                  source = nearest
trust region: pre-existing centroids that moved more than MAX_CENTROID_STEP are pulled back       (5.6)
activation records ← cos(x_unit, final centroid); invariants; mutual kNN; BatchMetrics → state/ontology_metrics.csv
```

Orphans are pushed into lac's orphan buffer, whose flush rule is "at least `DICTIONARY_K_MIN · ORPHAN_BUFFER_MIN_FACTOR` (12) rows, or — partial flush — this batch produced orphans and the buffer holds at least 2". Because every batch with orphans either extracts (two or more) or wires its single orphan to the nearest anchor, the buffer is empty between batches: in practice orphans never wait for a later batch, and the 12-row rule never decides.

## 5.3 Extraction: OMP K-sweep with signed repair

`extract_attractors(x_in, x_unit, config)` = `_omp_extract` + `repair_extraction`, used for the first batch and for orphans.

**1. Input scale.** The OMP input is `X = DICTIONARY_INPUT_SCALE · x_in` (10). lac fits the dictionary with sklearn's `MiniBatchDictionaryLearning`, whose sparse coding during dictionary updates penalises codes with `alpha = 1`; for inputs of norm ≤ 1 that penalty zeroes almost every code and the atoms stay at their initialisation (measured before the repair: mixed, near-duplicate atoms). Scaling by `s` is equivalent to `alpha / s`; OMP directions and the relative error are scale-invariant. The EMA never sees the scale (but see `engine_weight` in [5.9](#59-activation-records-and-batch-metrics)).

**2. K-sweep.** `K = max(2, min(DICTIONARY_K_MIN, n)), +DICTIONARY_K_STEP, …, ≤ min(MAX_CONCEPT_COUNT, n)` (4, 6, … up to 40), codes by OMP with `CONCEPTS_PER_CHUNK = 1` non-zero per row:

```text
err(K)  = ‖X − A_K D_K‖_F² / ‖X‖_F²          dead(K) = #{atoms with no |a_ij| > 1e-5} / K
tol(K)  = RECONSTRUCTION_ERROR_TOLERANCE (0.015) + dead(K) · DEAD_CONCEPT_PENALTY (0.05)
```

A `K` with `dead(K) > MAX_DEAD_CONCEPT_RATIO` (0.25) is skipped while no `K` has been accepted and ends the sweep afterwards (the last accepted `K` is kept). When `err(K_prev) − err(K) < tol(K)` the sweep stops and **keeps `K_prev`** — the elbow: the extra atoms did not pay for themselves, and the smaller dictionary is the parsimonious model (lac kept the larger `K` here). If no `K` was ever accepted, the last attempt is used. Fewer than `DICTIONARY_K_MIN` rows → one unit atom per row (still repaired below). Local activations: per row, the top-`CONCEPTS_PER_CHUNK` atoms by `|a_ij|` with `|a_ij| > 1e-5`.

**3. Signed repair** (`repair_extraction`, against the unit rows `x̂`) — OMP atom signs are arbitrary and lac weighted activations by `|coefficient|`, so an anti-aligned row could "activate" an atom:

| Repair | Rule | Why |
|---|---|---|
| sign | flip atom `j` when `Σ_{users i} cos(x̂_i, d_j) < 0` | an anchor must point towards its members |
| intra-extraction soft merge | visiting atoms by usage, atom `j` is absorbed by the first already kept atom with `cos(d_j, d_k) > SOFT_MERGE_LOW` (0.85); a host becomes `normalize(Σ max(usage, 1) · d)` | OMP splits one phenomenon into near-duplicate atoms (3 tight clusters gave 5 atoms before the repair); lac applied the merge rule only against existing centroids |
| alignment floor | activations with `cos(x̂_i, d_j) < MIN_ACTIVATION_ALIGNMENT` (0.20) are dropped; a row left without an atom is rerouted to its best atom (`rerouted = True`, weight `max(cos, 0)`), and flagged `weak = True` when even that is below the floor. If that atom is then absorbed into an existing anchor, the routed record (`absorbed`) no longer carries the flags; the snapshot's alignment test still marks it weak ([5.9](#59-activation-records-and-batch-metrics)) | the coverage invariant wins, and consumers can tell a weak membership |
| cleanup | unused atoms are removed, ids remapped to `0..K′−1`, `chunk_counts` = activations per atom | |

## 5.4 Assignment, EMA and orphans

**Adaptive threshold** (`compute_adaptive_threshold`): with fewer than 10 anchors `τ = MIN_ASSIGN_THRESHOLD` (0.75 for Qwen3, 0.55 for MiniLM — [5.10](#510-calibration-per-embedder)); otherwise `τ = clip(percentile_85(off-diagonal cos(c_j, c_k)), MIN_ASSIGN_THRESHOLD, MAX_ASSIGN_THRESHOLD = 0.80)` — anchor-to-anchor similarity is a loose upper bound for insight-to-anchor similarity.

**Assignment** (`assign_and_update`): `sim_ij = cos(x_i, c_j)`, one similarity matrix per batch against the centroids as the batch found them (the EMA updates of earlier rows in the batch do not change later rows' assignment). If `max_j sim_ij ≥ τ`, the row activates every anchor among its `TOP_K_ASSIGN` (2) best with `sim_ij ≥ τ` and `sim_ij ≥ MIXTURE_RATIO (0.9) · max_j sim_ij` — an insight can belong to two anchors when it mixes two phenomena — and each activation updates that centroid. Otherwise the row is an orphan.

**EMA with concept inertia** (`ConceptStore.update_concept_centroid`), per activation:

```text
α′ = max(0.01, CENTROID_ALPHA (0.05) / sqrt(count_j + 1)) · d_j        c_j ← normalize( (1 − α′) · c_j + α′ · x_i )        count_j += 1
```

A young anchor moves quickly, a mature one slowly; `d_j` is the damping of [5.6](#56-stability-guards).

**Orphans.** When two or more rows are orphans, extraction runs on them. New atoms whose best cosine to an existing anchor exceeds `SOFT_MERGE_LOW` (0.85) are absorbed by that anchor (`soft_merge_orphans`): their activations are re-pointed and the host is EMA-updated with the buffered vectors (`route_absorbed_activations`); the remaining atoms become new anchors. A single orphan is wired to its nearest anchor (`assign_orphans_nearest`, EMA included).

## 5.5 How the evidence weight acts

An insight enters as `x = w · x̂` with `‖x̂‖ = 1`:

* **Assignment is invariant to `w`.** Assignment to existing anchors, soft merge and kNN are cosine-based, so the weight never changes which existing anchor an insight joins; it does shape extraction (below), and so which new anchors exist.
* **The pull scales with `w`.** In the EMA, `α′ · x_i = α′ · w · x̂_i`: a strong insight moves its anchor more (test: `w = 1` moves a centroid more than 3× as far as `w = 0.1`).
* **Extraction listens to strong evidence.** OMP minimises a reconstruction loss in which a row scaled by `w` counts ∝ `w²`, so weak insights rarely mint anchors of their own.
* **Edges record both.** `alignment = cos(x̂, c)` is the geometric fit; `strength = alignment · w` is the evidence ([5.9](#59-activation-records-and-batch-metrics)).

## 5.6 Stability guards

None of the guards freezes learning or diverts rows to the orphan buffer: both would trip on small, legitimately skewed datasets and let orphan extraction mint micro-anchors.

```text
τ_density = max(DENSITY_FLOOR 0.25, DENSITY_MULTIPLE 3.0 / N_anchors)          hub threshold: 3 × the uniform share
share_j   = count_j / next_chunk_id   (before the batch)        d_j = min(1, τ_density / share_j)        EMA damping
Δ_j = c_j(after) − c_j(before);  if ‖Δ_j‖ > MAX_CENTROID_STEP (0.10):   c_j ← normalize(c_j(before) + (MAX_CENTROID_STEP / ‖Δ_j‖) · Δ_j)
```

* **Adaptive hub threshold.** A fixed 25 % hub rule flagged the demo's largest theme on every run; `τ_density` scales with the number of anchors (the demo's largest theme holds 10 of 28 = 36 % with 4 anchors, `τ = 75 %`; with MiniLM 9 of 28 = 32 % with 7 anchors, `τ = 43 %`).
* **Per-anchor damping.** An over-represented anchor keeps accepting members — assignment, orphan routing and extraction are unchanged — but its centroid moves proportionally less, so a hub cannot be dragged towards the mean of everything it absorbs. Damping uses `N` and the shares from before the batch; the reported `density_threshold` and the hub warning use `N` after it.
* **Trust region.** After the batch, a pre-existing centroid whose move exceeds `MAX_CENTROID_STEP` is pulled back onto that radius; alignments are measured against the final centroids. Calibrated on a same-domain second batch: largest healthy move 0.010 with Qwen3 and 0.023 with MiniLM, so the limit leaves at least 4× headroom.

Telemetry per batch: `density_threshold`, `damped_attractors` (anchors with `d_j < 1`, updated or not), `max_centroid_step` (the raw move before clamping), `clamped_attractors`.

## 5.7 Links between anchors

`calculate_knn_topology`: with `k = min(RELATED_TO_PEER_COUNT (3), n − 1)`, an undirected RELATED_TO edge `(j, k)` exists when **each** anchor is in the other's top-`k` by cosine and `cos(c_j, c_k) > RELATED_TO_MIN_WEIGHT` (0.30); the weight is the cosine; edges are stored from the smaller to the larger id and recomputed from scratch after every batch.

Mutual nearest neighbours keep the latent plane sparse (at most `k · N / 2` edges) and suppress hubs, at a price: **an anchor can end up with no link at all** — when its nearest anchors all have closer neighbours of their own. No invariant requires a link; only insights must be covered. In a workspace with four datasets, for example, the retail anchor `discount ↑ · margin ↓` ranked sixth among the neighbours of its nearest anchor (five anchors from other datasets were closer to that one), so it had no RELATED_TO edge. On the demo alone three of the four anchors are linked (A-0–A-1 0.61, A-0–A-2 0.55, A-1–A-2 0.31); `A-3` has none — its best cosine to another anchor is 0.06, below `RELATED_TO_MIN_WEIGHT`. The sphere view flattens 1152 dimensions into three, so visual proximity there is only a rough guide to these cosines.

## 5.8 How an anchor is described

lac stores **no text** for an anchor: only its centroid, `chunk_count`, `last_updated_batch` and `created_at` (`state/concepts.npz`, `state/state.json`). Everything readable is derived from the members at snapshot time (`graph._attractor_nodes`) and recomputed after every batch, so the description follows the membership. Over the members `(record_i, alignment_i, strength_i)` — weak memberships included, with their small strength:

```text
signature(label) = Σ_i strength_i · coef_i(label) / max_l |coef_i(l)|  /  Σ_i strength_i        (each member's components scaled to [−1, 1] first)
evidence_mass    = Σ_i strength_i          dispersion = 1 − mean_i alignment_i          mass = lac chunk_count
```

| Text | Derivation | Example |
|---|---|---|
| `label` | the two strongest signature entries, each rendered as `<label> ↑` / `<label> ↓` or `corr(<a>~<b>) strengthens` / `weakens`, joined by ` · `; `Attractor <k>` without members | `discount ↑ · margin ↓` |
| `signature` | the six strongest entries as `[{"component", "value"}]` | `[{"component": "discount", "value": 1.0}, {"component": "margin", "value": -0.49}, …]` |
| `dimensions`, `targets`, `datasets` | sorted sets over the members' conditions, targets and datasets | `["category", "channel", "region"]`, `["discount"]` |
| `n_patterns`, `distinct_scopes`, `mass`, `evidence_mass`, `dispersion` | counts and sums above | `9, 9, 9, 6.96, 0.024` |
| `last_updated_batch`, `created_at` | lac batch sequence mapped back to the batch id; creation time | |
| `centroid` | `{dim, norm, representation_version, fingerprint}` — the vector contract, not the vector | |
| `description` (prompt only) | `graph.describe_components(signature)`: the two strongest entries as ASCII prose (`<label> up` / `down`, `correlation between a and b strengthens` / `weakens`) joined by ` and `; falls back to the label | `discount up and margin down` |

The prompt line reads `- A-1 "discount up and margin down": 9 patterns over 9 distinct scopes; related: A-2 (0.31), A-0 (0.61)`; the UI drawer says `A recurring pattern learned from 9 insights across 9 different subgroups.` followed by the signature. An anchor's name can change when new members arrive; its id `A-k` and its centroid identity do not, so labels are never used as keys.

## 5.9 Activation records and batch metrics

**Activation record** (`models.activation_record`, journal and graph): `{pattern_id, attractor_id, alignment, strength, engine_weight, source, weak, batch_id, row_id}` with `source ∈ {cold_start, assign, omp, absorbed, nearest, reroute}`. One record per (pattern, anchor), the best alignment kept:

```text
alignment_at_ingest = cos(x̂_i, c_j)  against the final centroid of that batch        strength = alignment · w_i
```

The snapshot recomputes `alignment` against the **current** centroids, so edge weights follow the living ontology, and keeps `alignment_at_ingest`. `engine_weight` is lac's own weight and is not comparable across sources: a cosine for `assign` and `nearest`, `max(cos, 0)` for `reroute`, but `|OMP coefficient|` of the scaled input (≈ `10 · w · cos`) for `cold_start`, `omp` and `absorbed` — and 1.0 when a too-small orphan buffer makes extraction fall back to one anchor per row. In the snapshot a membership is `weak` when the journal flagged it (rerouted below the floor at ingest) or its current alignment is below `MIN_ACTIVATION_ALIGNMENT`. The UI (dashed line) and the walk read that one flag: weak memberships are not walked ([7.3](07_question_answering.md#73-transversal-traversal)). The coverage invariant counts every activation, weak ones included.

**Batch metrics** (lac `BatchMetrics`, one row per batch in `state/ontology_metrics.csv`): `batch_id (the batch sequence), elapsed_s, ingested, assigned_instant (ingested − orphaned), orphaned, orphan_rate (orphans / ingested), total_concepts, new_extracted (atoms after the signed repair), new_kept, soft_merged (merges into existing anchors; intra-extraction merges are not counted), extraction_yield (kept / extracted), related_to_edges, avg_degree (2E / N), max_concept_density_pct, centroid_drift (mean move of the touched centroids), adaptive_thresh, density_threshold, damped_attractors, max_centroid_step, clamped_attractors, warnings`. A subset is copied into the batch record ([9.5](09_operations.md#95-batch-metrics)). Warnings (`apply_health_warnings`, thresholds from `Config`): `orphan_rate > WARN_ORPHAN_RATE` (0.5), `extraction_yield < WARN_MIN_EXTRACTION_YIELD` (0.1), `avg_degree` outside `WARN_AVG_DEGREE` (1, 8), a hub above `τ_density`, any clamped centroid. They are advisory: a new, unrelated dataset legitimately arrives with an orphan rate of 100 %.

## 5.10 Calibration per embedder

Cosine thresholds belong to the embedder, not to the method. Qwen3 places unrelated texts closer together than MiniLM (label cosine 0.71 vs 0.43; mean retail ↔ housing anchor cosine 0.21 vs 0.10). `scripts/compare_embedders.py` measures the two quantities that bound `MIN_ASSIGN_THRESHOLD`, on the demo followed by a same-domain batch (demo seed 8) and by `housing.csv`:

| | MiniLM-L12 | Qwen3 → 384 |
|---|---|---|
| smallest assignment alignment of the same-domain batch (must be assigned) | 0.842 | 0.788 |
| largest alignment of the unrelated batch to a retail anchor at 0.55 (must stay orphan) | none: all orphans | 0.71 (53 % orphans) |
| `MIN_ASSIGN_THRESHOLD` | **0.55** | **0.75** |
| orphan rate of the unrelated batch at that threshold | 1.0 | 1.0 |

**Rule:** after changing `EMBEDDING_MODEL`, rerun the script and set the floor between the two measured values; `domains_separated` must be `True`. With 10 or more anchors the adaptive threshold is `clip(p85, MIN, MAX)`, so the floor still applies.

`RELATED_TO_MIN_WEIGHT` stays 0.30 for both models. A global floor cannot keep links within datasets: under Qwen3 the largest retail ↔ housing cosine (0.66) exceeds the weakest within-domain link (0.58). Measured: one cross-dataset link with three datasets (`corr(discount~margin) weakens · margin ↓` ↔ `median house value ↓ · total rooms ↓`, 0.57 — both "a value metric falls"), three with four datasets. On six retail and housing questions no evidence item came from the other dataset, but in the four-dataset workspace one of four retail questions pulled one item of another dataset into its evidence through such a link. `cross_domain_links` and `cross_domain_evidence` in the script track this; restricting links to anchors that share a dataset is the open design option.

## 5.11 Removing patterns: the orphan rule

Deleting a dataset ([6.8](06_graph_and_storage.md#68-deleting-a-dataset)) removes its patterns and their memberships; `LatentOntology.forget(kept_activations, n_rows)` then recounts every anchor's `chunk_count` from the surviving activations and decides which anchors go (`orphan_anchors`):

```text
anchor dropped  ⇔  no remaining member  ∧  no RELATED_TO link (in the topology as it stood before the deletion)
```

An anchor that keeps a member stays. An anchor left without members but still linked to another anchor stays too: its centroid keeps its place in the mutual-kNN topology and can receive future insights that align with it, so a later upload of a related dataset joins it instead of minting a new one. Its label becomes `Attractor k` until it has members again. Centroids are never un-averaged — a survivor keeps the position its history gave it — and the row counter follows the rewritten journal. When the last pattern of the workspace goes, every anchor goes with it: an empty knowledge base starts cold again. Accordingly the mass invariant of [5.13](#513-guarantees-and-measured-behaviour) applies to anchors *created* in a batch; an older one may legitimately be empty.

## 5.12 Configuration

lac names, SIG-sized defaults (lac was tuned for thousands of text chunks, SIG sees tens to hundreds of insights):

| Parameter | lac | SIG | Reason |
|---|---|---|---|
| `CONCEPTS_PER_CHUNK` | 2 | **1** | one dominant phenomenon per insight at extraction; mixtures come from `TOP_K_ASSIGN` |
| `DICTIONARY_K_MIN` / `_STEP` | 20 / 20 | **4 / 2** | tens of insights |
| `MAX_CONCEPT_COUNT` | 200 | **40** | |
| `RELATED_TO_PEER_COUNT` | 7 | **3** | small anchor sets would become near-complete graphs |
| `RELATED_TO_MIN_WEIGHT` | 0.15 | **0.30** | composite vectors have a higher baseline similarity |
| `MIN` / `MAX_ASSIGN_THRESHOLD` | 0.30 / 0.45 | **0.75 / 0.80** (MiniLM 0.55 / 0.80) | same reason; the floor is calibrated per embedder (5.10) |
| `SOFT_MERGE_LOW` | 0.55 | **0.85** | phenomenon clusters are tight (0.93–0.99) |
| `CENTROID_ALPHA` 0.05, `TOP_K_ASSIGN` 2, `MIXTURE_RATIO` 0.9, `ADAPTIVE_PERCENTILE` 85, `ORPHAN_BUFFER_MIN_FACTOR` 3, `RECONSTRUCTION_ERROR_TOLERANCE` 0.015, `DEAD_CONCEPT_PENALTY` 0.05, `MAX_DEAD_CONCEPT_RATIO` 0.25, `DICTIONARY_BATCH_SIZE` 256, `RANDOM_SEED` 42 | lac | lac | unchanged |
| centering | running mean | **none** | lac's frame moved between batches, so stored and query vectors would drift apart; SIG keeps one frame |
| `DICTIONARY_INPUT_SCALE`, `MIN_ACTIVATION_ALIGNMENT` | — | 10, 0.20 | adapter repairs (5.3) |
| `DENSITY_FLOOR`, `DENSITY_MULTIPLE`, `MAX_CENTROID_STEP` | fixed 25 % hub warning | 0.25, 3.0, 0.10 | guards (5.6) |
| `WARN_ORPHAN_RATE`, `WARN_MIN_EXTRACTION_YIELD`, `WARN_AVG_DEGREE` | hard-coded 0.50, 0.10, (1, 8) | same, configurable | advisory warnings |

## 5.13 Guarantees and measured behaviour

Checked after every batch by `check_invariants()` (a violation raises `OntologyError` → the batch fails with `ontology_failure` and the checkpoint is restored): every ingested insight has at least one activation; no anchor created in the batch has mass 0 (an older anchor may be empty after a deletion, [5.11](#511-removing-patterns-the-orphan-rule)); centroids are unit-norm. Checked by tests: `next_chunk_id == journal rows == vector rows`, because an insight's `row_id` is its journal row.

Tests (`tests/test_ontology.py`): cold start with full coverage, at least three anchors, and the members of each planted cluster sharing one anchor; the orphan rule (a memberless unlinked anchor goes, a linked or populated one stays) and `forget` recounting and renumbering; assignment; orphans → OMP → new anchor; single-orphan nearest fallback; soft merge into an existing anchor; the weight scales the EMA pull; the `τ_density` formula; damping slows an over-represented anchor without changing membership; the trust region caps a move and keeps unit norm; sign repair; state round-trip.

Measured (demo → same-domain batch → `housing.csv`; Qwen3 at 0.75, MiniLM at 0.55): no hub warning, `damped_attractors = 0`, `clamped_attractors = 0` — healthy operation is not altered. The same-domain batch is fully assigned; `housing.csv` arrives entirely as orphans (the advisory orphan-rate warning fires, as it should for a new domain) and OMP mints 11 anchors of its own (MiniLM: 9). Qwen3's labels are less separable than MiniLM's, so the demo's two one-off phenomena share `A-3` where MiniLM gives each a singleton (4 anchors instead of 7), while the three recurring mechanisms keep their own anchors.
