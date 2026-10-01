# SDD 07 — Latent ontology (attractors / latent anchors)

## Purpose
Integrate insight vectors into the **existing** dynamic ontology of lac v2 (vendored as `ltir/engines/lac/` from `lac/v2_orchestrator`): living attractors, adaptive assignment, orphan buffer, OMP dictionary extraction, soft merge and a mutual-kNN topology. This tests whether dictionary-style attractors can act as cross-dimensional statistical concepts.

## Scope
`ltir/ontology.py::LatentOntology` (adapter). The lac engine is called with `ltir.config.Config`; there is no second settings object.

## Inputs
`ingest(vectors (n, D) unit float32, weights (n,) insight_weight, pattern_ids, batch_seq:int, batch_id:str)`.

## Outputs
`OntologyUpdate(activations: list[activation record], row_ids, metrics, new_attractors)`. Updated `ConceptStore` state (saved with `save()`). `topology()` returns mutual-kNN edges `[{source, target, weight}]`.

## Dependencies (reused from lac; configured by `Config`)
`storage.ConceptStore` (`update_concept_centroid`, `append_concepts`, orphan buffer, `save/load`); `ontology_engine.{extract_attractors, compute_adaptive_threshold, assign_and_update, soft_merge_orphans, route_absorbed_activations, build_kept_local_to_global, remap_activation_edges, assign_orphans_nearest, calculate_knn_topology}`; `observability.{BatchMetrics, MetricsRecorder, snapshot_embeddings, mean_centroid_drift, avg_related_to_degree, max_concept_density_pct, density_threshold, apply_health_warnings}`.

## Lifecycle (same stage order as lac's `v2_orchestrator.main.run_batch`, which is not vendored)
```text
x_unit = l2(vectors)                the single frame (no centering; SDD 06)
x_in   = w · x_unit                 evidence-magnitude encoding (see Weight)
d_j    = min(1, τ_density / share_j)  per-attractor damping from the shares before the batch (Guards)
store empty?  ── yes ─► extract_attractors(x_in, x_unit) ─► append_concepts           source=cold_start
              └─ no ──► τ = compute_adaptive_threshold
                        assign_and_update(x_in, τ, d)  → activations (damped EMA) + orphans    source=assign
                        push_orphans
                        should_extract_orphans?
                          yes ─► extract_attractors(buffer, unit(buffer)) ─► soft_merge_orphans
                                 route_absorbed_activations (EMA into existing)                 source=absorbed
                                 append_concepts + remap_activation_edges                      source=omp
                          single orphan ─► assign_orphans_nearest                              source=nearest
trust region: pre-existing centroids that moved > MAX_CENTROID_STEP are pulled back onto that radius
activation records ← cosine(x_unit, final centroid) ; invariants ; mutual kNN ; BatchMetrics → state/ontology_metrics.csv
```
`extract_attractors` (cold start and orphan buffer alike) multiplies its OMP matrix by `DICTIONARY_INPUT_SCALE` and then runs the signed-insight repair. EMA updates and routing never see that scale. `remap_activation_edges` keeps fields on the activation (including `rerouted`), so ingest does not zip two lists by position.

### Signed extraction (inside `extract_attractors`)
| Repair | Why |
|---|---|
| **OMP input scale s = 10** | lac fits codes with `MiniBatchDictionaryLearning` (lasso, α = 1). For inputs of norm ≤ 1, soft-thresholding zeroes almost every code, so atoms stay near their SVD initialisation (measured: mixed, near-duplicate atoms). OMP directions and relative reconstruction error are scale-invariant, so scaling by s is α/s. EMA updates and routing never see s. |
| **Sign repair** | lac weights activations by \|coefficient\|, so an anti-aligned row could "activate" an atom. Atom sign is arbitrary: atoms whose users are anti-aligned on balance are flipped. |
| **K-sweep elbow** | lac stopped when the improvement from K_prev to K fell below the tolerance but kept **K** (the larger dictionary). The sweep now keeps K_prev: the extra atoms did not pay for themselves, and the smaller dictionary is the parsimonious model. |
| **Intra-extraction soft merge** | even so, OMP can split one phenomenon into near-duplicate atoms (measured before the repairs: 3 tight clusters → 5 atoms). lac applies `SOFT_MERGE_LOW` only against existing centroids; the same rule is applied among atoms of the same extraction (usage-weighted mean). |
| **Alignment floor + reroute** | activations with cosine < `MIN_ACTIVATION_ALIGNMENT` are dropped; an uncovered row goes to its best atom (`source=reroute`, coverage invariant) and is flagged `weak=true` when even that alignment is below the floor (weight clipped at 0); unused atoms are removed. Traversal ignores ACTIVATES below `ACTIVATION_THRESHOLD` (0.40) anyway. |

### Guards (stability without freezing)
No guard freezes learning or diverts rows to the orphan buffer (both would trip on small, legitimately skewed datasets and let orphan extraction mint micro-atoms):
* **Adaptive hub threshold** `τ_density = max(DENSITY_FLOOR, DENSITY_MULTIPLE / N_attractors)` = 3× the uniform share, never below 25 %. The demo's largest theme is not a hub (Qwen3: 10 of 28 = 36 % with 4 attractors, τ = 75 %; MiniLM: 9 of 28 = 32 % with 7, τ = 43 %); the fixed 25 % rule flagged it on every run.
* **Per-attractor damping**: with `share_j = chunk_count_j / next_chunk_id` measured before the batch, the EMA step of attractor j is multiplied by `d_j = min(1, τ_density / share_j)`. An over-represented attractor keeps accepting members (assignment, orphan routing and extraction are unchanged) but its centroid moves proportionally less, so a hub cannot be dragged towards the mean of everything it absorbs.
* **Trust region**: after the batch, a pre-existing centroid whose move exceeds `MAX_CENTROID_STEP` is pulled back to `normalize(c_old + (MAX_CENTROID_STEP / ‖Δ‖)·Δ)`; alignments are measured against the final centroids. Calibrated on a same-domain second batch (largest healthy move 0.023, mean drift 0.007): `MAX_CENTROID_STEP = 0.10` (≈ 4× headroom).
* Telemetry per batch: `density_threshold`, `damped_attractors`, `max_centroid_step` (raw), `clamped_attractors`; warnings use the same thresholds (`apply_health_warnings(metrics, config)`).

### Weight (requirement §9)
Evidence-magnitude encoding: a pattern enters as `x = w·x̂` (‖x̂‖ = 1, w = insight_weight clipped to [1e-3, 1]).
* **Assignment** (`assign_and_update`, soft merge, kNN) is cosine-based, hence *invariant* to w. Strength never changes which attractor a pattern belongs to.
* **EMA update** (`update_concept_centroid`): `c' = normalize((1−α′)c + α′·w·x̂)` with α′ = max(0.01, α/√(mass+1)) · d_j (damping, Guards). The pull on a centroid scales ≈ w (test: w = 1 moves a centroid > 3× more than w = 0.1).
* **Dictionary fitting** (OMP): rows scaled by w weight the reconstruction loss ∝ w², so weak insights rarely mint their own atoms.
* **Activation**: `alignment = cos(x̂, c)` (geometric fit) and `strength = alignment·w` (evidence), both stored on ACTIVATES.

## Data contract — activation record (journal + graph)
`models.activation_record`: `{pattern_id, attractor_id:int, alignment, strength, engine_weight (lac weight: cosine or |OMP coef|), source ∈ {cold_start, assign, omp, absorbed, nearest, reroute}, weak, batch_id, row_id}`. The graph recomputes `alignment` against *current* centroids (SDD 09) and keeps `alignment_at_ingest`.

Attractor node properties (SDD 08): `attractor_id, mass (lac chunk_count), evidence_mass (Σ strength), n_patterns, distinct_scopes, dimensions, targets, datasets, signature` (strength-weighted signed components), `label` (top-2 signature components, e.g. `discount ↑ · margin ↓`), `dispersion` (1 − mean alignment), `last_updated_batch, created_at, centroid{dim, norm, representation_version, fingerprint}`.

## Configuration (lac names; SIG defaults sized for 10–500 insights instead of thousands of chunks)
| Name | lac default | SIG default | Reason |
|---|---|---|---|
| `CONCEPTS_PER_CHUNK` | 2 | **1** | one dominant phenomenon per insight at extraction; multi-membership comes from assignment `TOP_K_ASSIGN` |
| `DICTIONARY_K_MIN` / `_STEP` | 20 / 20 | **4 / 2** | tens of insights |
| `MAX_CONCEPT_COUNT` | 200 | **40** | |
| `RELATED_TO_PEER_COUNT` | 7 | **3** | small attractor sets would become near-complete |
| `RELATED_TO_MIN_WEIGHT` | 0.15 | **0.30** | composite vectors have a higher baseline similarity |
| `MIN/MAX_ASSIGN_THRESHOLD` | 0.30/0.45 | **0.75/0.80** (Qwen3; MiniLM 0.55/0.80) | same reason; the floor is embedder-specific (Calibration) |
| `SOFT_MERGE_LOW` | 0.55 | **0.85** | phenomenon clusters are tight (0.93–0.99) |
| `CENTROID_ALPHA`, `TOP_K_ASSIGN`, `MIXTURE_RATIO`, `ADAPTIVE_PERCENTILE`, `ORPHAN_BUFFER_MIN_FACTOR`, `RECONSTRUCTION_ERROR_TOLERANCE`, `DEAD_CONCEPT_PENALTY`, `MAX_DEAD_CONCEPT_RATIO`, `DICTIONARY_BATCH_SIZE`, `RANDOM_SEED` | lac | **lac** | unchanged |
| centering | running mean (first batch uncentred, later batches centred by it) | **none** (removed) | lac's frame moves between batches, so cross-batch cosines drift and query vectors live elsewhere; SIG keeps one unit-vector frame |
| `DICTIONARY_INPUT_SCALE`, `MIN_ACTIVATION_ALIGNMENT` | — | 10, 0.20 | adapter repairs |
| `DENSITY_FLOOR`, `DENSITY_MULTIPLE`, `MAX_CENTROID_STEP` | — (fixed 25 % warning) | 0.25, 3.0, 0.10 | Guards |
| `WARN_ORPHAN_RATE`, `WARN_MIN_EXTRACTION_YIELD`, `WARN_AVG_DEGREE` | 0.50, 0.10, (1, 8) hard-coded | same, configurable | advisory warnings |

## Calibration (embedder-specific cosine thresholds)
Cosine thresholds belong to the embedder, not to the method: Qwen3 places unrelated texts closer together than MiniLM (label cosine 0.71 vs 0.43; mean retail ↔ housing attractor cosine 0.21 vs 0.10). `scripts/compare_embedders.py` measures the two quantities that bound `MIN_ASSIGN_THRESHOLD` (demo, then demo seed 8, then `housing.csv`):

| | MiniLM-L12 | Qwen3 → 384 |
|---|---|---|
| smallest assignment alignment of the same-domain batch (must be assigned) | 0.842 | 0.788 |
| largest alignment of the unrelated batch to a retail attractor at 0.55 (must stay orphan) | none (all orphans) | 0.71 (53 % orphans) |
| `MIN_ASSIGN_THRESHOLD` | **0.55** | **0.75** |
| unrelated batch orphan rate at that threshold | 1.0 | 1.0 |

Rule: after changing `EMBEDDING_MODEL`, rerun the script and set the floor between the two measured values (`domains_separated` must be `True`). With ≥ 10 attractors the adaptive threshold is `clip(p85, MIN, MAX)` (SDD 16 §12), so the floor still applies.

`RELATED_TO_MIN_WEIGHT` stays 0.30 for both. Under Qwen3 one mutual-kNN edge joins a retail and a housing attractor (`corr(discount~margin) weakens · margin ↓` ↔ `median house value ↓ · total rooms ↓`, 0.57: both say "a value metric falls"). No global floor separates it: the largest retail ↔ housing cosine (0.66) exceeds the weakest within-domain edge (0.58). The traversal multiplies by the edge weight and keeps the activation floors; on six retail and housing questions no evidence item came from the other domain, for MiniLM and Qwen3 alike. Known limitation; `cross_domain_links` and `cross_domain_evidence` in the script track it.

## Failure modes
`OntologyError` (dim mismatch, invariant violation) marks the batch FAILED, and the checkpoint is restored (SDD 09).

## Invariants
Checked every batch by `check_invariants()` (a violation raises `OntologyError`): every ingested pattern has ≥ 1 activation; no attractor has mass 0; centroids are unit-norm.
Checked by tests (SDD 09): `store.next_chunk_id == journal rows == mmap rows`, because a pattern's `row_id` is its journal row.

## Observability
lac `BatchMetrics` per batch (ingested, orphaned, orphan_rate, total_concepts, new_extracted/kept, soft_merged, extraction_yield, related_to_edges, avg_degree, max_concept_density_pct, centroid_drift, adaptive_thresh, density_threshold, damped_attractors, max_centroid_step, clamped_attractors, warnings) appended to `state/ontology_metrics.csv` and copied into the batch record. The hub warning fires above `τ_density`, not at a fixed 25 %. The K-sweep itself is silent: `new_extracted` counts the active atoms of the selected dictionary; the swept K before dead atoms are dropped, the per-K reconstruction error and the stop rule that fired are not recorded.

## Testing requirements
`tests/test_ontology.py`: cold start (coverage, unit norm, one attractor per planted cluster), assignment, orphans → OMP → new concept, single-orphan nearest fallback, soft merge absorption, weight → EMA pull, `τ_density` formula, damping slows an over-represented attractor without changing membership, trust region caps a batch move (unit norm, warning, CSV columns), sign repair, state round-trip.

## Integration points
`Engine.process()` stage UPDATING_ONTOLOGY; `graph.build_snapshot()` reads `store` for Attractor nodes and `topology()` for RELATED_TO.

## Current implementation status
Implemented. Synthetic demo (Qwen3-Embedding-0.6B → 384): 28 insights → 4 attractors. The three recurring planted phenomena keep their own anchors (`delivery days ↑ · return rate ↑` 10 patterns, `discount ↑ · margin ↓` 9 — including the APAC∧tablets erosion scope that MiniLM left as a singleton — and `margin ↑` 7); the two unique phenomena (the EU∧phones correlation break and the EU∧laptops∧retail contrast) share one 2-member anchor (`corr(discount~margin) weakens · margin ↓`) because Qwen3's metric-label embeddings are less separable than MiniLM's (SDD 06). With MiniLM the same demo gives 7 attractors (each unique phenomenon a singleton). Guards on the calibration run (demo → same-domain batch → housing; Qwen3 at 0.75 and MiniLM at 0.55): no hub warning, `damped_attractors = 0`, `clamped_attractors = 0`, largest healthy centroid move 0.010 (Qwen3) / 0.023 (MiniLM) against `MAX_CENTROID_STEP` 0.10 — healthy operation is not altered. The same-domain batch is fully assigned. `housing.csv` arrives fully as orphans (the advisory `orphan_rate` warning fires, as it should for a new domain) and OMP mints 11 attractors of its own (MiniLM: 9); one RELATED_TO edge crosses the domains under Qwen3 (Calibration).
