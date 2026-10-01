# SDD 16 — Core mathematics as implemented

## Purpose
State every formula the system actually computes, in pipeline order, with the code that computes it, the parameters that enter it and the reason it is there. SDDs 03–10 describe the modules; this document is the single place where the mathematics is written out so it can be checked against the code line by line. Where a quantity is a heuristic rather than a derived statistic, it says so.

## Scope
Everything numeric between a loaded `DataFrame` and an answer: the EDA stages (`ltir/engines/eda/main_upd.py`), the adapter (`ltir/discovery.py`), selection and weight (`ltir/quality.py`), canonical components (`ltir/canonical.py`), the encoder (`ltir/encoder.py`), the latent ontology (`ltir/ontology.py`, `ltir/engines/lac/`), graph properties (`ltir/structural.py`, `ltir/graph.py`), retrieval (`ltir/query.py`, `ltir/traversal.py`), the benchmark (`ltir/experiment.py`) and the sphere projection (`ltir/sphere.py`). Text contracts are in SDD 17.

## Notation
| Symbol | Meaning |
|---|---|
| `N` | rows of the profiled table (`profile["total_rows"]`) |
| `m` | number of metric (numeric) columns kept by profiling |
| `S` | a subgroup: the rows covered by a conjunction of equality selectors; `n = |S|`, `p = n / N` |
| `med(·)`, `MAD(·)` | median; median absolute deviation around the median |
| `z_k(S)` | signed robust shift of metric `k` inside `S` |
| `C`, `C_S` | Pearson correlation matrix of the metrics over all rows / over `S` |
| `x̂` | unit insight vector (1152-d); `w` its `insight_weight` |
| `c_j` | attractor (latent anchor) centroid `j`, unit norm |

Defaults named in capitals are `Config` fields (SDD 01); the value in force is the one in `ltir/config.py`.

---

## 1. Identities and hashes
| Quantity | Formula | Code |
|---|---|---|
| dataset id | `"ds-" + sha256(file bytes ‖ repr(sorted(bins.items())) [‖ repr(sorted(categories)) if any apply])[:12]` | `ingestion.dataset_fingerprint` |
| pattern id | `"P-" + sha1(json([dataset_id, sorted condition exprs]))[:12]` | `models.pattern_id` |
| row-set identity | `sha1(sorted covered row positions as int64 bytes)[:16]` | `discovery.build_insights` (`row_hash`) |
| closed intent | `int(ext(S)) = {(a, v) : a ∈ profiled categoricals, value of a is v on every covered row, v not null-like}` (Galois closure; the pattern id is taken over it) | `discovery.closed_intent` |
| representation fingerprint | `sha1(json(EmbeddingSpec fields))[:10]` | `models.EmbeddingSpec.fingerprint` |

Same bytes + same options → same ids, which is what makes re-ingestion idempotent (SDD 09) and lets the pattern id double as the duplicate guard.

---

## 2. Profiling (`step1_profile_data`)
1. **Identifiers**: a column is dropped when `nunique == count(non-null)` and its dtype is not float (continuous measures are all-unique by nature).
2. **Typing**: numerics = numpy numeric dtypes; categoricals = object / category / bool / string, cast to `str`.
3. **Constant categoricals** (`nunique ≤ 1`) are dropped before any pair rule.
4. **Nested-column rule**: for every pair `(a, b)` of categoricals with `u = nunique`, if `|distinct (a, b) pairs| ≤ 1.10 · max(u_a, u_b)` and the coarser column's top level holds `< 95 %` of rows, the finer column (larger `u`) is dropped. A near-constant coarser column is excluded because it is trivially "a function of" anything.
5. **Numerics**: zero-variance columns are dropped; then, in column order, a column is dropped when any earlier kept column has `|corr| > 0.95` with it (upper triangle of `|C|`).
6. `global_corr = C` over the kept metrics with `NaN → 0` (used by pass 1).

## 3. Dimension selection (`step2_evaluate_macro_groupings`)
For each categorical `g` with `nunique(g) ≤ √N` and each metric `y`, over the rows where both are present (`n`, `k` groups, `k ≥ 2`, `n > k`):

```text
SS_between = Σ_groups n_i (ȳ_i − ȳ)²        SS_total = Σ (y − ȳ)²        MS_within = (SS_total − SS_between) / (n − k)
power(g, y) = ε² = max(SS_between − (k − 1) · MS_within, 0) / SS_total
```

ε² (Kelley) is η² = SS_between / SS_total with its H0 expectation `(k−1)/(n−1)` removed, so a 45-level column no longer outranks a 3-level one by chance. `power(g) = mean_y power(g, y)`; columns with `power(g) > median_g power(g)` are kept (at most 6), and if fewer than `MIN_SEARCH_DIMENSIONS` (3) survive, the top ones by power are back-filled so that 2-/3-conjunctions exist.

## 4. Search space (`step3_generate_search_space`)
Per selected column, levels are ordered by share `f_v` (descending) and kept while the cumulative mass **before** the level is `< 0.95` — i.e. up to and including the level that crosses 95 %; only the tail beyond it is dropped. `nan` / `<NA>` / `None` levels are skipped.

Candidates are all conjunctions of 2 selectors on distinct attributes; the 3-conjunctions are added when `|2-conj| + |3-conj| ≤ COMPUTE_BUDGET` (5000). When even the 2-conjunctions exceed the budget, the `COMPUTE_BUDGET` pairs with the largest expected support `Π f_v` (independence approximation) are kept. Single selectors are never candidates (EDA design: single-attribute effects are the macro level of §3).

## 5. Pass-1 screening (`step4_evaluate_micro_slices`)
Size screen: `n ≥ n_min = max(5m, ⌊0.005 N⌋)` and `n ≠ N`.

**Robust scale** (`calculate_mad`):
```text
MAD(y) = med(|y − med(y)|);   if MAD = 0:  MAD := (0.6745 / 0.7979) · mean(|y − med(y)|)
```
The fallback is the normal-consistent mean absolute deviation (`MAD = 0.6745 σ`, `mean|·| = 0.7979 σ` under normality); it is 0 only for a constant column. Without it a metric with more than half of its values tied (zero-inflated counts, coarse grids) made every subgroup look infinitely shifted.

**Robust shift** (`robust_z_score`), computed on the global scale `MAD_k = MAD(y_k)` over all rows:
```text
|z_k(S)| = min( 0.6745 · |med_S(y_k) − med(y_k)| / (MAD_k + 1e-6),  ROBUST_Z_CAP = 10 )
```
`0.6745 · Δ / MAD = Δ / σ̂` with `σ̂ = MAD / 0.6745`, so `|z|` is a shift in robust standard deviations. The cap bounds the influence of a degenerate scale; the weight factor (§10) saturates long before 10 anyway. The EDA returns the magnitude; the adapter restores the sign (§8).

**Subgroup-discovery score**: `SD(S) = Σ of the three largest |z_k(S)|` (`top_shifts`).

**EMM score** (correlation-matrix divergence; only when `n ≥ 3m` and `m ≥ 2`):
```text
Δ = C − C_S   with  Δ_ij := 0 where C_S,ij is undefined (metric constant in S)
raw(S) = ‖Δ‖_F                       (Frobenius over all m(m−1) off-diagonal entries, counted twice)
shrink(S) = sqrt( max(0, n − n_min) / (N − n_min) )
EMM_eda(S) = raw(S) · shrink(S)
```
`shrink` is a heuristic reliability factor: a correlation estimated from few rows is pulled towards "no change". An undefined local correlation is *no evidence of change*, not a change to 0.

**Volume utility**: `vol(S) = √p · (1 − p)`, maximal at `p = 1/3` (0.3849), 0 at `p = 0` and `p = 1` — moderate, actionable subgroups beat near-majorities.

**Deduplication before validation** (adapter): (1) selectors with the same extent merge into one cohort whose conditions are `int(ext(S))` (§1) — R5; (2) `temp_index = clip₊(zscore(SD)) + clip₊(zscore(EMM_eda)) + clip₊(zscore(vol))` over the distinct cohorts (`zscore` is scipy's population z-score, `NaN → 0`); (3) greedy in `temp_index` order, a cohort is absorbed by a higher-ranked kept cohort with the same primary metric (pass-1 top shift), the same sign of `med_S(y) − med(y)` and
```text
J(A, B) = |A ∩ B| / |A ∪ B| ≥ REDUNDANCY_JACCARD (0.88)        (boolean row masks: count_nonzero(a & b))
```
— R6; (4) the first `VALIDATION_BUDGET` (50) survivors go to pass 2, with the closed intent's attributes as their scope (implied attributes are never tested as confounders).

## 6. Pass-2 validation (`step4b_deep_validation`)
**Bootstrap stability** of the shift score. For `B = BOOTSTRAP_RESAMPLES = 20` resamples of `S` (n of n, with replacement, seeded by the adapter with `EDA_RANDOM_SEED`):
```text
b_r = Σ_{k ∈ top_shifts} |z_k(S_r)|          CV = std(b) / (mean(b) + 1e-6)
final_sd = SD(S) · (1 − min(CV, 0.9))        stability = final_sd / SD(S) = 1 − min(CV, 0.9) ∈ [0.1, 1]
```
`std` is numpy's population standard deviation. Stability is the fraction of the score that survives resampling noise.

**Confounder drivers** (categoricals not in the subgroup's own scope). With `q_S` and `q` the level distributions inside `S` and overall (aligned, missing levels = 0):
```text
JS(S) = jensenshannon(q_S, q)                        (scipy: the JS *distance*, natural log, ∈ [0, 0.8326])
driver if  JS(S) > 0.15  and  p_χ²(S) < 0.01 / #categoricals
```
`p_χ²` is the chi-square test of independence on the 2 × levels table (rows of `S` vs. the rest), columns with zero total dropped; a degenerate table gives `p = 1`. The named level is `argmax_v (q_S(v) − q(v))` — the level whose share *grew* most. The significance gate is needed because the JS distance of an empirical distribution is inflated by sampling noise in small subgroups.

**Hidden shifts**: a metric outside `top_shifts` with `|z_k(S)| > 1.5` is reported, signed. At most three drivers per subgroup.

## 7. Adapter: from EDA rows to `Insight` (`ltir/discovery.py`)
| Quantity | Formula | Note |
|---|---|---|
| signed shift | `z_k(S) = sign(med_S(y_k) − med(y_k)) · |z_k(S)|` (sign of a zero difference is +1) | EDA keeps magnitudes only |
| EMM per pair | `emm_score = EMM_eda / sqrt(m(m−1))` | RMS correlation change per metric pair ∈ [0, 2]; dataset-independent scale for `MIN_EMM_SCORE`, `WEIGHT_EMM_REF` |
| covariance pair | `argmax_{i<j} |C_S,ij − C_ij|` over pairs defined in `S`; `{}` if none or all `≤ 1e-12` (needs `n ≥ 3m`) | `{pair, local_corr, global_corr, delta}` |
| pair shifts | `z_k(S)` for the pair metrics missing from `top_shifts` (`source = covariance_pair`) | so a covariance insight can be targeted |
| target | the shift with the largest magnitude | `effect_size = z_target` |
| median test | `scale = MAD(y_S) or MAD_k`; `se = 1.2533 · 1.4826 · scale / √n`; `z = (med_S − med) / se`; `p = 2 · Φ̄(|z|)` | asymptotic s.e. of the median `1.2533 σ/√n` with `σ ≈ 1.4826 · MAD`; falls back to `sd/√n` if the scale is 0; `p = 1` for `n < 2` |
| Bonferroni | `p_adj = min(1, p · n_tests)`, `n_tests = |distinct cohorts| · m` | identical extents are one test; overlapping cohorts are still counted as independent tests (conservative) |
| integrated index | `clip₊(zscore(final_sd)) + clip₊(zscore(emm)) + clip₊(zscore(vol))` over the validated set | the upstream EDA's step-5 index (print-only there, not vendored), computed by the adapter |

## 8. Selection rules (`quality.select_insights`)
```text
significant = |effect_size| ≥ MIN_EFFECT_Z (0.5)  ∧  p_adj ≤ MAX_P_ADJUSTED (0.05)
stable      = stability ≥ MIN_STABILITY (0.5)
shift_ok    = significant ∧ stable
emm_ok      = emm_score ≥ MIN_EMM_SCORE (0.08)  ∧  covariance pair present
```
If `¬shift_ok ∧ emm_ok` the insight is retyped `covariance` and retargeted on the pair metric with the larger `|z|` (its shift is never cited as a shift). Then, in order: R1 `support ≥ MIN_SUPPORT_ROWS` (30); R2/R3 `shift_ok ∨ emm_ok` (reason `unstable` if significant but not stable, `not_significant` if `|z| ≥ 0.5` but `p_adj` fails, else `weak_effect`); R4 `weight ≥ MIN_INSIGHT_WEIGHT` (0.2); R7 top `MAX_INSIGHTS_PER_BATCH` (200) by weight. R5/R6 run before validation (§5).

## 9. Insight weight (`quality.weight_factors`, `insight_weight`)
Factors in `[0, 1]`:
```text
effect     = 1 − exp(−|effect_size| / WEIGHT_EFFECT_REF)      (1.5)     covariance: 1 − exp(−emm_score / WEIGHT_EMM_REF) (0.08)
stability  = clip(stability, 0, 1)                                       covariance: None (not measured)
confidence = clip(−log10(max(p_adj, 1e-300)) / WEIGHT_CONFIDENCE_REF, 0, 1)   (6)   covariance: None
support    = clip(vol / 0.3849, 0, 1)                                    0.3849 = max of √p(1−p)
emm        = 0.5 + 0.5 · (1 − exp(−emm_score / WEIGHT_EMM_REF))          a bonus that never zeroes the weight
```
Weighted geometric mean over the **measured** factors, exponents `a = WEIGHT_EXPONENTS = (0.35, 0.25, 0.20, 0.10, 0.10)` for (effect, stability, confidence, support, emm):
```text
w = WEIGHT_FLOOR + (1 − WEIGHT_FLOOR) · exp( Σ_{k measured} (a_k / Σ_{measured} a) · ln max(f_k, 1e-3) )      WEIGHT_FLOOR = 0.05
```
Unmeasured factors are dropped and the exponents renormalised: in a product any substituted constant (e.g. 0.5) is a fixed tax, not a neutral value. Worked example (demo, `category=phones ∧ region=US`, `z = +2.208`, `stability 0.974`, `p_adj = 0`, `vol = 0.270`, `emm = 0.0223`): factors `0.7705, 0.9744, 1.0, 0.7016, 0.6214` → `w = 0.05 + 0.95 · exp(−0.1808) = 0.843`.

Where `w` enters: the ontology input magnitude (§12), ACTIVATES `strength = alignment · w`, attractor `evidence_mass`, the seed score (0.10 · w) and the node rank in traversal (§15–16).

## 10. Canonical components (`canonical.canonicalize`)
The phenomenon of an insight is the list of signed components the encoder composes:
```text
(humanize(metric_k), z_k)   for the target and every shift with |z_k| ≥ MIN_COMPONENT_Z (0.5)   [covariance insights: only |z| ≥ 0.5]
(“correlation between a and b”, sign · EMM_COMPONENT_WEIGHT · emm_score / WEIGHT_EMM_REF)     if covariance-typed or emm_score ≥ MIN_EMM_SCORE
```
`sign = −1` when `|corr|` weakens or reverses (both `|C_ij|, |C_S,ij| > 0.1` with opposite signs), `+1` when it strengthens; `EMM_COMPONENT_WEIGHT = 0.5`, so an EMM score of `0.14` becomes a component of `0.875`, comparable to a moderate shift. Magnitude words (`mild < 1 ≤ moderate < 2 ≤ strong < 3 ≤ extreme`) are text only (SDD 17).

## 11. Tripartite encoder (`encoder.InsightEncoder`)
With `E(·)` the unit sentence embedding — Qwen3-Embedding-0.6B truncated to its first 384 Matryoshka dimensions and **re-normalised** (`E(x) = normalize(f(x)_{:384})`; a slice of a unit vector is shorter than 1), or MiniLM-L12 (native 384), or the hashing stand-in in tests:
```text
s = E(scope text)                 t = E(target text)                 p = normalize( Σ_k coef_k · E(label_k) )   (fallback: E(phenomenon text) if the sum is 0)
v = normalize( [ w_s · s ; w_t · t ; w_p · p ] )        (w_s, w_t, w_p) = BLOCK_WEIGHTS = (0.45, 0.55, 1.0),  dim = 3 · 384 = 1152
```
Because the three blocks are unit vectors, the cosine of two insight vectors decomposes exactly:
```text
cos(v, v′) = ( w_s² s·s′ + w_t² t·t′ + w_p² p·p′ ) / (w_s² + w_t² + w_p²)  =  0.135 · s·s′ + 0.201 · t·t′ + 0.664 · p·p′
```
so the phenomenon carries two thirds of the similarity, the target one fifth and the scope one seventh. Direction lives in the sign of `coef_k`: "margin ↑" and "margin ↓" have `p·p′ ≈ −1` while their sentence embeddings would be nearly identical (measured cosine 0.56 between "increase" and "decrease" sentences). A question is encoded by the same composition (`encode_query`), with the question text standing in for an empty scope or target and signed `±2.0` components for the metrics it names (§15). The query instruction prefix (`Instruct: …\nQuery: `) is applied only to that free question text (and to the empty-component fallback); recognised scope/target strings and every `E(label_k)` are embedded exactly as for documents, so `p_query · p_doc` keeps its exact ±1 geometry.

## 12. Latent ontology (`ltir/ontology.py` over `ltir/engines/lac/ontology_engine.py`, `storage.py`)
**One frame.** `x̂ = normalize(v)` is the row stored in the journal, fed to the engine and compared with query vectors; there is no centering (since `ltir-rep-2`).

**Evidence-magnitude encoding.** The engine receives `x = w · x̂` with `w` clipped to `[1e-3, 1]`. Cosine assignment is scale-invariant, so `w` never changes *which* attractor a pattern matches; it scales the EMA pull and the OMP reconstruction loss (∝ `w²`).

**Attractor extraction** (`extract_attractors` = `_omp_extract` + `repair_extraction`), used for the first batch and for the orphan buffer:
1. Input `X = DICTIONARY_INPUT_SCALE · x` (10). sklearn's `MiniBatchDictionaryLearning` fits sparse codes with lars (`alpha = 1`) during dictionary updates; for inputs of norm ≤ 1 that penalty zeroes nearly every code and the atoms stay at their SVD initialisation. Scaling the inputs by `s` is equivalent to `alpha / s`; OMP directions and the relative error are scale-invariant.
2. K-sweep `K = DICTIONARY_K_MIN (4), +DICTIONARY_K_STEP (2), …, ≤ min(MAX_CONCEPT_COUNT (40), n)`; codes by OMP with `CONCEPTS_PER_CHUNK = 1` non-zero per row:
   ```text
   err(K)  = ‖X − A_K D_K‖_F² / ‖X‖_F²          dead(K) = #{atoms with no |a_ij| > 1e-5} / K
   tol(K)  = RECONSTRUCTION_ERROR_TOLERANCE (0.015) + dead(K) · DEAD_CONCEPT_PENALTY (0.05)
   ```
   A step with `dead(K) > MAX_DEAD_CONCEPT_RATIO` (0.25) is skipped while no `K` has been accepted yet and ends the sweep otherwise (the last accepted `K` is kept); when `err(K_prev) − err(K) < tol(K)` the sweep stops and **keeps `K_prev`** (the elbow: the extra atoms did not pay for themselves). If no `K` was ever accepted the last attempt is used. Fewer than `DICTIONARY_K_MIN` rows → one unit atom per row.
   Local activations: per row, the top-`CONCEPTS_PER_CHUNK` atoms by `|a_ij|` with `|a_ij| > 1e-5` (`weight = |a_ij|`, the lac `engine_weight`).
3. **Signed repair** against the unit rows `x̂` (atom sign is arbitrary in OMP):
   - flip atom `j` when `Σ_{users i} cos(x̂_i, d_j) < 0`;
   - intra-extraction soft merge, atoms visited by usage: atom `j` is absorbed by an already kept atom `k` with `cos(d_j, d_k) > SOFT_MERGE_LOW` (0.85); a host's centroid becomes `normalize(Σ_{group} max(usage, 1) · d)`;
   - alignment floor: activations with `cos(x̂_i, d_j) < MIN_ACTIVATION_ALIGNMENT` (0.20) are dropped; a row left without an atom is rerouted to its best atom (`rerouted = True`; `weak = True` and weight clipped at 0 when even that is below the floor — the coverage invariant wins, consumers can tell);
   - unused atoms are removed and ids remapped to `0..K′−1`; `chunk_counts` = activations per atom.

**Adaptive assignment threshold** (`compute_adaptive_threshold`): with fewer than 10 attractors `τ = MIN_ASSIGN_THRESHOLD` (0.75 for Qwen3, 0.55 for MiniLM; calibrated per embedder, SDD 07); otherwise `τ = clip(percentile_85(off-diagonal cos(c_j, c_k)), MIN_ASSIGN_THRESHOLD, MAX_ASSIGN_THRESHOLD = 0.80)` — concept–concept similarity is a loose upper bound for pattern–concept similarity.

**Assignment** (`assign_and_update`): `sim_ij = cos(x_i, c_j)`; if `max_j sim_ij ≥ τ` the row activates every `j` among its `TOP_K_ASSIGN` (2) best with `sim_ij ≥ τ` and `sim_ij ≥ MIXTURE_RATIO (0.9) · max_j sim_ij`, each activation updating that centroid; otherwise the row is an orphan.

**EMA with concept inertia** (`update_concept_centroid`), per activation:
```text
α′ = max(0.01, CENTROID_ALPHA (0.05) / sqrt(count_j + 1)) · d_j        c_j ← normalize( (1 − α′) c_j + α′ · x_i )       count_j += 1
```
With `x_i = w_i x̂_i` the pull scales with the insight weight (test: `w = 1` moves a centroid > 3× more than `w = 0.1`).

**Guards** (no freeze, no orphan diversion):
```text
τ_density = max(DENSITY_FLOOR 0.25, DENSITY_MULTIPLE 3.0 / N_attractors)          hub threshold = 3 × the uniform share
share_j   = count_j / next_chunk_id   (before the batch)        d_j = min(1, τ_density / share_j)      (EMA damping)
Δ_j = c_j(after) − c_j(before);  if ‖Δ_j‖ > MAX_CENTROID_STEP (0.10):  c_j ← normalize(c_j(before) + (MAX_CENTROID_STEP / ‖Δ_j‖) · Δ_j)
```
Damping slows an over-represented attractor's centroid without changing which rows it accepts; the trust region bounds any single batch's move (calibrated: largest healthy move 0.010 with Qwen3, 0.023 with MiniLM, so ≥ 4× headroom).

**Orphans**: buffered; extraction runs when the buffer holds `≥ DICTIONARY_K_MIN · ORPHAN_BUFFER_MIN_FACTOR` (12) rows, or — partial flush — when this batch produced orphans and the buffer has ≥ 2 rows; a lone orphan is wired to its nearest attractor (`assign_orphans_nearest`, EMA update included). New atoms with `max_k cos(d, c_k) > SOFT_MERGE_LOW` are absorbed by that attractor (`soft_merge_orphans`): their activations are re-pointed and the host updated with the buffered vectors (`route_absorbed_activations`); the rest become new attractors.

**Mutual k-NN topology** (`calculate_knn_topology`): `k = min(RELATED_TO_PEER_COUNT (3), n−1)`; an undirected RELATED_TO edge `(j, k)` exists when each is in the other's top-`k` by cosine and `cos(c_j, c_k) > RELATED_TO_MIN_WEIGHT` (0.30); weight = cosine.

**Batch metrics** (`observability`): `orphan_rate = orphans / ingested`, `extraction_yield = kept / extracted`, `avg_degree = 2E / #attractors`, `max_concept_density_pct = 100 · max_j count_j / next_chunk_id`, `centroid_drift = mean L2 move of the centroids touched this batch`. Warnings (config-driven): `orphan_rate > WARN_ORPHAN_RATE` (0.5), `yield < WARN_MIN_EXTRACTION_YIELD` (0.1), `avg_degree` outside `WARN_AVG_DEGREE` (1, 8), hub `> τ_density`, any clamped centroid.

## 13. Activation records and graph weights (`ontology._activation_records`, `graph._activation_edges`)
```text
alignment_at_ingest = cos(x̂_i, c_j)  (final centroid of that batch)        strength = alignment · w_i
```
One record per (pattern, attractor), the best alignment kept. The snapshot recomputes `alignment = cos(x̂_i, c_j)` against the **current** centroid, so edge weights follow the living ontology; `strength = alignment · w`.

## 14. Structural plane and attractor properties
**Structural edges** (`structural.structural_edges`, within one dataset; `C(A)` = condition set):
- SPECIALIZES `A → B` when `C(B) ⊂ C(A)` strictly and no present pattern `M` has `C(B) ⊂ C(M) ⊂ C(A)` (covering relation = Hasse diagram); GENERALIZES is the inverse; props `support_ratio = n_A / n_B`, `parent_z`, `child_z` on the parent's target.
- SIBLING when `|C(A) ∖ C(B)| = |C(B) ∖ C(A)| = 1` and both differing conditions have the same attribute.
- CONTRASTS when `overlap = |C(A) ∩ C(B)| / min(|C(A)|, |C(B)|) ≥ CONTRAST_MIN_OVERLAP` (0.5) and some metric has opposite-signed shifts with `min(|z_A|, |z_B|) ≥ CONTRAST_MIN_SHIFT` (0.5); the strongest such metric is recorded; weight = overlap; `relation ∈ {specialization_reversal, sibling, overlap}`.
- TARGETS weight `= min(1, |z_k| / 3)` for the target and every shift with `|z_k| ≥ MIN_COMPONENT_Z`.

**Attractor properties** (`graph._attractor_nodes`), over members `(record_i, alignment_i, strength_i)`:
```text
signature(label) = Σ_i strength_i · coef_i(label) / max_l |coef_i(l)|  /  Σ_i strength_i        (each member's components scaled to [−1, 1] first)
evidence_mass = Σ_i strength_i        dispersion = 1 − mean_i alignment_i        mass = lac chunk_count_j
```
The label is the two signature entries with the largest `|value|` (SDD 17 §8).

## 15. Query parsing and seed scoring (`ltir/query.py`)
Tokens are `[A-Za-z0-9]+`, lower-cased. `stem(a, b)`: equal, or both ≥ 5 characters, same first 5, length difference ≤ 3.
- **Metrics**: name parts = humanised words of ≥ 3 letters. A metric matches when all parts match, or ≥ 2 parts match including one non-generic word, or its head word matches and that head is non-generic and unique among metrics (generic: median, mean, average, total, count, rate, value, score, index, …). Only the best-covered metrics (`hits / parts`) are kept.
- **Conditions**: a value matches when all its words stem-match tokens; values of ≤ 4 upper-case characters (`US`, `EU`) must appear verbatim.
- **Direction** `= sign(#positive words − #negative words)`; 0 (and ignored) for relationship questions (`correl…`, `relationship`, `coupl…`, …).
- **Components**: covariance question with ≥ 2 metrics → `("correlation between a and b", ±2.0)` (−2 for break/weaken words); otherwise `(humanize(t), direction · 2.0)` per metric, or none when there is no direction word (the phenomenon block then falls back to the question text — no "higher" is assumed).

**Seed score** per pattern (`score_pattern`):
```text
target    = 1 (primary)  |  0.7 (a shift with |z| ≥ MIN_COMPONENT_Z)  |  0.6 (covariance pair)  |  0
direction = +1 if sign(z of the first material queried metric) = query direction, −0.5 if opposite, 0 if unknown
            relationship questions: 1 for a covariance insight on a queried pair (or any pair when no metric was named), else 0
scope     = |conds ∩ wanted| / |wanted|  −  0.5 · #attributes requested with a different value
semantic  = cos(v_pattern, v_query)
score     = 0.35·target + 0.25·scope + 0.15·direction + 0.15·semantic + 0.10·w        (lexical query: a metric or a condition was recognised)
score     = 0.70·semantic + 0.30·w                                                       (otherwise)
```
Seeds: patterns in score order with `score ≥ max(SEED_MIN_SCORE (0.25), SEED_RELATIVE_MIN (0.75) · best)`, at most `SEED_TOP_K` (3), skipping direct SPECIALIZES/GENERALIZES neighbours of already chosen seeds; fallback: the best semantic match.

## 16. Transversal traversal (`traversal.traverse`)
Best-first search over the path grammar `P0 (lattice){0,h} → A (RELATED_TO){0,L} → P1 (lattice){0,h}` with `h = STRUCTURAL_HOPS` (1), `L = MAX_LATENT_HOPS` (1), path length `≤ TRAVERSAL_MAX_DEPTH` (5). Path score = seed score (floored at `1e-3`) × the product of edge factors:
```text
ACTIVATES (P → A, weight ≥ ACTIVATION_THRESHOLD 0.40):     factor = alignment
RELATED_TO (A → A, weight ≥ RELATION_THRESHOLD 0.40):      factor = weight
reverse ACTIVATES (A → P, not a seed, weight ≥ 0.40):      factor = alignment
lattice hop (SPECIALIZES/GENERALIZES out-edges, CONTRASTS): factor = STRUCTURAL_EDGE_DECAY (0.85) × overlap for CONTRASTS
```
Every factor is ≤ 1, so a positive seed score keeps the heap order meaningful. The search state is `(node, phase, structural hops used, latent hops used)` (Dijkstra): a higher-scoring arrival with less remaining budget cannot shadow one that can still expand. Per node the best path wins; node rank `= path score · w` (seeds keep their seed score); `route ∈ {seed, structural (reached in P0), transversal (reached in P1)}`; `scope_overlap = max_seed Jaccard(conditions)`; `transversal_only = transversal ∧ overlap = 0`; `structural_distance` = BFS hops over all structural edges. Result: seeds first, then by rank, `|seeds| + MAX_RETRIEVED` (12) patterns; the evidence takes the first `EVIDENCE_MAX_PATTERNS` (10).

**Baselines** (`qa.compute_baselines`): structural-only = BFS closure of the seeds over the lattice edge set to depth 5; naive text-NN = top-12 patterns by `cos(E(question), E(canonical document))`; plus the set differences reported in the UI.

## 17. Hypothesis benchmark (`experiment.run_experiment`)
Cases: every pattern `S` whose scope contains a planted scope of a multi-scope mechanism (`synth.GROUND_TRUTH`); analogues = other patterns of the same mechanism sharing no condition with `S`. Rankers: transversal (seed forced to `S`, `max_retrieved = all`), structural BFS (by hops, then weight), text-NN and vector-NN (cosine of the raw insight vectors). Per case at cut-off `k`:
```text
recall@k = hits / |analogues|        precision@k = hits / min(k, |analogues|)        MRR = 1 / rank of the first analogue (0 if none)
```
Averages over cases are reported; labels come from the planted truth, never from the measured shifts (which the vectors encode).

## 18. Sphere projection (`sphere.sphere_figure` over lac's prosphera projector)
On the stacked matrix `[P ; A]` of pattern vectors and centroids (one frame): per-feature `robust_scale` (median / IQR of the 5–95 % quantile range) → `KernelPCA(kernel = cosine, 3 components)` → centre by the mean → unit directions `u = y / ‖y‖` scaled by `minmax(log ‖y‖²) ∈ [0.1, 1]`. Points therefore lie inside the unit ball, direction from the kernel PCA, radius from the (log) spread. Marker size `3 + 6w` for patterns, `8 + 1.2 · n_patterns` for attractors.

## 19. Parameters that enter the mathematics
| Stage | Parameters (default) |
|---|---|
| search | `COMPUTE_BUDGET` 5000, `VALIDATION_BUDGET` 50, `MIN_SEARCH_DIMENSIONS` 3, `EDA_RANDOM_SEED` 42; EDA constants 0.95 (mass, corr), 1.10 (nesting), `n_min`, `ROBUST_Z_CAP` 10, `BOOTSTRAP_RESAMPLES` 20, JS 0.15, χ² 0.01/#cats, hidden shift 1.5 |
| selection / weight | `MIN_SUPPORT_ROWS` 30, `MIN_EFFECT_Z` 0.5, `MAX_P_ADJUSTED` 0.05, `MIN_STABILITY` 0.5, `MIN_EMM_SCORE` 0.08, `MIN_INSIGHT_WEIGHT` 0.2, `REDUNDANCY_JACCARD` 0.88 (pre-validation), `MAX_INSIGHTS_PER_BATCH` 200, `WEIGHT_EFFECT_REF` 1.5, `WEIGHT_CONFIDENCE_REF` 6, `WEIGHT_EMM_REF` 0.08, `WEIGHT_EXPONENTS` (.35,.25,.20,.10,.10), `WEIGHT_FLOOR` 0.05 |
| canonical / encoder | `MIN_COMPONENT_Z` 0.5, `EMM_COMPONENT_WEIGHT` 0.5, `BLOCK_WEIGHTS` (0.45, 0.55, 1.0), `EMBEDDING_TRUNCATE_DIM` 384 |
| ontology | `DICTIONARY_INPUT_SCALE` 10, `DICTIONARY_K_MIN` 4, `_STEP` 2, `MAX_CONCEPT_COUNT` 40, `CONCEPTS_PER_CHUNK` 1, `RECONSTRUCTION_ERROR_TOLERANCE` 0.015, `DEAD_CONCEPT_PENALTY` 0.05, `MAX_DEAD_CONCEPT_RATIO` 0.25, `SOFT_MERGE_LOW` 0.85, `MIN_ACTIVATION_ALIGNMENT` 0.20, `MIN/MAX_ASSIGN_THRESHOLD` 0.75/0.80 (MiniLM 0.55/0.80), `ADAPTIVE_PERCENTILE` 85, `TOP_K_ASSIGN` 2, `MIXTURE_RATIO` 0.9, `CENTROID_ALPHA` 0.05, `ORPHAN_BUFFER_MIN_FACTOR` 3, `RELATED_TO_PEER_COUNT` 3, `RELATED_TO_MIN_WEIGHT` 0.30, `DENSITY_FLOOR` 0.25, `DENSITY_MULTIPLE` 3.0, `MAX_CENTROID_STEP` 0.10 |
| graph | `CONTRAST_MIN_OVERLAP` 0.5, `CONTRAST_MIN_SHIFT` 0.5 |
| retrieval | `SEED_TOP_K` 3, `SEED_MIN_SCORE` 0.25, `SEED_RELATIVE_MIN` 0.75, `ACTIVATION_THRESHOLD` 0.40, `RELATION_THRESHOLD` 0.40, `MAX_LATENT_HOPS` 1, `STRUCTURAL_HOPS` 1, `TRAVERSAL_MAX_DEPTH` 5, `STRUCTURAL_EDGE_DECAY` 0.85, `MAX_RETRIEVED` 12, `EVIDENCE_MAX_PATTERNS` 10 |

## 20. Heuristics and known approximations
* Bonferroni over `|distinct cohorts| · m` treats overlapping subgroups as independent tests: conservative, never anti-conservative.
* Pre-validation near-duplicate pruning ranks by pass-1 `temp_index` (the EDA's own ranking), not by the post-validation weight.
* `SD` sums the three largest shifts; `temp_index` and the integrated index add clipped z-scores of incommensurable quantities (the EDA's design; monotone in each).
* The EMM reliability factor `sqrt((n − n_min)/(N − n_min))` is a shrinkage heuristic, not a standard error; the per-pair normalisation only fixes the scale across datasets.
* Weight exponents, block weights, `EMM_COMPONENT_WEIGHT` and the seed-score weights are design choices validated on the synthetic data and the benchmark, not fitted.
* OMP dictionary learning on tens of rows is far from its intended regime; the signed repair (§12) is what makes it usable for insight vectors.
* The adaptive assignment threshold only becomes adaptive with ≥ 10 attractors; the JS driver threshold 0.15 is on the natural-log distance scale.

## Testing requirements
`tests/test_discovery_contract.py` (signed shifts, planted mechanisms, determinism), `tests/test_quality.py` (factor bounds, monotonicity, the exact weight formula, rule order), `tests/test_canonical_embedding.py` (unit norms, opposite directions → negative cosine, disjoint scopes with one phenomenon → high cosine), `tests/test_ontology.py` (coverage, one attractor per planted cluster, EMA pull ∝ w, sign repair), `tests/test_structural.py`, `tests/test_traversal.py` (path score and factors, thresholds, budgets), `tests/test_e2e.py::test_hypothesis_apparatus` (ranker ordering).

## Integration points
Every formula above is owned by one module (SDD 03–10); a change to any of them updates this document, the owning SDD and — for anything that changes stored vectors or canonical text — `CANONICAL_VERSION` / `REPRESENTATION_VERSION` (SDD 06, SDD 17).

## Current implementation status
Verified against the code on 2026-10-01 (demo with Qwen3: 84 cohorts → 50 validated → 28 insights → 4 attractors; benchmark MRR 0.581, recall@3 0.521 vs ≤ 0.318 / 0.111 for the baselines).
