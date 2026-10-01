# SDD 04 — Insight model, selection policy, insight weight

## Purpose
Define the canonical `Insight` (graph `Pattern`), the explicit policy deciding which validated candidates become persistent knowledge, and the `insight_weight` through which statistical evidence shapes the latent ontology and retrieval.

## Scope
`ltir/models.py` (contracts) and `ltir/quality.py` (rules R1–R7, weight).

## Inputs
`list[Insight]` from `build_insights()` (SDD 03) — already distinct cohorts with closed-intent conditions — and `Config`.

## Outputs
`SelectionResult(kept: list[Insight], rejections: list[Rejection], stats: {input, kept, <reason>: count})`. Kept insights carry `weight`, `weight_factors`, `aliases`, and possibly `phenomenon_type="covariance"`.

## Dependencies
numpy only.

## Data contract — `Insight`
| Field | Meaning |
|---|---|
| `id` | `P-` + sha1(dataset_id, sorted condition exprs)[:12]. Deterministic, so re-runs are idempotent |
| `dataset_id`, `batch_id` | provenance keys |
| `conditions` | sorted `Condition(attribute, value)` tuple (scope / selector conjunction) |
| `expression` | exact EDA selector string, e.g. `category=='phones' AND region=='US'` |
| `target` | primary metric (largest \|shift\|; for covariance insights, the pair metric with the larger \|shift\|) |
| `shifts` | `Shift(metric, robust_z signed, local_median, global_median, global_mad, source)`, sorted by \|z\| |
| `support`, `support_fraction` | covered rows, fraction of dataset |
| `baseline`, `local`, `effect_size` | global median, subgroup median, signed robust z of `target` |
| `sd_score`, `sd_raw_score` | EDA `final_sd_score` (bootstrap-penalised), pass-1 `sd_aggregate_score` |
| `emm_score`, `volume_utility` | EDA `emm_stabilized_score / √(m(m−1))` (RMS correlation change per metric pair, SDD 03), `volume_utility` |
| `stability` | `sd_score / sd_raw_score` ∈ [0.1, 1] |
| `integrated_index` | EDA step-5 z-sum, computed by the adapter |
| `p_value`, `p_adjusted` | adapter median test, Bonferroni-adjusted |
| `drivers` | EDA `root_cause_drivers` (confounders) |
| `row_hash` | sha1 of sorted covered row positions (cover identity) |
| `phenomenon_type` | `shift` or `covariance` |
| `covariance` | `{pair, local_corr, global_corr, delta}` |
| `aliases` | selectors merged into this cohort before validation (R5 identical extent, R6 near duplicate; SDD 03) |
| `weight`, `weight_factors` | see below |
| `provenance` | dataset_id, filename, batch_id, engine, steps, expression, `rows_ref`, `multiple_testing_family` |

Serialisation: `Insight.to_record()` / `Insight.from_record()` (JSON-safe).

## Selection rules (fixed order)
| Rule | Condition to pass | Rejection reason |
|---|---|---|
| — | `significant = \|effect_size\| ≥ MIN_EFFECT_Z ∧ p_adjusted ≤ MAX_P_ADJUSTED`; `stable = stability ≥ MIN_STABILITY`; `shift_ok = significant ∧ stable`; `emm_ok = emm_score ≥ MIN_EMM_SCORE ∧ covariance pair defined`. If `shift_ok` fails but `emm_ok` holds, retype as `covariance` and retarget on the pair (an absent, insignificant **or unstable** shift is never cited as a median shift) | — |
| R1 support | `support ≥ MIN_SUPPORT_ROWS` (on top of the EDA's own min size) | `min_support` |
| R2 / R3 strength and stability | `shift_ok ∨ emm_ok` | `unstable` (significant but not stable), `not_significant` (\|z\| large enough, p fails), `weak_effect` (otherwise) |
| R4 weight | `weight ≥ MIN_INSIGHT_WEIGHT` | `low_weight` |
| R5 cover equivalence | runs in discovery **before validation** (SDD 03): one cohort per extent, described by its closed intent | `cover_equivalent` |
| R6 near duplicate | runs in discovery before validation: same primary metric and sign, row Jaccard ≥ `REDUNDANCY_JACCARD` → alias of the higher-ranked cohort | `near_duplicate` |
| R7 budget | keep the top `MAX_INSIGHTS_PER_BATCH` by weight | `budget` |

`select_insights(insights, config)` applies the retype, R1–R4 and R7. Discovery and selection rejections are persisted together to `datasets/<id>/rejections.json` and counted in batch metrics (`pruned`).

## Insight weight
Inputs: EDA effect, bootstrap stability, adapter significance, EDA volume utility, EDA EMM score.

Normalisation to factors in [0, 1] (dataset-independent, monotone):

| Factor | `shift` insight | `covariance` insight |
|---|---|---|
| effect | 1 − exp(−\|z\| / `WEIGHT_EFFECT_REF`) | 1 − exp(−emm / `WEIGHT_EMM_REF`) |
| stability | `stability` | `None` — not measured by the EDA for a correlation change; left out of the mean |
| confidence | clip(−log10(p_adj) / `WEIGHT_CONFIDENCE_REF`, 0, 1) | `None` — same |
| support | volume_utility / max(√p(1−p)) (max at p = 1/3, = 0.3849) | same |
| emm | 0.5 + 0.5·(1 − exp(−emm / `WEIGHT_EMM_REF`)) (a bonus that never zeroes the weight) | same |

Combined (weighted geometric mean, exponents `WEIGHT_EXPONENTS` = effect .35, stability .25, confidence .20, support .10, emm .10):

```text
w = WEIGHT_FLOOR + (1 − WEIGHT_FLOOR) · Π_{k measured} max(f_k, 1e-3) ^ (a_k / Σ_{k measured} a_k)
```

The exponents are renormalised over the *measured* factors. Substituting a constant such as 0.5 for an unmeasured factor is not neutral in a product (it would cost every covariance insight a fixed 0.5^0.45 = 0.73), so unmeasured factors are excluded instead.

Where the weight enters:
1. **Ontology** (SDD 07): evidence-magnitude encoding `x = w·x̂`. Assignment is cosine, so unaffected. The EMA pull on a centroid scales with w, and OMP reconstruction weights rows by w.
2. **Activation strength**: `strength = alignment · w` on ACTIVATES edges; attractor `evidence_mass = Σ strength`.
3. **Retrieval ranking** (SDD 10): node score = path score × w; seed score includes 0.10·w.
4. **Selection**: R4.

## Configuration
`MIN_SUPPORT_ROWS` 30, `MIN_EFFECT_Z` 0.5, `MIN_EMM_SCORE` 0.08, `MIN_STABILITY` 0.5, `MAX_P_ADJUSTED` 0.05, `MIN_INSIGHT_WEIGHT` 0.2, `REDUNDANCY_JACCARD` 0.88 (SDD 03), `MAX_INSIGHTS_PER_BATCH` 200, `WEIGHT_EFFECT_REF` 1.5, `WEIGHT_CONFIDENCE_REF` 6, `WEIGHT_EMM_REF` 0.08, `WEIGHT_EXPONENTS` (.35,.25,.2,.1,.1), `WEIGHT_FLOOR` 0.05. `MIN_EMM_SCORE` and `WEIGHT_EMM_REF` are on the per-pair RMS scale: 0.08 after shrinkage corresponds to a raw RMS correlation change of ≈ 0.2–0.4 per pair for subgroups holding 5–15 % of the rows (the planted EU∧phones break scores 0.086).

## Failure modes
If nothing is kept, the pipeline raises `no_viable_insights` (batch FAILED, nothing persisted).

## Invariants
* `WEIGHT_FLOOR ≤ weight ≤ 1`, and the weight is monotone non-decreasing in \|z\|, stability, −log p and EMM. Unmeasured factors are `None` in `weight_factors` and never enter the mean.
* No two kept insights share a `row_hash` or an `id` (identical extents were merged before validation).
* Every validated candidate (selection input) is either kept or has exactly one `Rejection`. Pass-1 candidates beyond `VALIDATION_BUDGET` never reach selection.

## Testing requirements
`tests/test_quality.py`: bounds and monotonicity, the formula, reason codes, R7 budget, covariance retargeting. R5/R6: `tests/test_discovery_contract.py`.

## Integration points
Consumed by canonicalisation (05), encoder weights (07), graph properties (08), traversal ranking (10).

## Current implementation status
Implemented. Synthetic data: 50 validated → 28 kept (22 `weak_effect`), 2 of them `covariance`: the planted discount–margin correlation break in EU∧phones (0.086) and a mixture effect in tablets∧online (discount and delivery days co-move there because APAC rows carry both planted shifts, 0.14).
