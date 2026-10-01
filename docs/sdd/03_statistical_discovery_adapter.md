# SDD 03 — Statistical discovery adapter

## Purpose
Use the existing automatic-EDA engine (`main_upd.py`, vendored as `ltir/engines/eda/main_upd.py` from `eda/scripts/`) as the single source of statistical discovery. Expose its results through a typed, stable contract (`Dataset → Profile → Candidates → Validated insights`).

## Scope
`ltir/discovery.py`. The adapter **calls** the EDA functions and adds only what they do not produce.

## Inputs
`pandas.DataFrame` from SDD 02 and `Config`.

## Outputs
* `DiscoveryResult(profile: DatasetProfile, candidates: list[Candidate], validated: list[Candidate], data: DataFrame, n_tests: int)` from `run_discovery()`.
* `list[Insight]` (unfiltered, weight unset) from `build_insights()`.
* `covers_of(result) -> {expression: row positions}`.

## Dependencies
`ltir.engines.eda.main_upd` (pysubgroup, scipy, pandas), imported directly. There is no dependency on the original `eda/` folder.

## Reused EDA behaviour (algorithms as vendored, with the numerical repairs listed below)
| Step | EDA function | Behaviour |
|---|---|---|
| Profile | `step1_profile_data` | drop identifier columns (non-float, all-unique); drop constant categoricals; nested-column pruning (unique pairs ≤ 1.10·max cardinality → the finer column is dropped, only when the coarser one is informative: top level < 95 %); numeric zero-variance and \|corr\| > 0.95 pruning; global correlation matrix |
| Macro screen | `step2_evaluate_macro_groupings` | cardinality ≤ √N; power = ε² = (SS_between − (k−1)·MS_within) / SS_total per (category, metric), averaged over metrics (η² is biased upwards by (k−1)/(n−1), which favours high-cardinality columns); keep categories above the median power (max 6) |
| Search space | `step3_generate_search_space` | per dimension, the levels up to and including the one that crosses 95 % cumulative mass (only the long tail is dropped; a 60/40 column keeps both levels); 2-conjunctions, plus 3-conjunctions within `compute_budget`; above the budget the 2-conjunctions with the largest expected support (product of level shares) are kept instead of an empty space |
| Pass 1 | `step4_evaluate_micro_slices` | min size max(5·#numerics, 0.5 % N); robust z = min(0.6745·\|Δmedian\| / scale, 10) with scale = MAD, or the normal-consistent mean absolute deviation (0.845·mean\|x − med\|) when the MAD is 0 (more than half of the values tie); SD = top-3 shift sum; EMM = ‖C_local − C_global‖_F over the entries defined in the subgroup (a metric constant inside the subgroup contributes no change) · √((n−n_min)/(N−n_min)); volume = √p(1−p) |
| Ranking | workflow `temp_index` | Σ clip₊(zscore) of SD, EMM, volume → top `VALIDATION_BUDGET` |
| Pass 2 | `step4b_deep_validation` | 20 bootstrap resamples (n of n, with replacement) of the top-shift sum; `final_sd = sd·(1 − min(CV, 0.9))`; drivers: a categorical whose subgroup distribution has JS distance > 0.15 **and** a chi-square test of independence (subgroup vs rest) with p < 0.01 / #categoricals, named by the level whose share grew most; numeric metrics outside the top-3 with a signed hidden shift > 1.5 robust σ |
| Index | step-5 formula | `integrated_index` = Σ clip₊(zscore) of final SD, EMM, volume over the validated set (step 5 only prints it, so the adapter computes it) |

## Edits made to `main_upd.py` (each is a local, commented change; the workflow and Spyder cells are unchanged)
1. The data load and workflow run under `if __name__ == "__main__":` (the module is importable).
2. Identifier rule: float columns are never identifiers (continuous targets are all-unique).
3. `step2_evaluate_macro_groupings(profile, min_categories=0)`. The default keeps behaviour; `>0` backfills categories by power rank so narrow schemas still yield conjunctions.
4. Constant categorical columns are removed before the overlap rule (a single-valued column is "redundant" with every other column and used to drop the informative one — the `Over18` failure).
5. Overlap rule fires only when the retained (coarser) column is informative (top level < 95 %).
6. `calculate_mad` falls back to the normal-consistent mean absolute deviation when the MAD is 0; `robust_z_score` is capped at 10 (bounded influence: a degenerate scale cannot dominate the ranking).
7. Step 2 uses ε² instead of η².
8. Step 3 keeps the level that crosses 95 % mass and truncates an over-budget space by expected support instead of returning nothing.
9. EMM ignores correlations undefined inside the subgroup instead of reading them as 0.
10. Bootstrap: 20 full-size resamples instead of 5 resamples of 90 %.
11. Confounder drivers need a significant skew (chi-square) and name the level with the largest share gain; hidden shifts are reported signed in robust σ.

## Adapter additions
| Addition | Definition | Why |
|---|---|---|
| Conditions | `(attribute_name, str(attribute_value))` of the pysubgroup selectors, sorted | typed scope for the lattice (SDD 08) |
| Signed shift | `robust_z = sign(median_local − median_global) · EDA magnitude` | EDA returns \|·\|; direction is needed for CONTRASTS and phenomenon vectors |
| EMM scale | `emm_score = emm_stabilized / √(m(m−1))`, m = number of metrics | the Frobenius norm grows with the number of off-diagonal entries; the normalised value is the RMS correlation change per metric pair (∈ [0, 2], after the EDA's reliability shrinkage), so `MIN_EMM_SCORE` / `WEIGHT_EMM_REF` mean the same on every dataset |
| Covariance pair | `argmax_{i<j} \|C_local[i,j] − C_global[i,j]\|` over pairs defined in the subgroup (NaN-aware; `{}` when none) | explains *which* relation drives the EMM score |
| Pair shifts | robust z of pair metrics missing from the EDA top-3 (`Shift.source="covariance_pair"`) | lets EMM-only insights be targeted (SDD 04) |
| Significance | two-sided asymptotic median test: `z = (m_l − m_g)/(1.2533·1.4826·MAD_l/√n)`; `p_adjusted = min(1, p·n_tests)` (Bonferroni over candidates × metrics) | the EDA has no p-values; used for the selection rule and weight |
| Stability | `final_sd / sd_raw = 1 − min(CV, 0.9)` | recovers the bootstrap factor that step4b folds into `final_sd` |
| Determinism | `np.random.seed(EDA_RANDOM_SEED)` before step4b | `DataFrame.sample` uses the global RNG |
| Provenance | dataset, filename, batch, engine path, steps, exact selector string, `rows_ref`, testing family size | traceability requirement |

## Data contracts
`Candidate(expression, conditions, row_indices, row_count, volume_utility, top_shifts[(metric, |z|)], sd_aggregate_score, emm_stabilized_score, temp_index, validated, final_sd_score, drivers)`. `Insight` is defined in SDD 04. The primary target is the EDA's largest shift.

## Configuration
`COMPUTE_BUDGET` (5000, EDA default), `VALIDATION_BUDGET` (50, EDA default), `MIN_SEARCH_DIMENSIONS` (3), `EDA_RANDOM_SEED` (42).

## Failure modes
`DiscoveryError(code)`: `no_numeric_targets` (step1 left no numerics), `invalid_schema` (no categorical dimension; suggests `BIN_COLUMNS`), `no_candidates` (empty search space or nothing passed pass 1).

## Invariants
* `0 < len(candidates) ≤ profile.search_space_size` (the difference is the slices rejected by the pass-1 size screen, e.g. rare levels).
* `len(validated) ≤ VALIDATION_BUDGET`, and every validated candidate appears in `candidates`.
* The same data and config give identical insights (seeded bootstrap).
* `Insight.support == len(covers[expression])`.

## Testing requirements
`tests/test_discovery_contract.py`: profile contract, field and provenance contract, planted directions, determinism, planted phenomena kept, failure codes.

## Integration points
`Engine.process()` stages PROFILING → DISCOVERING (`on_stage` callback) → VALIDATING_INSIGHTS.

## Current implementation status
Implemented. Measured on the synthetic data: 88 conjunctions, 84 candidates (4 rare `partner` slices fail the size screen), 50 validated (`VALIDATION_BUDGET`), 28 kept; no confounders are reported because the demo dimensions are drawn independently (before the chi-square gate, JS noise on small slices produced spurious drivers). All six planted mechanisms are recovered (`tests/test_discovery_contract.py`).
