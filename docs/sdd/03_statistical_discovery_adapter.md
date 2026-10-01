# SDD 03 — Statistical discovery adapter

## Purpose
Use the existing automatic-EDA engine (`main_upd.py`, vendored as `ltir/engines/eda/main_upd.py` from `eda/scripts/`) as the single source of statistical discovery. Expose its results through a typed, stable contract (`Dataset → Profile → Candidates → Validated insights`).

## Scope
`ltir/discovery.py`. The adapter **calls** the EDA functions and adds only what they do not produce.

## Inputs
`pandas.DataFrame` from SDD 02 and `Config`.

## Outputs
* `DiscoveryResult(profile, candidates: list[Candidate], validated: list[Candidate], data: DataFrame, n_tests: int, pass1_subgroups: int, rejections: list[Rejection])` from `run_discovery()`; `candidates` are the distinct cohorts (one per extent), `rejections` the `cover_equivalent` / `near_duplicate` merges.
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
| Ranking | workflow `temp_index` | Σ clip₊(zscore) of SD, EMM, volume over the **distinct** cohorts → near-duplicate pruning → top `VALIDATION_BUDGET` (see *Deduplication before validation*) |
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

## Deduplication before validation (R5, R6)
The bootstrap of step 4b is the expensive stage and `VALIDATION_BUDGET` slots are scarce, so redundant cohorts are removed **between pass 1 and the ranking into step 4b** (`run_discovery`):
1. **Closure (R5)** — `merge_identical_extents`: pass-1 selectors with the same extent (same covered rows) become one cohort described by its closed intent `closed_intent(data, rows, categoricals)` = every (attribute, value) that holds on all covered rows, over the profiled categoricals (post step-1 pruning; null-like levels excluded). `expression` keeps the member selector with most conditions; the others become `aliases` and `Rejection(…, "cover_equivalent")`.
2. `temp_index` over the distinct cohorts.
3. **Near duplicates (R6)** — `prune_near_duplicates`, greedy in `temp_index` order: a cohort with the primary metric (pass-1 top shift) and sign of a higher-ranked kept cohort and row Jaccard ≥ `REDUNDANCY_JACCARD` (0.88, boolean-mask intersection) is absorbed as its alias → `Rejection(…, "near_duplicate")`.
4. The first `VALIDATION_BUDGET` survivors go to step 4b; the closed intent's attributes are passed as the cohort's scope, so implied attributes are never reported as confounders.

`n_tests` (Bonferroni family) = distinct cohorts × metrics: identical extents are one test.

## Adapter additions
| Addition | Definition | Why |
|---|---|---|
| Conditions | closed intent of the extent (superset of the selector's `(attribute_name, str(attribute_value))`), sorted | typed scope for the lattice (SDD 08); identical cohorts get identical ids |
| Signed shift | `robust_z = sign(median_local − median_global) · EDA magnitude` | EDA returns \|·\|; direction is needed for CONTRASTS and phenomenon vectors |
| EMM scale | `emm_score = emm_stabilized / √(m(m−1))`, m = number of metrics | the Frobenius norm grows with the number of off-diagonal entries; the normalised value is the RMS correlation change per metric pair (∈ [0, 2], after the EDA's reliability shrinkage), so `MIN_EMM_SCORE` / `WEIGHT_EMM_REF` mean the same on every dataset |
| Covariance pair | `argmax_{i<j} \|C_local[i,j] − C_global[i,j]\|` over pairs defined in the subgroup (NaN-aware; `{}` when none) | explains *which* relation drives the EMM score |
| Pair shifts | robust z of pair metrics missing from the EDA top-3 (`Shift.source="covariance_pair"`) | lets EMM-only insights be targeted (SDD 04) |
| Significance | two-sided asymptotic median test: `z = (m_l − m_g)/(1.2533·1.4826·MAD_l/√n)`; `p_adjusted = min(1, p·n_tests)` (Bonferroni over candidates × metrics) | the EDA has no p-values; used for the selection rule and weight |
| Stability | `final_sd / sd_raw = 1 − min(CV, 0.9)` | recovers the bootstrap factor that step4b folds into `final_sd` |
| Determinism | `np.random.seed(EDA_RANDOM_SEED)` before step4b | `DataFrame.sample` uses the global RNG |
| Provenance | dataset, filename, batch, engine path, steps, exact selector string, `rows_ref`, testing family size | traceability requirement |

## Data contracts
`Candidate(expression, conditions (closed intent), row_indices, row_count, volume_utility, top_shifts[(metric, |z|)], sd_aggregate_score, emm_stabilized_score, temp_index, validated, final_sd_score, drivers, aliases)`. `Insight` is defined in SDD 04. The primary target is the EDA's largest shift.

## Configuration
`COMPUTE_BUDGET` (5000, EDA default), `VALIDATION_BUDGET` (50, EDA default), `MIN_SEARCH_DIMENSIONS` (3), `EDA_RANDOM_SEED` (42), `REDUNDANCY_JACCARD` (0.88).

## Failure modes
`DiscoveryError(code)`: `no_numeric_targets` (step1 left no numerics), `invalid_schema` (no categorical dimension; suggests `BIN_COLUMNS`), `no_candidates` (empty search space or nothing passed pass 1).

## Invariants
* `0 < len(candidates) ≤ pass1_subgroups ≤ profile.search_space_size`; `pass1_subgroups − len(candidates)` = the `cover_equivalent` rejections.
* `len(validated) ≤ VALIDATION_BUDGET`; validated cohorts have pairwise distinct extents; every validated candidate appears in `candidates`.
* Galois antitone: for validated cohorts, a strictly smaller extent has a strictly larger intent.
* The same data and config give identical insights (seeded bootstrap).
* `Insight.support == len(covers[expression])`.

## Testing requirements
`tests/test_discovery_contract.py`: profile contract, field and provenance contract, planted directions, determinism, planted phenomena kept, `closed_intent`, identical extents merged before validation (planted implied column), near-duplicate pruning order, extent/intent antitone property, failure codes.

## Integration points
`Engine.process()` stages PROFILING → DISCOVERING (`on_stage` callback) → VALIDATING_INSIGHTS.

## Current implementation status
Implemented. Deduplication effect (validation slots that would have gone to duplicate cohorts): demo, `housing.csv`, `Walmart_adapted.csv` — 0 (step-1 nesting pruning already removes their dependent columns); HR attrition (`WA_Fn-UseC_-HR-Employee-Attrition.csv` with `--categories Education,JobLevel,StockOptionLevel,EnvironmentSatisfaction,JobSatisfaction`) — 143 pass-1 subgroups → 135 cohorts (8 merged by closure) and 9 near-duplicates pruned, so 17 bootstrap slots go to distinct cohorts; 8 cohorts gain implied conditions; before the change 3 of the 50 validation slots held duplicate extents. Measured on the synthetic data: 88 conjunctions, 84 candidates (4 rare `partner` slices fail the size screen), 50 validated (`VALIDATION_BUDGET`), 28 kept; no confounders are reported because the demo dimensions are drawn independently (before the chi-square gate, JS noise on small slices produced spurious drivers). All six planted mechanisms are recovered (`tests/test_discovery_contract.py`).
