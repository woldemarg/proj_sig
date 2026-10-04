# 2. Discovery — from a table to validated subgroups

> **In one paragraph.** Ingestion validates the uploaded file, optionally turns number-coded columns into categories and numeric columns into quantile bands, and gives the dataset a content-addressed id. The vendored automatic-EDA engine then searches conjunctions of two and three `attribute = value` selectors in two passes: a cheap screen for robust median shifts and correlation changes, and an expensive bootstrap validation of the best 50. Between the passes the adapter merges selectors that cover identical rows and prunes near-duplicate cohorts, so the bootstrap is spent only on distinct subgroups. The adapter also adds what the EDA does not produce — the direction of each shift, a significance test, provenance — and hands typed candidates to [3. Insights](03_insights.md).

**Code** `subgroup_miner/ingestion.py`, `subgroup_miner/discovery.py`, `subgroup_miner/vendor/eda/main_upd.py`, `subgroup_miner/config.py` (`MinerConfig`) · **Tests** `tests/test_discovery_contract.py`, `tests/test_persistence.py` (ingestion failures, column options), `tests/test_subgroup_miner_standalone.py` · **Previous** [1. Overview](01_overview.md) · **Next** [3. Insights](03_insights.md)

---

## 2.1 Ingestion

`load_dataset(path, config, filename=, bins=, categories=)` (`config` a `MinerConfig`) turns a file into a `LoadedDataset(frame, dataset_id, filename, bins, categories, derived_columns, warnings)`.

| Input | Form | Source |
|---|---|---|
| file | `.csv`, `.tsv`, `.txt`, `.parquet` | `POST /api/upload` (the console's upload; saved under `WORKSPACE_DIR/uploads/`), `POST /api/demo`, or a path given to `Engine.ingest_file` (the measurement scripts, the tests) |
| `bins` | `"col:q,col2:q"` (`col` alone means 4 quantiles); `""` = none; `None` = workspace default `BIN_COLUMNS` | UI *Split numbers into bands* = the form field `bins` of `POST /api/upload` |
| `categories` | `"col,col2"`; `""` = none; `None` = workspace default `CATEGORICAL_COLUMNS` | UI *Treat as categories* = the form field `categories` |

An empty or missing upload field means "workspace default" (`None`); `POST /api/demo` (the UI's *Try demo*) sends an explicit `""` for both options; `Engine.submit` takes the options (`None` unless one is passed; `bins=""` is an explicit "none"), and `Engine.ingest_file(path)` submits with the workspace defaults.

Before a batch exists, `Engine.upload` refuses a file above `MAX_UPLOAD_MB` (200): it deletes the upload and raises `file_too_large`, which the API answers with HTTP 413. Steps, in order:

1. **File.** A missing file → `unsupported_file`.
2. **Options.** The band and category specs are parsed before the file is read; a malformed spec → `invalid_options`.
3. **Type and parse.** Unknown extensions → `unsupported_file`. CSV and `.txt` use delimiter sniffing (`sep=None`, python engine), TSV a tab, Parquet `read_parquet` (without pyarrow installed: `unreadable_file`). Parser errors → `unreadable_file`. pandas' default NA parsing applies: a literal `NA` becomes missing, which is why the synthetic region is called `US`, not `NA`.
4. **Shape.** Fewer than 2 columns or fewer than `MIN_ROWS` (50) rows → `invalid_schema`.
5. **Categories, then bands.** A *categorical override* casts a column to text (`Int64` first when it holds integral floats, so `1.0` becomes `"1"`; NaN becomes `missing`): this is how integer-coded dimensions such as `Store` or `Holiday_Flag` become scope columns instead of metrics or identifiers. A *band* derives `<col>_band` with values `q1 … qk` (quantile bins, duplicate edges dropped) plus `missing`, and **drops the source column**, so the band acts as a dimension and can never become a tautological target (`median_income` shifting inside `median_income_band = q4`). Workspace defaults (`None`) are applied leniently — columns the dataset lacks are skipped with one warning — while an explicit option is strict: an unknown column → `invalid_options`. A band on a non-numeric column is `invalid_options` in both modes; since categories are applied first, that includes a column named in both options (it is already text when the band is made). Because NaN becomes the level `missing` (not a null), `x_band = missing` can be a selector like any other value.
6. **Targets.** No numeric column → `no_numeric_targets`. All-null columns produce a warning.

The dataset id is a pure function of the bytes and the applied options:

```text
dataset_id = "ds-" + sha256( file bytes ‖ repr(sorted(bins.items())) [‖ repr(sorted(categories)) when any apply] )[:12]
```

Identical content with identical options always gets the same id; a second upload of it is `SKIPPED` ([9.1](09_operations.md#91-the-batch-lifecycle)). The frame keeps the file's row order, so EDA row positions are file rows. The file itself is not copied: an upload stays under `uploads/`, and the batch record's `source_path` names it (provenance; [6.3](06_graph_and_storage.md#63-the-workspace-on-disk)). Semantic typing (identifiers, categoricals, redundant columns) is not done here: it is EDA step 1.

> **Running example.** `retail_synthetic.csv`: 5,000 rows × 11 columns, no bands, no overrides → `ds-6e53eb7fb0f9`. On real data the options matter: `housing.csv` needs `bins=median_income:4,housing_median_age:4` (only one native categorical).

## 2.2 The EDA engine in five steps

The adapter (`discovery.run_discovery(df, config, on_stage=None)`; `on_stage("DISCOVERING")` is called once profiling and the search space are done, which moves the batch from PROFILING to DISCOVERING) calls the vendored engine step by step, in the order of the EDA's own workflow, and converts its DataFrames into typed objects; downstream of discovery the pipeline reads only typed candidates, the profile, the rejections and the covers. Two process-wide side effects come with the engine: importing it silences `RuntimeWarning`, and the adapter reseeds NumPy's global RNG before the bootstrap. Notation for this chapter:

| Symbol | Meaning |
|---|---|
| `N` | rows of the profiled table |
| `m` | metric (numeric) columns kept by profiling |
| `S`, `n = |S|`, `p = n / N` | a subgroup (the rows covered by a conjunction of selectors), its size and share |
| `med`, `MAD` | median; median absolute deviation around the median |
| `z_k(S)` | the robust shift of metric `k` inside `S` |
| `C`, `C_S` | Pearson correlation matrix of the metrics over all rows / over `S` |

### Step 1 — profile the table (`step1_profile_data`)
1. **Identifiers** are dropped: a column with `nunique == count(non-null)` whose dtype is not float (continuous measures are all-unique by nature, so floats are never identifiers).
2. **Typing**: numerics are numpy numeric dtypes; categoricals are object / category / bool / string, cast to `str` — a null categorical value becomes the string `nan`, which counts in the checks below and can be named as a confounder level but never becomes a selector. Columns of any other dtype (dates, for example) are dropped.
3. **Constant categoricals** (`nunique ≤ 1`) are dropped before any pair rule — a single-valued column is trivially "a function of" every other column.
4. **Nested columns**: for a pair of categoricals with `|distinct (a, b) pairs| ≤ 1.10 · max(nunique)`, the finer column is dropped (on a tie, the later one), but only when the coarser one is informative (its top level holds < 95 % of rows).
5. **Numerics**: zero-variance columns are dropped; then a column is dropped when any earlier column has `|corr| > 0.95` with it (also an earlier column that was itself dropped). `C` is computed over the kept metrics (`NaN → 0`).

### Step 2 — choose the dimensions (`step2_evaluate_macro_groupings`)
For each categorical `g` with `nunique(g) ≤ √N` and each metric `y`, over the rows where both are present (`k ≥ 2` groups, `n > k`):

```text
SS_between = Σ_groups n_i (ȳ_i − ȳ)²      SS_total = Σ (y − ȳ)²      MS_within = (SS_total − SS_between) / (n − k)
power(g, y) = ε² = max(SS_between − (k − 1) · MS_within, 0) / SS_total
```

ε² is η² = SS_between / SS_total with its chance expectation `(k − 1)/(n − 1)` removed, so a 45-level column does not outrank a 3-level one by chance. `power(g)` is the mean over metrics; columns above the median power are kept (at most 6), and if fewer than `MIN_SEARCH_DIMENSIONS` (3) survive, the strongest remaining eligible ones are back-filled so that 2- and 3-conjunctions exist. A column with more than `√N` levels is never eligible, not even for the back-fill.

### Step 3 — build the search space (`step3_generate_search_space`)
Per selected dimension, levels are ordered by share and kept up to and including the one that crosses 95 % cumulative mass — only the long tail is dropped, so a 60/40 column keeps both levels; `nan` / `<NA>` / `None` levels are skipped. Candidates are all conjunctions of 2 selectors on distinct attributes, plus the 3-conjunctions when the total stays within `COMPUTE_BUDGET` (5000). When even the 2-conjunctions exceed the budget, the pairs with the largest expected support `Π f_v` (product of level shares) are kept. Single selectors are never candidates: single-attribute effects are what step 2 measures.

### Step 4 — pass-1 screening (`step4_evaluate_micro_slices`)
A subgroup is screened when `n ≥ n_min = max(5m, ⌊0.005 N⌋)` and `n ≠ N`.

**Robust scale and shift.**

```text
MAD(y) = med(|y − med(y)|);   if MAD = 0:   MAD := (0.6745 / 0.7979) · mean(|y − med(y)|)
|z_k(S)| = min( 0.6745 · |med_S(y_k) − med(y_k)| / (MAD_k + 1e-6),  ROBUST_Z_CAP = 10 )
```

`0.6745 · Δ / MAD = Δ / σ̂` with `σ̂ = MAD / 0.6745`: the shift is measured in robust standard deviations ("sd" everywhere in the UI and the prompt). The MAD fallback is the normal-consistent mean absolute deviation; without it a metric with more than half of its values tied (zero-inflated counts) made every subgroup look infinitely shifted. The cap keeps a degenerate scale from dominating the ranking. The scale `MAD_k` is the global one, so shifts are comparable across subgroups.

**Scores.**

```text
SD(S)      = sum of the three largest |z_k(S)|                                   (top_shifts)
Δ          = C − C_S, with Δ_ij := 0 where C_S,ij is undefined (metric constant in S); only when n ≥ 3m and m ≥ 2
EMM_eda(S) = ‖Δ‖_F · sqrt( max(0, n − n_min) / (N − n_min) )                   (reliability shrinkage)
vol(S)     = √p · (1 − p)                                                        (maximal at p = 1/3: 0.3849)
```

`SD` rewards median shifts, `EMM` rewards a changed correlation structure (exceptional model mining), `vol` rewards moderate, actionable subgroups over near-majorities. The shrinkage pulls correlations estimated from few rows towards "no change"; an undefined local correlation is no evidence of change, not a change to 0.

### Step 4b — pass-2 validation (`step4b_deep_validation`)
**Bootstrap stability.** For `B = 20` resamples of `S` (n of n, with replacement; the adapter seeds the global RNG with `EDA_RANDOM_SEED`):

```text
b_r = Σ_{k ∈ top_shifts} |z_k(S_r)|        CV = std(b) / (mean(b) + 1e-6)
final_sd = SD(S) · (1 − min(CV, 0.9))      stability = final_sd / SD(S) = 1 − min(CV, 0.9) ∈ [0.1, 1]   (0 if SD(S) = 0)
```

The adapter seeds the RNG once before the whole step, so a cohort's resamples depend on its position in the validation order; the result is reproducible for the same data and configuration.

**Confounders** (categoricals outside the subgroup's own scope). With `q_S`, `q` the level distributions inside `S` and overall, a column is a driver when `jensenshannon(q_S, q) > 0.15` (scipy's JS *distance*, natural log) **and** the chi-square test of independence (rows of `S` vs the rest) has `p < 0.01 / #categoricals`. The driver is named by the level whose share grew most. The significance gate matters: the JS distance of an empirical distribution is inflated by sampling noise in small subgroups. **Hidden shifts**: a metric outside the top three with `|z| > 1.5` is reported, signed. At most three drivers per subgroup — categorical drivers come first, so the cut drops hidden shifts first — as strings such as `[payment] heavily skewed to 'cash' (JS: 0.20)` or `[discount] hidden shift (+1.8 robust sigma)`.

> **Running example.** Profiling keeps the numerics `discount, margin, delivery_days, return_rate` and the categoricals `region, category, channel, payment, weekday, store_size`, and drops `order_id` as an identifier. Step 2 keeps `region, category, channel`; the planted noise columns `payment, weekday, store_size` fall below the median power. The search space has 88 conjunctions; 84 pass the size screen (four rare `partner` slices do not); the top 50 are validated. `category=='phones' AND region=='US'` covers 438 rows, its top shifts are discount and margin, its bootstrap stability is 0.974, and it has no confounders (the demo dimensions are drawn independently).

The vendored engine differs from upstream in a set of local, commented numerical repairs — the MAD fallback and cap, constant categoricals dropped before the nesting rule (which needs an informative coarser column), ε² instead of η², the 95 %-mass rule, the over-budget truncation, NaN-aware EMM, the full-size bootstrap, confounders gated by a chi-square test and named by the level with the largest share gain, signed hidden shifts — each listed with its reason in [`subgroup_miner/vendor/PROVENANCE.md`](../subgroup_miner/vendor/PROVENANCE.md).

## 2.3 Deduplication before validation

The bootstrap of step 4b is the expensive stage and `VALIDATION_BUDGET` (50) slots are scarce. Two selectors that cover exactly the same rows are one subgroup described twice, and two cohorts that overlap almost completely with the same effect are one finding. `run_discovery` therefore removes redundancy **between pass 1 and the ranking into step 4b**:

1. **Closure** (`merge_identical_extents`). Pass-1 selectors are grouped by extent (the set of covered rows). Each group becomes one cohort described by its **closed intent**

   ```text
   int(ext(S)) = { (a, v) : a is a profiled categorical, every covered row has a = v, v is not null-like }
   ```

   — the Galois closure of formal concept analysis: every condition that holds on all of the cohort's rows, including conditions the selector did not state (an implied column). `expression` keeps the member selector with the most conditions; the other selectors become `aliases` and `Rejection(…, "cover_equivalent")`.
2. **Ranking.** `temp_index = clip₊(zscore(SD)) + clip₊(zscore(EMM_eda)) + clip₊(zscore(vol))` over the distinct cohorts (scipy's population z-score, `NaN → 0`) — the EDA workflow's own ranking.
3. **Near duplicates** (`prune_near_duplicates`), greedy in `temp_index` order: a cohort is absorbed as an alias of a higher-ranked kept cohort with the same primary metric (pass-1 top shift), the same sign of `med_S(y) − med(y)` and

   ```text
   J(A, B) = |A ∩ B| / |A ∪ B| ≥ REDUNDANCY_JACCARD (0.88)        (boolean row masks)
   ```

   → `Rejection(…, "near_duplicate")`.
4. **Validation.** The first `VALIDATION_BUDGET` survivors go to step 4b, with the closed intent's attributes passed as their scope, so an implied attribute is never reported as a confounder of its own cohort.

The Bonferroni family is `n_tests = distinct cohorts × m`: identical extents are one test. Because conditions are closed intents, the lattice of [6.1](06_graph_and_storage.md#61-the-structural-plane) is the Hasse diagram of a concept lattice: a strictly smaller extent always has a strictly larger intent.

## 2.4 What the adapter adds

| Addition | Definition | Why |
|---|---|---|
| conditions | the closed intent, sorted `(attribute, str(value))` | a typed scope for the lattice; identical cohorts get identical pattern ids |
| signed shift | `z_k(S) = sign(med_S(y_k) − med(y_k)) · |z_k(S)|` (a zero difference counts as +) | the EDA keeps magnitudes only; direction drives CONTRASTS, the phenomenon vector and the answer |
| EMM per pair | `emm_score = EMM_eda / sqrt(m(m − 1))`, applied as the pass-1 rows become candidates (so `temp_index`, step 4b and the insight all see the per-pair value) | the Frobenius norm grows with the number of off-diagonal entries; per pair it is an RMS correlation change in `[0, 2]`, so `MIN_EMM_SCORE` means the same on every dataset |
| covariance pair | `argmax_{i<j} |C_S,ij − C_ij|` over pairs defined in `S` (needs `n ≥ 3m`; `{}` if none) → `{pair, local_corr, global_corr, delta}` | says *which* relation drives the EMM score |
| pair shifts | `z_k(S)` for pair metrics missing from the top three (`Shift.source = "covariance_pair"`) | lets a correlation-change insight be targeted on one of its metrics |
| significance | two-sided asymptotic median test on the primary metric, `se = 1.2533 · 1.4826 · MAD(y_S) / √n` with the subgroup MAD computed like the EDA's (zero-MAD fallback included); if that is 0, the global MAD; if that is 0 too, `se = sd(y_S) / √n`. `z = (med_S − med) / se`, `p = 2 · Φ̄(|z|)` (`p = 1` for `n < 2` or a zero or non-finite `se`); `p_adjusted = min(1, p · n_tests)` | the EDA has no p-values; selection and weight need one. Bonferroni over overlapping cohorts is conservative, never anti-conservative |
| stability | `final_sd / sd_raw = 1 − min(CV, 0.9)` | recovers the bootstrap factor that step 4b folds into `final_sd` |
| determinism | `np.random.seed(EDA_RANDOM_SEED)` before step 4b | `DataFrame.sample` uses the global RNG; same data and config → identical insights |
| provenance | dataset, file, batch, engine path (`subgroup_miner/vendor/eda/main_upd.py`), steps, exact selector, size of the testing family; the graph service adds `rows_ref` (`datasets/<ds>/covers.npz#<pattern id>`) to the journal record | traceability |

## 2.5 Contracts

* `run_discovery(df, config, on_stage=None) → DiscoveryResult(profile, candidates, validated, data, n_tests, pass1_subgroups, rejections)`: `candidates` are the distinct cohorts, `validated` the ones step 4b returned, `rejections` the `cover_equivalent` / `near_duplicate` merges.
* `Candidate(expression, conditions, row_indices, row_count, volume_utility, top_shifts[(metric, |z|)], sd_aggregate_score, emm_stabilized_score (per pair), temp_index, validated, final_sd_score, drivers, aliases)`.
* `build_insights(result, config, dataset_id=, batch_id=, filename=) → list[Insight]` — unfiltered, weight unset ([3.1](03_insights.md#31-the-insight-record)); the primary target is the largest shift.
* `covers_of(result) → {expression: row positions}` for every distinct cohort; `Engine.process` persists the covers of the kept insights as `covers.npz` (`Workspace.save_covers`), keyed by pattern id, and points each journal record's `provenance.rows_ref` at them (`Workspace.covers_ref`).

## 2.6 Configuration

| Parameter | Default | Effect |
|---|---|---|
| `MIN_ROWS`, `MAX_UPLOAD_MB` | 50, 200 | ingestion limits (`MAX_UPLOAD_MB` is the graph service's upload limit, a `Settings` field) |
| `BIN_COLUMNS`, `CATEGORICAL_COLUMNS` | "", "" | workspace defaults for bands and categorical overrides |
| `COMPUTE_BUDGET` | 5000 | maximum conjunctions in the search space (EDA default) |
| `VALIDATION_BUDGET` | 50 | cohorts sent to the bootstrap (EDA default) |
| `MIN_SEARCH_DIMENSIONS` | 3 | dimension back-fill floor in step 2 |
| `REDUNDANCY_JACCARD` | 0.88 | near-duplicate threshold |
| `EDA_RANDOM_SEED` | 42 | bootstrap seed |

## 2.7 Failure modes and guarantees

| Code | Raised by | Meaning |
|---|---|---|
| `file_too_large` | `PipelineError` (`Engine.upload`) | an upload above `MAX_UPLOAD_MB`; refused with HTTP 413 before a batch is registered |
| `unsupported_file`, `unreadable_file`, `invalid_options`, `invalid_schema`, `no_numeric_targets` | `IngestionError` | missing file or unsupported type, parse error, bad options, too few rows or columns, no numeric column |
| `no_numeric_targets`, `invalid_schema`, `no_candidates` | `DiscoveryError` | profiling left no metric; no categorical dimension (the message suggests `BIN_COLUMNS`); empty search space or nothing passed pass 1 |
| `internal_error` | `DiscoveryError` | a closed intent that does not contain its own selector (an invariant check that should never fire) |

Guarantees (`tests/test_discovery_contract.py`; on the demo unless a toy frame is named):
* `0 < len(candidates) ≤ search space size`, `0 < len(validated) ≤ VALIDATION_BUDGET`, `n_tests = len(candidates) · m`; by construction `len(candidates) ≤ pass1_subgroups`.
* On a toy frame with an implied column: `pass1_subgroups − len(candidates)` equals the `cover_equivalent` rejections, the merged cohort carries the implied condition and no driver for it, and the validated cohorts have pairwise distinct extents; near duplicates are pruned in rank order (a second toy case).
* Galois antitone: among validated cohorts, a strictly smaller extent has a strictly larger intent.
* Same data and config → identical insights; `Insight.support == len(covers[expression])`.
* The planted mechanisms of the demo are recovered with the right signs (the test asserts five of the six; the sixth, delay → returns, is recovered on the demo — APAC∧online and US∧laptops∧retail — and guarded by the benchmark regression of [10.3](10_verification.md#103-hypothesis-benchmark)).

## 2.8 Measured behaviour

| Dataset | Pass-1 subgroups → distinct cohorts | Near duplicates pruned | Validated | Kept insights |
|---|---|---|---|---|
| demo (`retail_synthetic.csv`) | 84 → 84 | 0 | 50 | 28 |
| `housing.csv` with two bands | 102 → 102 | 0 | 50 | 40 |

On the demo and on `housing.csv`, no profiled categorical is constant on another's subgroups (the demo dimensions are drawn independently; housing has one native categorical and two bands), so there is no implied condition and closure finds nothing to merge. The toy frames of [2.7](#27-failure-modes-and-guarantees) exercise both merges: closure with an implied column, and near-duplicate pruning.
