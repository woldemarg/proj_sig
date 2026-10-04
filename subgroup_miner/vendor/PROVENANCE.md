# Provenance of the vendored discovery engine

`eda/main_upd.py` is **copied into this repository** so the miner runs without the original eda project. The copy has diverged from its original, and every difference is listed below. The SHA-256 prefix is of the original file at copy time (2026-09-30).

| Vendored file | Original | sha256[:16] of original | Changes in the copy |
|---|---|---|---|
| `eda/main_upd.py` | eda project: `scripts/main_upd.py` | `058133ef64d7787c` | the 3 integration edits (made before vendoring), the numerical repairs below, and the removal of the standalone runner and step 5 (edit 10) |

## Integration edits (before vendoring)
1. The data load and workflow run under `if __name__ == "__main__":`, so the module can be imported. Edit 10 later removed both blocks.
2. `step1_profile_data`: float columns are never treated as identifiers, because continuous targets are all-unique.
3. `step2_evaluate_macro_groupings(profile, min_categories=0)`: an optional dimension floor. The default keeps the original behaviour.

## Numerical repairs (math audit; each is a local change, explained in docs/02_discovery.md §2.2)
4. `step1`: constant categoricals are dropped before the overlap rule. The overlap rule drops the finer column only when the coarser one is informative (top level < 95 %).
5. `calculate_mad`: a normal-consistent mean-absolute-deviation fallback when the MAD is 0; `robust_z_score` is capped at 10.
6. `step2`: ε² (bias-corrected) instead of η² as the grouping power.
7. `step3`: the level crossing 95 % cumulative mass is kept. An over-budget space keeps the 2-conjunctions with the largest expected support instead of returning nothing.
8. `step4`: correlations undefined inside a subgroup contribute no EMM change; they were read as 0.
9. `step4b`: 20 full-size bootstrap resamples instead of 5 × 90 %. Confounder drivers need a significant chi-square skew and name the level with the largest share gain. Hidden shifts are reported signed in robust σ.

## Removals
10. These parts are removed: the standalone runner (the guarded load of a data file from the original eda project at the top, and the workflow at the bottom); `step5_integrate_and_materialize`, a print-only report; and the then-unused `zscore` import. The miner never called them. `subgroup_miner/discovery.py` computes the runner's `temp_index` with the same formula (`_z_positive`: z-score, fill 0, clip at 0). Step 5's integrated index is not computed, because nothing read it.

## Other copied assets
| Asset | Original | Location |
|---|---|---|
| `housing.csv` (a realistic demo dataset) | eda project: `data/housing.csv` | `data/housing.csv` (local, git-ignored) |

## Taking upstream changes
Do not re-copy the file over its local changes. Re-apply an upstream change by hand, keeping edits 1–10 (upstream changes to step 5 or the runner do not apply), then run `python -m pytest`. `subgroup_miner/discovery.py` stays the boundary where vendored results become `Insight` records.
