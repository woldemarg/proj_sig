# 3. Insights — record, selection and weight

> **In one paragraph.** A validated candidate becomes an `Insight`: a typed, JSON-serialisable record that names its subgroup, its signed shifts, its scores and its provenance, and that every later stage reads (the graph calls it a `Pattern`). A fixed, ordered rule set decides which insights become persistent knowledge, and an `insight_weight` in `[0.05, 1]` summarises how strong the evidence is. The weight never changes *which* existing anchor an insight is assigned to; it changes how hard the insight pulls on that anchor, how much it shapes the extraction of new anchors, and how high it ranks in retrieval.

**Code** `ltir/models.py` (`Insight`, `Shift`, `Condition`, `Rejection`), `ltir/quality.py` · **Tests** `tests/test_quality.py` · **Previous** [2. Discovery](02_discovery.md) · **Next** [4. Representation](04_representation.md)

---

## 3.1 The insight record

`Insight.to_record()` / `Insight.from_record()` give the JSON form that is written once to `journal/patterns.jsonl`, carried unchanged as the snapshot's Pattern node properties, and mirrored to Neo4j (nested values as `<key>_json` strings, `None` values left out; [6.6](06_graph_and_storage.md#66-neo4j-mirror)). Every module reads insights through `Insight.from_record`, which ignores fields it does not know, so older records stay readable.

| Field | Content |
|---|---|
| `id` | `P-` + `sha1(json([dataset_id, condition expressions in (attribute, value) order]))[:12]` — deterministic, so a re-run produces the same ids |
| `dataset_id`, `batch_id` | provenance keys |
| `conditions` | the closed intent ([2.3](02_discovery.md#23-deduplication-before-validation)) as `[{"attribute", "value"}, …]` sorted by attribute then value; values are strings even for number-coded columns |
| `expression` | the EDA selector of the cohort (the merged selector with most conditions), verbatim, e.g. `category=='phones' AND region=='US'` |
| `target` | raw column name of the primary metric: the largest shift; for a covariance insight, the pair metric with the larger shift |
| `shifts` | `[{"metric", "robust_z" (signed), "local_median", "global_median", "global_mad", "source": "eda" | "covariance_pair"}]`, sorted by `|robust_z|` descending |
| `support`, `support_fraction` | covered rows, and their share of the table |
| `baseline`, `local`, `effect_size` | global median, subgroup median and signed robust z of the target |
| `sd_score`, `sd_raw_score` | EDA bootstrap-penalised `final_sd_score`, pass-1 `sd_aggregate_score` |
| `emm_score`, `volume_utility` | per-pair RMS correlation change, EDA volume utility |
| `stability` | `sd_score / sd_raw_score` ∈ `[0.1, 1]` (0 when `sd_raw_score` is 0); `None` for a covariance insight |
| `p_value`, `p_adjusted` | adapter median test on the EDA's primary metric, Bonferroni-adjusted; `None` for a covariance insight |
| `drivers` | EDA confounder strings |
| `row_hash` | 16 hex of `sha1` over the sorted covered row positions — the identity of the row set |
| `phenomenon_type` | `"shift"` or `"covariance"` |
| `covariance` | `{"pair": [a, b], "local_corr", "global_corr", "delta"}` or `{}` |
| `aliases` | selectors merged into this cohort before validation |
| `weight`, `weight_factors` | `insight_weight`, and the five factors with `null` for unmeasured ones |
| `provenance` | `{dataset_id, filename, batch_id, engine, steps, expression, rows_ref, multiple_testing_family}` |

The journal adds `row_id` (the vector's row), `canonical` (the canonical form of [4.2](04_representation.md#42-the-canonical-form), including its document) and `embedding` (`{fingerprint, model_id, dim, representation_version}`). What the record does **not** hold: the covered rows themselves (only their positions, in `covers.npz`), the profiled DataFrame, any raw cell value other than condition values and medians — the rows stay in the dataset's source copy (and, for web uploads, in `uploads/`; [6.3](06_graph_and_storage.md#63-the-workspace-on-disk)). The vectors are stored beside it: the insight vector in `embeddings.mmap`, its three blocks and the document embedding in `journal/blocks/<batch>.npz`.

> **Running example** (abridged record of `P-bc4657a04746`):
> ```text
> conditions  category=phones, region=US            expression  category=='phones' AND region=='US'
> target      discount                               support     438 (8.76 %)
> shifts      discount +2.208 (19.19 vs 10.74), margin −1.096 (14.75 vs 19.45),
>             delivery_days −0.253, return_rate −0.227 (source covariance_pair)
> sd_score    3.465    sd_raw_score 3.557    stability 0.974    emm_score 0.0223    volume_utility 0.270
> p_adjusted  < 1e-300 (reported as 0)       multiple_testing_family 336 (= 84 cohorts × 4 metrics)
> covariance  delivery_days ~ return_rate: 0.88 overall → 0.73 in the subgroup
> weight      0.843
> ```

## 3.2 Selection rules

`select_insights(insights, config)` applies a retype step and then rules in a fixed order; every validated candidate is either kept or rejected with exactly one reason. Rejections from discovery and selection are persisted together in `datasets/<id>/rejections.json` and counted in the batch metrics (`pruned`).

```text
significant = |effect_size| ≥ MIN_EFFECT_Z (0.5)  ∧  p_adjusted ≤ MAX_P_ADJUSTED (0.05)
stable      = stability ≥ MIN_STABILITY (0.5)
shift_ok    = significant ∧ stable
emm_ok      = emm_score ≥ MIN_EMM_SCORE (0.08)  ∧  a covariance pair exists
```

| Step | Passes when | Rejection reason |
|---|---|---|
| retype | if `¬shift_ok ∧ emm_ok`, the insight becomes `covariance` and is retargeted on the pair metric with the larger shift (ties by metric name); its target, effect size and medians follow the new target, and `p_value`, `p_adjusted` and `stability` become `None` — they tested the median shift that failed, not the correlation change | — |
| R1 support | `support ≥ MIN_SUPPORT_ROWS` (30), on top of the EDA's own size floor | `min_support` |
| R2/R3 strength and stability | `shift_ok ∨ emm_ok` | `unstable` (significant but not stable), `not_significant` (`|z|` large enough, `p` fails), `weak_effect` (otherwise) |
| R4 weight | `weight ≥ MIN_INSIGHT_WEIGHT` (0.2) | `low_weight` |
| R5 identical extent, R6 near duplicate | run in discovery, before validation ([2.3](02_discovery.md#23-deduplication-before-validation)) | `cover_equivalent`, `near_duplicate` |
| R7 budget | the top `MAX_INSIGHTS_PER_BATCH` (200) by weight (ties by expression) | `budget` |

If nothing is kept, the batch fails with `no_viable_insights`: the journal and the ontology are untouched (the source copy, `profile.json` and `rejections.json` of the dataset folder are already written).

**How the effect gate and the significance gate combine.** They measure different things on purpose. `MIN_EFFECT_Z` asks whether the shift is large against the spread of the whole table (practical relevance, global MAD). The median test asks whether the subgroup's median is pinned down well enough, given the subgroup's own spread and size, after correcting for every cohort and metric the screen looked at — the validated cohorts were chosen by those very shifts, so a family of only the validated primaries would understate the multiplicity, and a standard error on the global spread would overstate the precision of a dispersed subgroup's median. With the subgroup as spread as the table, the test needs

```text
|z| ≥ 0.6745 · z_crit · 1.2533 · 1.4826 / √n          z_crit = Φ̄⁻¹(MAX_P_ADJUSTED / (2 · n_tests))
```

On the demo (`n_tests` 336, `z_crit` 3.79) that is 0.87 sd at 30 rows and 0.5 sd at about 90 rows; above that the effect gate decides (housing: 612 tests, 98 rows). On both datasets every rejection is `weak_effect`: the significance gate removed nothing the effect gate would have kept. A dispersed subgroup needs a proportionally larger shift, because its median is genuinely less certain, and is rejected as `not_significant`.

A covariance insight cites only its correlation change. Its median shifts stay in the record as measurements, but they are neither components nor phrases nor prompt lines ([4.2](04_representation.md#42-the-canonical-form)), and its validation is reported as the correlation change with no median test ([7.4](07_question_answering.md#74-the-evidence-object)).

## 3.3 Insight weight

The weight combines five normalised factors, each in `[0, 1]` and monotone in its evidence:

| Factor | Shift insight | Covariance insight |
|---|---|---|
| effect | `1 − exp(−|effect_size| / WEIGHT_EFFECT_REF)` (1.5) | `1 − exp(−emm_score / WEIGHT_EMM_REF)` (0.08) |
| stability | `clip(stability, 0, 1)` | not measured — `None` |
| confidence | `clip(−log10(max(p_adjusted, 1e-300)) / WEIGHT_CONFIDENCE_REF, 0, 1)` (6) | not measured — `None` |
| support | `clip(volume_utility / 0.3849, 0, 1)` — 0.3849 is the maximum of `√p(1 − p)` | same |
| emm | `0.5 + 0.5 · (1 − exp(−emm_score / WEIGHT_EMM_REF))` — a bonus that never zeroes the weight | same |

They are combined as a weighted geometric mean over the **measured** factors, with exponents `a = WEIGHT_EXPONENTS = (0.35, 0.25, 0.20, 0.10, 0.10)` for (effect, stability, confidence, support, emm):

```text
w = WEIGHT_FLOOR + (1 − WEIGHT_FLOOR) · exp( Σ_{k measured} (a_k / Σ_{measured} a) · ln max(f_k, 1e-3) )        WEIGHT_FLOOR = 0.05
```

A geometric mean makes the factors complementary: a strong effect cannot buy back an unstable one. Unmeasured factors are dropped and the exponents renormalised, because in a product any substituted constant is a fixed tax rather than a neutral value (a 0.5 for the two factors a covariance insight lacks would cost every such insight `0.5^0.45 = 0.73`).

> **Running example.** Factors for `P-bc4657a04746`: effect 0.7705, stability 0.9744, confidence 1.0, support 0.7016, emm 0.6214, so `w = 0.05 + 0.95 · exp(−0.1808) = 0.843`.

## 3.4 Where the weight acts

| Consumer | Use | Chapter |
|---|---|---|
| latent ontology | an insight enters as `x = w · x̂`. Cosine assignment ignores the scale, so assignment to an existing anchor does not depend on `w`; the EMA pull on a centroid does, and so does OMP extraction, which runs on the `w`-scaled orphans (reconstruction loss ∝ `w²`) and therefore decides which new anchors appear | [5.5](05_latent_anchors.md#55-how-the-evidence-weight-acts) |
| ACTIVATES edges | `strength = alignment · w`; an anchor's `evidence_mass = Σ strength` | [5.9](05_latent_anchors.md#59-activation-records-and-batch-metrics) |
| retrieval | seed score `+ 0.10 · w` (`+ 0.30 · w` for questions without a recognised metric or condition); node rank `= path score · w` | [7.2](07_question_answering.md#72-seeds), [7.3](07_question_answering.md#73-transversal-traversal) |
| selection | rule R4 | [3.2](#32-selection-rules) |

## 3.5 Configuration

| Parameter | Default | Note |
|---|---|---|
| `MIN_SUPPORT_ROWS` | 30 | |
| `MIN_EFFECT_Z`, `MAX_P_ADJUSTED`, `MIN_STABILITY` | 0.5, 0.05, 0.5 | the shift test |
| `MIN_EMM_SCORE` | 0.08 | per-pair RMS scale after shrinkage: ≈ 0.2–0.4 raw change per pair for subgroups with 5–15 % of the rows; the planted EU∧phones correlation break scores 0.086. Part of the fingerprint ([4.6](04_representation.md#46-representation-identity-and-versions)): it decides whether a correlation component enters the vector |
| `MIN_INSIGHT_WEIGHT`, `MAX_INSIGHTS_PER_BATCH` | 0.2, 200 | |
| `WEIGHT_EFFECT_REF`, `WEIGHT_CONFIDENCE_REF`, `WEIGHT_EMM_REF` | 1.5, 6, 0.08 | saturation scales; `WEIGHT_EMM_REF` also scales the correlation component's coefficient, so it is part of the fingerprint |
| `WEIGHT_EXPONENTS`, `WEIGHT_FLOOR` | (0.35, 0.25, 0.20, 0.10, 0.10), 0.05 | |

## 3.6 Guarantees

* `WEIGHT_FLOOR ≤ weight ≤ 1`; the weight is monotone non-decreasing in `|z|`, stability, `−log p` and the EMM score; unmeasured factors are `None` and never enter the mean.
* No two kept insights share a `row_hash` or an `id`.
* Every validated candidate is kept or has exactly one rejection; cohorts beyond `VALIDATION_BUDGET` never reach selection.

## 3.7 Measured behaviour

On the demo, 50 validated candidates give 28 insights; the 22 rejections are all `weak_effect`. Two of the 28 are covariance insights: the planted discount–margin correlation break in EU∧phones (EMM 0.086) and a mixture effect in tablets∧online, where discount and delivery days co-move because the APAC rows carry both planted shifts (EMM 0.14).
