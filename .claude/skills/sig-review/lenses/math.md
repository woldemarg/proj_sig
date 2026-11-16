# Lens: mathematics, statistics and algorithmic correctness

Report: `01-mathematics.md`. Apply the rules of `../SKILL.md`.

Review the mathematical and statistical correctness of the target: are the estimators, tests, time-series operations, spatial statistics, network measures and the algorithms that combine them the ones the chapters define, and are they computed as specified?

`docs/02`–`docs/05` and `docs/07` state the quantities. `docs/11_reference.md` §11.4 is their index. The planted phenomena in `insight_graph_service/core/demo.py` are the reference cases. Do not replace a declared estimator with a textbook default (a mean for a median, a t-test for the median test, Pearson for the correlation the code uses) unless the chapter says that default.

This lens owns the quantity. Place, calendars, which edge is walked, stored bytes, the scenario and the screen are other lenses (`../SKILL.md`, the ownership table). A second computation that returns a different value is this lens; a repeated call with the same result is `system`; unused code is `cleanup`.

## What to check

### 1. Formulas and algorithms

- Do the formula, the order of operations and the sign match the chapter?
- Are normalisation, aggregation, division and scaling the ones specified, and are their units kept?
- Are different quantities mixed in one chain (a p-value used as a weight, a path length used as a similarity, a z used as a probability)?
- Do termination, selection and state updates match the described algorithm, and do the invariants hold on the boundary (empty input, a zero denominator, `NaN`, a tie)?
Passing tests do not prove correctness. A property the chapter states as a limit is not a defect.

### 2. Statistics

The screen and the test answer different questions on purpose (§2.2–§2.4, §3.2–§3.3).

- A shift is a robust z: `0.6745 · Δmedian / MAD` against the **global** scale, capped, with the zero-MAD fallback. The sign is `sign(med_S − med)`. A mean, a plain SD or a per-subgroup scale is a different quantity.
- The median test uses the subgroup's own spread and `√n` (or `√n_eff`), two-sided, Bonferroni over **distinct cohorts × metrics**. It gates. It is not a factor of the insight weight: the cohorts were chosen by the same shift, so the p-value is not a calibrated probability.
- The effect gate compares the shift to the table's spread. The significance gate asks whether the subgroup's median is pinned down. One passing and the other failing is the rule, not a contradiction.
- EMM is the RMS correlation change **per pair** of metrics the scope may explain. An undefined local correlation is not a change to 0. The Frobenius norm of the whole matrix is not that score.
- Bootstrap stability resamples the subgroup (blocks, for spatial data). The coefficient of variation folds into `stability`; it is not a second p-value.
- A confounder needs both the Jensen–Shannon gate and the chi-square gate adjusted by the number of categoricals. The distance alone is inflated in a small subgroup.
- The weight is the weighted geometric mean of the **measured** factors, exponents renormalised, floor applied after. An unmeasured factor is dropped, not replaced by a constant. A strong effect does not buy back an unstable one.
- A family is whatever the chapter counts. The cohort Bonferroni and a per-cell false-discovery cut are different families; one must not be applied where the other is specified.

### 3. Time series

A series is the dynamics of one cohort mined once over the whole table, not a model refit on each period (§2.10). Flash Event Blindness is that design, not a missed test.

- `z(t)` is the cohort's level in the bin minus the **rest of the bin**, in the upload's robust unit, uncapped. A bin too thin, too short or without spread is unmeasured (`None`), not zero and not the cohort against itself.
- A tie-heavy metric is levelled by its bin mean; its admitted shift stays a median's. A count series is a log rate ratio, not that z. A metric the chapter gives no series (a lag, a zoned residual) has none.
- Smoothing is the local-level RTS smoother: process noise `Q`, observation noise the bin's own `se²`. An unmeasured bin is predicted, not observed.
- An event is a local extremum of the smoothed level, topographic prominence, both sides clearing the excess and the sampling noise, Bonferroni over pairs of bins. The first and last measured bins are never events. A monotone series has none. A ripple below the excess is not an event however small its p.
- Trend compares the first measured bin to the last. `spike` needs an event; a dip that recovers is `flat`.
- A carrier is a difference in differences inside the cohort, Bonferroni over the conditions tried. An echo is the same metric, kind and bin on a strictly broader intent. `time_score`, `recency` and `core_age` are the functions §7.2 defines, applied to the interval the `temporal` lens owns.

### 4. Geospatial statistics

- Moran's I of a subgroup is over that subgroup's own cells and the ring-1 links that stay inside it. `ρ = max(0, I)`. `n_eff = n · (1 − ρ) / (1 + ρ)`. Scattered cells have no inside links and are not discounted.
- Under event weights a share's `n_eff` is Kish's `(Σw)² / Σw²`, times the Moran factor, with no floor. Below one effective value the test is undefined (`z` 0).
- The neighbourhood band and Moran's I / LISA read different neighbour sets (an empty neighbour counts as 0 in the band; only occupied neighbours enter I). They are not unified.
- LISA uses the conditional-randomisation moments and a Benjamini–Hochberg cut over cells that have neighbours. A permutation floor, or an uncorrected per-cell p, is not that test.
- A zone baseline **replaces** the metric (a count as `log(1 + n)`). The tests run on the replaced column. `n_eff` discounts variance; it does not remove the bias the context null is for. `p_context` is adjusted by `context_family`.
- A condition does not explain a metric it is computed from (a hot spot does not explain `points`).
- A lift and a share are different ratios. A geo edge's share (of cells) and a base edge's share (of events) do not compare. Compactness is the share of neighbour links that stay inside the extent.

### 5. Network measures

- `SPECIALIZES` is extent containment on the concept lattice: a strictly smaller extent has a strictly larger intent. Overlap, similarity or a shared theme is not that edge.
- `CONTRASTS` is an overlapping scope with an opposite shift. `SIBLING` is another value under the same parent.
- `RELATED_TO` is mutual k nearest neighbours by cosine, above the floor. An anchor with no such neighbour is allowed. The sphere's three dimensions are not those cosines.
- `CO_OCCURS` weight is the capped product of co-membership alignments. An OMP coefficient and a cosine are not one scale. `ACTIVATES` strength is alignment times insight weight. Whether that edge is walked is the `graph` lens.
- Cosine assignment ignores the weight's scale. The EMA pull and OMP extraction see it. A path factor the chapter caps at 1 must stay capped.
- A seed score, a path score and a node rank are the combinations §7.2 and §7.3 define. None of them is a probability unless the chapter says so.

## Expected result

Only problems you can substantiate. For each:

- the quantity (name it: the robust z, the median test, `n_eff`, the event gate, the co-occurrence weight, …);
- the semantics the chapter gives it;
- how the implementation deviates;
- a concrete input (a cohort, a series, a cell set, a small graph) on which the result is wrong;
- the minimal fix;
- a control value, an invariant or a test that tells the right result from the wrong one.

A declared limitation (conservative Bonferroni, the ignored covariance of neighbouring smoothed levels, flash blindness, an anchor with no link) is not a finding. An undescribed contract is an uncertainty, not a confirmed defect.
