import pandas as pd
import numpy as np
import pysubgroup as ps
import itertools
from scipy.stats import zscore
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%

# Guarded so the module can be imported as a library (sig/ltir/discovery.py)
# without loading a dataset; Spyder cells still run it (__name__ == "__main__").
if __name__ == "__main__":
    # url = "../data/housing.csv"
    url = "../data/demo_points.csv"
    df = pd.read_csv(url)

# %%

# ==========================================
# ROBUST STATISTICAL PRIMITIVES
# ==========================================


# Normal-consistency constants: MAD = 0.6745 sigma, mean absolute deviation = 0.7979 sigma.
_MAD_FROM_MEAN_ABS_DEV = 0.6745 / 0.7979
# Bounded influence: beyond this many robust sigmas the exact magnitude no longer
# matters for ranking, and a degenerate scale cannot dominate the aggregate score.
ROBUST_Z_CAP = 10.0


def calculate_mad(series):
    """Median Absolute Deviation as the robust scale of a metric.

    When more than half of the values tie (zero-inflated or heavily discretised
    metrics) the MAD is 0 and every subgroup would look infinitely shifted; the
    scale then falls back to the normal-consistent mean absolute deviation
    (0 only for a truly constant column)."""
    values = np.asarray(series, dtype=float)
    if values.size == 0:
        return 0.0
    median = np.median(values)
    abs_dev = np.abs(values - median)
    mad = np.median(abs_dev)
    if mad == 0:
        mad = _MAD_FROM_MEAN_ABS_DEV * np.mean(abs_dev)
    return float(mad)


def robust_z_score(local_median, global_median, global_mad, epsilon=1e-6):
    """Scale-invariant robust effect size |median_local - median_global| / sigma_hat with
    sigma_hat = MAD / 0.6745 (normal-consistent), capped at ROBUST_Z_CAP."""
    return min(0.6745 * abs(local_median - global_median) / (global_mad + epsilon), ROBUST_Z_CAP)


def volume_preference_score(p):
    """sqrt(p) * (1 - p): favours moderate, actionable groups (peaks at p = 1/3, 0 at p = 0 and 1)."""
    return np.sqrt(p) * (1 - p)


def extract_dimension_attrs(selector):
    """Recursively extracts attribute names from pysubgroup selectors."""
    if hasattr(selector, "attribute_name"):
        return {selector.attribute_name}
    if hasattr(selector, "selectors"):
        return set().union(
            *[extract_dimension_attrs(s) for s in selector.selectors]
        )
    if hasattr(selector, "_selectors"):
        return set().union(
            *[extract_dimension_attrs(s) for s in selector._selectors]
        )
    return set()


def subgroup_from_indices(data, row_indices):
    """Resolve a subgroup slice from compact row index storage."""
    return data.iloc[row_indices]


# ==========================================
# STEP 1: Ingestion & Semantic Profiling
# ==========================================
def step1_profile_data(data):
    data_safe = data.copy()

    # Continuous float measures are naturally all-unique; only non-float
    # columns with row-count parity are treated as identifiers.
    identifiers = [
        col
        for col in data_safe.columns
        if data_safe[col].nunique() == data_safe[col].dropna().shape[0]
        and not pd.api.types.is_float_dtype(data_safe[col])
    ]
    data_safe = data_safe.drop(columns=identifiers)

    numerics = data_safe.select_dtypes(include=[np.number]).columns.tolist()
    categoricals = data_safe.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    for col in categoricals:
        data_safe[col] = data_safe[col].astype(str)

    # A single-valued column is "redundant" with every other column under the
    # pair rule below, which would then drop the *informative* one; remove it first.
    constant_cats = [c for c in categoricals if data_safe[c].nunique() <= 1]
    categoricals = [c for c in categoricals if c not in constant_cats]
    data_safe = data_safe.drop(columns=constant_cats)

    # Nested columns (city -> country): the finer one is redundant for the subgroup search
    # because its subgroups are specialisations of the coarser one's. The rule only fires
    # when the coarser column is informative: a near-constant column (top level >= 95 %)
    # is trivially "a function of" any other column and must not drop it.
    top_share = {c: float(data_safe[c].value_counts(normalize=True).iloc[0]) for c in categoricals}
    cats_to_drop = set()
    for i in range(len(categoricals)):
        for j in range(i + 1, len(categoricals)):
            col1, col2 = categoricals[i], categoricals[j]
            if col1 in cats_to_drop or col2 in cats_to_drop:
                continue

            unique_pairs = data_safe[[col1, col2]].drop_duplicates().shape[0]
            u1, u2 = data_safe[col1].nunique(), data_safe[col2].nunique()
            finer, coarser = (col1, col2) if u1 > u2 else (col2, col1)
            if unique_pairs <= 1.10 * max(u1, u2) and top_share[coarser] < 0.95:
                cats_to_drop.add(finer)

    categoricals = [c for c in categoricals if c not in cats_to_drop]

    if numerics:
        variances = data_safe[numerics].var()
        valid_nums = variances[variances > 0].index.tolist()

        init_corr = data_safe[valid_nums].corr().abs()
        upper_tri = init_corr.where(
            np.triu(np.ones(init_corr.shape), k=1).astype(bool)
        )
        nums_to_drop = [
            col for col in upper_tri.columns if any(upper_tri[col] > 0.95)
        ]
        valid_nums = [n for n in valid_nums if n not in nums_to_drop]
        global_corr = data_safe[valid_nums].corr().fillna(0).values
    else:
        valid_nums, global_corr = [], np.array([])

    return {
        "data_safe": data_safe,
        "numerics": valid_nums,
        "categoricals": categoricals,
        "global_corr": global_corr,
        "total_rows": len(data_safe),
    }


# ==========================================
# STEP 2: Macro Grouping
# ==========================================
def step2_evaluate_macro_groupings(profile, min_categories=0):
    """min_categories > 0 backfills the strongest sub-median categories so the
    conjunction search space (2-/3-conjunctions) is not empty on narrow schemas."""
    data = profile["data_safe"]
    numerics = profile["numerics"]
    categoricals = profile["categoricals"]
    macro_registry = []
    dynamic_cardinality_cap = np.sqrt(profile["total_rows"])

    for cat in categoricals:
        if data[cat].nunique() > dynamic_cardinality_cap:
            continue

        for num in numerics:
            valid = data[[cat, num]].dropna()
            n = len(valid)
            group_means = valid.groupby(cat)[num].mean()
            group_sizes = valid.groupby(cat).size()
            k = len(group_sizes)
            if n <= k or k < 2:
                continue
            global_mean = valid[num].mean()
            ss_between = float(np.sum(group_sizes * (group_means - global_mean) ** 2))
            ss_total = float(np.sum((valid[num] - global_mean) ** 2))
            if ss_total <= 0:
                continue
            # eta^2 = SS_between / SS_total is biased upwards by (k - 1) / (n - 1) under H0,
            # which favours high-cardinality columns; epsilon^2 subtracts that expectation.
            ms_within = (ss_total - ss_between) / (n - k)
            power = max(ss_between - (k - 1) * ms_within, 0.0) / ss_total
            macro_registry.append({"category": cat, "target": num, "power": power})

    df_macro = pd.DataFrame(macro_registry)
    if df_macro.empty:
        return []

    category_power = (
        df_macro.groupby("category")["power"]
        .mean()
        .sort_values(ascending=False)
    )
    dynamic_power_threshold = category_power.median()
    selected = category_power[
        category_power > dynamic_power_threshold
    ].index.tolist()[:6]
    if len(selected) < min_categories:
        selected = category_power.index.tolist()[:min_categories]
    return selected


# ==========================================
# STEP 3: Dynamic Action Space
# ==========================================
def step3_generate_search_space(profile, top_categories, compute_budget=5000):
    data = profile["data_safe"]
    base_selectors = []
    level_share = {}

    for col in top_categories:
        val_frequencies = (
            data[col].value_counts(normalize=True).sort_values(ascending=False)
        )
        # keep every level up to and including the one that crosses 95 % cumulative mass;
        # only the long tail beyond it is dropped (a 60/40 column keeps both levels)
        cumulative_mass = val_frequencies.cumsum()
        mass_before = cumulative_mass.shift(1, fill_value=0.0)
        valid_uniques = cumulative_mass[mass_before < 0.95].index.tolist()

        for val in valid_uniques:
            if str(val) not in ["nan", "<NA>", "None"]:
                base_selectors.append(ps.EqualitySelector(col, val))
                level_share[(col, val)] = float(val_frequencies[val])

    def expected_share(conj):
        # independence approximation of the subgroup size: pairs with a tiny expected
        # support fail the pass-1 size screen anyway, so they are the first to give way
        # when the search space exceeds the budget
        sels = getattr(conj, "selectors", None) or getattr(conj, "_selectors", None) or []
        return float(np.prod([level_share[(s.attribute_name, s.attribute_value)] for s in sels]))

    search_vectors = []
    combo_2d = [
        ps.Conjunction([s1, s2])
        for s1, s2 in itertools.combinations(base_selectors, 2)
        if s1.attribute_name != s2.attribute_name
    ]

    if len(combo_2d) <= compute_budget:
        search_vectors += combo_2d
        combo_3d = [
            ps.Conjunction([s1, s2, s3])
            for s1, s2, s3 in itertools.combinations(base_selectors, 3)
            if len({s1.attribute_name, s2.attribute_name, s3.attribute_name})
            == 3
        ]
        if len(search_vectors) + len(combo_3d) <= compute_budget:
            search_vectors += combo_3d
    else:
        combo_2d.sort(key=expected_share, reverse=True)
        search_vectors += combo_2d[:compute_budget]

    return search_vectors


# ==========================================
# STEP 4: PASS 1 - Robust Screening
# ==========================================
def step4_evaluate_micro_slices(profile, search_vectors):
    data = profile["data_safe"]
    numerics = profile["numerics"]
    global_corr = profile["global_corr"]
    total_rows = profile["total_rows"]

    global_medians = {num: np.median(data[num].dropna()) for num in numerics}
    global_mads = {num: calculate_mad(data[num].dropna()) for num in numerics}

    min_sg_size = max(len(numerics) * 5, int(total_rows * 0.005))
    insights = []

    for selector in search_vectors:
        cover_mask = selector.covers(data)
        row_indices = np.flatnonzero(cover_mask)
        sg_size = len(row_indices)

        if sg_size < min_sg_size or sg_size == total_rows:
            continue

        sg_data = subgroup_from_indices(data, row_indices)
        volume_utility = volume_preference_score(sg_size / total_rows)

        shift_profile = {}
        for num_col in numerics:
            s_median = np.median(sg_data[num_col].dropna())
            if not np.isnan(s_median):
                shift_profile[num_col] = robust_z_score(
                    s_median, global_medians[num_col], global_mads[num_col]
                )

        top_k_shifts = sorted(
            shift_profile.items(), key=lambda x: x[1], reverse=True
        )[:3]
        sd_aggregate_score = sum(shift for _, shift in top_k_shifts)

        min_rows_for_emm = len(numerics) * 3
        if sg_size < min_rows_for_emm or len(numerics) < 2:
            emm_stabilized_score = 0.0
        else:
            local_corr = sg_data[numerics].corr().values
            corr_delta = global_corr - local_corr
            corr_delta[np.isnan(corr_delta)] = 0.0  # undefined locally: no evidence of change
            raw_divergence = np.linalg.norm(corr_delta)
            reliability_penalty = np.sqrt(
                max(0, sg_size - min_sg_size) / (total_rows - min_sg_size)
            )
            emm_stabilized_score = raw_divergence * reliability_penalty

        insights.append(
            {
                "dimensions": str(selector),
                "dimension_attrs": extract_dimension_attrs(selector),
                "row_indices": row_indices,
                "row_count": sg_size,
                "volume_utility": volume_utility,
                "top_shifts": top_k_shifts,
                "sd_aggregate_score": sd_aggregate_score,
                "emm_stabilized_score": emm_stabilized_score,
            }
        )

    return pd.DataFrame(insights)


# ==========================================
# STEP 4B: PASS 2 - Deep Validation
# ==========================================
BOOTSTRAP_RESAMPLES = 20


def _skew_p_value(local_counts, global_counts):
    """Chi-square test of independence between subgroup membership and a category
    (subgroup vs. rest of the data); 1.0 when the table is degenerate."""
    from scipy.stats import chi2_contingency

    local, glob = local_counts.align(global_counts, fill_value=0)
    table = np.vstack([local.values, (glob - local).values]).astype(float)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
        return 1.0
    return float(chi2_contingency(table)[1])


def step4b_deep_validation(
    data, top_candidates, numerics, categoricals, global_medians, global_mads
):
    validated_insights = []
    global_cat_dists = {
        cat: data[cat].value_counts(normalize=True) for cat in categoricals
    }
    global_cat_counts = {cat: data[cat].value_counts() for cat in categoricals}
    # Bonferroni over the categorical columns screened per subgroup
    skew_alpha = 0.01 / max(len(categoricals), 1)

    for _, row in top_candidates.iterrows():
        sg_data = subgroup_from_indices(data, row["row_indices"])

        bootstrap_scores = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sample = sg_data.sample(frac=1.0, replace=True)
            b_score = 0
            for num_col, _ in row["top_shifts"]:
                s_median = np.median(sample[num_col].dropna())
                b_score += robust_z_score(
                    s_median, global_medians[num_col], global_mads[num_col]
                )
            bootstrap_scores.append(b_score)

        stability_penalty = np.std(bootstrap_scores) / (
            np.mean(bootstrap_scores) + 1e-6
        )
        final_sd_score = row["sd_aggregate_score"] * (
            1 - min(stability_penalty, 0.9)
        )

        drivers = []

        for cat in categoricals:
            if cat in row["dimension_attrs"]:
                continue

            local_dist = sg_data[cat].value_counts(normalize=True)
            aligned_local, aligned_global = local_dist.align(
                global_cat_dists[cat], fill_value=0
            )
            js_div = jensenshannon(aligned_local.values, aligned_global.values)

            # JS distance is inflated by sampling noise in small subgroups, so the skew
            # must also be significant (chi-square, Bonferroni over the columns screened)
            if js_div > 0.15 and _skew_p_value(
                sg_data[cat].value_counts(), global_cat_counts[cat]
            ) < skew_alpha:
                top_local_val = (aligned_local - aligned_global).idxmax()
                drivers.append(
                    f"[{cat}] heavily skewed to '{top_local_val}' (JS: {
                        js_div:.2f})"
                )

        top_shift_cols = [col for col, _ in row["top_shifts"]]
        for num in numerics:
            if num not in top_shift_cols:
                s_med = np.median(sg_data[num].dropna())
                shift = robust_z_score(
                    s_med, global_medians[num], global_mads[num]
                )
                if shift > 1.5:
                    sign = "+" if s_med > global_medians[num] else "-"
                    drivers.append(f"[{num}] hidden shift ({sign}{shift:.1f} robust sigma)")

        validated_insights.append(
            {
                "dimensions": row["dimensions"],
                "row_count": row["row_count"],
                "volume_utility": row["volume_utility"],
                "top_shifts": row["top_shifts"],
                "final_sd_score": final_sd_score,
                "emm_stabilized_score": row["emm_stabilized_score"],
                "root_cause_drivers": drivers[:3],
            }
        )

    return pd.DataFrame(validated_insights)


# ==========================================
# STEP 5: Narrative Materialization
# ==========================================
def step5_integrate_and_materialize(validated_df):
    if validated_df.empty:
        return print("No insights survived validation.")

    validated_df["z_sd"] = pd.Series(
        zscore(validated_df["final_sd_score"]), index=validated_df.index
    )
    validated_df["z_emm"] = pd.Series(
        zscore(validated_df["emm_stabilized_score"]), index=validated_df.index
    )
    validated_df["z_volume"] = pd.Series(
        zscore(validated_df["volume_utility"]), index=validated_df.index
    )

    validated_df["z_sd"] = validated_df["z_sd"].fillna(0)
    validated_df["z_emm"] = validated_df["z_emm"].fillna(0)
    validated_df["z_volume"] = validated_df["z_volume"].fillna(0)

    validated_df["integrated_index"] = (
        validated_df["z_sd"].clip(lower=0)
        + validated_df["z_emm"].clip(lower=0)
        + validated_df["z_volume"].clip(lower=0)
    )

    top_results = validated_df.sort_values(
        "integrated_index", ascending=False
    ).head(15)

    print("\n" + "=" * 80)
    print("ENTERPRISE DISCOVERY ENGINE: VALIDATED INSIGHTS")
    print("=" * 80)

    for idx, row in top_results.reset_index().iterrows():
        print(f"\n[{idx + 1}] DISCOVERY: {row['dimensions']}")
        print(f"    - COHORT SIZE: {row['row_count']} rows.")

        print("    - PRIMARY SHIFTS (MAD):")
        for metric, score in row["top_shifts"]:
            print(f"        * {metric}: {score:.2f} MAD divergence")

        print(
            f"    - CORRELATION STABILITY: Matrix Divergence (Z: {row['z_emm']:.2f})"
        )
        print(f"    - INDEX SCORE: {row['integrated_index']:.2f}")

        if row["root_cause_drivers"]:
            print("    - ROOT CAUSE DRIVERS:")
            for driver in row["root_cause_drivers"]:
                print(f"        > {driver}")
        else:
            print(
                "    - ROOT CAUSE DRIVERS: No secondary hidden drivers detected."
            )


# ==========================================
# EXECUTION WORKFLOW
# ==========================================
if __name__ == "__main__":
    profile = step1_profile_data(df)
    top_cats = step2_evaluate_macro_groupings(profile)
    space = step3_generate_search_space(profile, top_cats, compute_budget=5000)

    raw_micro_insights = step4_evaluate_micro_slices(profile, space)

    if not raw_micro_insights.empty:
        raw_micro_insights["temp_index"] = (
            pd.Series(zscore(raw_micro_insights["sd_aggregate_score"]))
            .fillna(0)
            .clip(lower=0)
            + pd.Series(zscore(raw_micro_insights["emm_stabilized_score"]))
            .fillna(0)
            .clip(lower=0)
            + pd.Series(zscore(raw_micro_insights["volume_utility"]))
            .fillna(0)
            .clip(lower=0)
        )
        top_50_candidates = raw_micro_insights.sort_values(
            "temp_index", ascending=False
        ).head(50)

        global_medians = {
            num: np.median(profile["data_safe"][num].dropna())
            for num in profile["numerics"]
        }
        global_mads = {
            num: calculate_mad(profile["data_safe"][num].dropna())
            for num in profile["numerics"]
        }

        validated_insights = step4b_deep_validation(
            profile["data_safe"],
            top_50_candidates,
            profile["numerics"],
            profile["categoricals"],
            global_medians,
            global_mads,
        )

        step5_integrate_and_materialize(validated_insights)
    else:
        print("No valid slices generated in Pass 1.")
