"""subgroup_miner on its own: a table in, validated insights and LLM context out (subgroup_miner/README.md)."""

from __future__ import annotations


def test_a_table_becomes_insights_and_llm_context(demo_csv):
    from subgroup_miner import MinerConfig, build_insights, covers_of, describe, load_dataset, recompute_statistics, run_discovery, select_insights
    from subgroup_miner.discovery import selector_conditions

    config = MinerConfig()
    table = load_dataset(demo_csv, config)
    found = run_discovery(table.frame, config)
    insights = select_insights(build_insights(found, config, dataset_id=table.dataset_id, batch_id="b1", filename=demo_csv.name), config).kept
    assert insights and all(0 < i.weight <= 1 for i in insights)
    context = describe(insights, limit=5)
    lines = context.splitlines()
    assert len(lines) == 5 and lines[0].startswith("1. ") and all(" sd " in line or "correlation" in line for line in lines)
    assert context.isascii()
    lead = next(i for i in insights if i.p_value is not None)  # recomputed from the table and its rows alone
    conditions = [(c.attribute, c.value) for c in lead.conditions]
    again = recompute_statistics(table.frame, conditions, lead.expression, [s.metric for s in lead.shifts], None, found.n_tests)
    assert list(again["rows"]) == sorted(covers_of(found)[lead.expression]) and list(again["expression_rows"]) == list(again["rows"])
    assert again["support"] == lead.support and again["p_value"] == lead.p_value and again["p_adjusted"] == lead.p_adjusted
    assert selector_conditions("kind=='Bed AND Breakfast' AND city=='x'") == [("kind", "Bed AND Breakfast"), ("city", "x")]
    assert [round(s["robust_z"], 9) for s in again["shifts"]] == [round(s.robust_z, 9) for s in lead.shifts]


def test_points_per_cell_and_level_are_counted_once_per_distinct_place():
    """``_level_counts`` is the crosstab of cells and levels, columns by overall frequency and a missing value in none;
    ``distinct_points`` indexes a place many events share once, and gives every point back its own."""
    import numpy as np
    import pandas as pd

    from subgroup_miner.spatial import _level_counts, distinct_points

    rng = np.random.default_rng(0)
    text = pd.Series(rng.choice(["a", "b", "c", None], 5000, p=[0.5, 0.3, 0.15, 0.05]), dtype="string")
    codes = rng.integers(0, 40, 5000)
    order = text.value_counts().index
    got = _level_counts((order.get_indexer(text), order), codes, 45)  # a profiled column: its codes by frequency
    want = pd.crosstab(codes[text.notna().to_numpy()], text.dropna().to_numpy()).reindex(
        index=range(45), columns=text.value_counts().index, fill_value=0
    )
    assert list(got.columns) == list(want.columns) and got.index.equals(pd.RangeIndex(45)) and (got.to_numpy() == want.to_numpy()).all()
    lat, lon = np.repeat([34.05, 37.77, 34.05], 4), np.repeat([-118.25, -122.42, -118.25], 4)
    pairs, back = distinct_points(lat, lon)
    assert len(pairs) == 2 and np.array_equal(pairs[back], np.column_stack([lat, lon]))
