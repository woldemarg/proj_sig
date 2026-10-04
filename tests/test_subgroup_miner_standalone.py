"""subgroup_miner on its own: a table in, validated insights and LLM context out (subgroup_miner/README.md)."""

from __future__ import annotations


def test_a_table_becomes_insights_and_llm_context(demo_csv):
    from subgroup_miner import MinerConfig, build_insights, describe, load_dataset, run_discovery, select_insights

    config = MinerConfig()
    table = load_dataset(demo_csv, config)
    found = run_discovery(table.frame, config)
    insights = select_insights(build_insights(found, config, dataset_id=table.dataset_id, batch_id="b1", filename=demo_csv.name), config).kept
    assert insights and all(0 < i.weight <= 1 for i in insights)
    context = describe(insights, limit=5)
    lines = context.splitlines()
    assert len(lines) == 5 and lines[0].startswith("1. ") and all(" sd " in line or "correlation" in line for line in lines)
    assert context.isascii()
