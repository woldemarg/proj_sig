# subgroup_miner

`subgroup_miner` finds the subgroups of a table whose metrics behave unusually. Each finding is a validated, weighted `Insight` with its scope, the shift or correlation change, and its statistics. The method is subgroup discovery with robust median shifts and Exceptional Model Mining, closed intents and bootstrap validation; [docs/02_discovery.md](../docs/02_discovery.md) and [docs/03_insights.md](../docs/03_insights.md) describe it.

It depends only on `insight_contracts` and on pandas, numpy, scipy and pysubgroup (`requirements.txt`). Use it alone, for example to give an LLM statistical context about a table, or feed its insights to `attractor_topology`.

```python
from subgroup_miner import MinerConfig, build_insights, describe, load_dataset, run_discovery, select_insights

config = MinerConfig()  # defaults; every field is documented in config.py
table = load_dataset("sales.csv", config)  # validation, optional quantile bands, content-addressed id
found = run_discovery(table.frame, config)  # candidates -> closed intents -> pruning -> bootstrap validation
insights = select_insights(
    build_insights(found, config, dataset_id=table.dataset_id, batch_id="run-1", filename="sales.csv"), config
).kept  # validity rules R1–R3 + evidence weight, heaviest first

prompt_context = describe(insights, limit=10)  # numbered ASCII lines, ready for a prompt
```

`Insight.to_record()` is a plain dictionary that can be stored or sent anywhere. `structural_edges(insights, config)` adds the subgroup lattice: SPECIALIZES, GENERALIZES, SIBLING and CONTRASTS between the scopes.

The EDA engine under `vendor/eda/` is a modified copy (`vendor/PROVENANCE.md`). `tests/test_subgroup_miner_standalone.py` runs the example above; `tests/test_architecture.py` checks that the package imports only the kernel.
