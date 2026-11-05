"""What a layer link says of itself (the cross-layer experiments of 2026-11-04, X-4, X-8, X-11): the time an event layer
covers carried onto every insight of its run — the card, the drawer and the prompt —, an aligned run with nothing to
count refused before it runs, and each refusal of a layer cohort with its metric and its reason in words: the dealt
layer reproduced the shift, or it rarely did but not after the correction for the run's layer cohorts."""

from __future__ import annotations

import pandas as pd
from conftest import make_config, make_engine
from fastapi.testclient import TestClient
from test_crosstype import RES, context_insight, cross_run, planted, upload_file

from insight_contracts import wording
from insight_contracts.graph import dataset_node_id
from insight_graph_service.server.app import create_app
from subgroup_miner.config import MinerConfig
from subgroup_miner.selection import select_insights


def timed(tmp_path, layer_dates: tuple[str, str]):
    """The planted target over 2025 and its schools as an event layer over ``layer_dates``."""
    target, context = planted(tmp_path)
    frame = pd.read_csv(target)
    frame["when"] = pd.date_range("2025-01-01", "2025-12-31", periods=len(frame)).strftime("%Y-%m-%d")
    frame.to_csv(target, index=False)
    schools = pd.read_csv(context)
    schools["when"] = pd.date_range(*layer_dates, periods=len(schools)).strftime("%Y-%m-%d")
    schools.to_csv(patrols := tmp_path / "patrols.csv", index=False)
    return target, patrols


def test_an_event_layer_run_carries_the_time_it_covers_onto_every_insight(engine, tmp_path):
    """A static reading of an event layer over another period is a link without a coincidence in time: the run's entry
    holds the share of the target's span the layer covers, the card words it in Ukrainian (no English warning), the
    drawer of every insight of the run says it and the prompt's dataset line names it."""
    target, patrols = timed(tmp_path, ("2025-07-01", "2026-06-30"))  # half of the target's year
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}", time="when")["upload_id"]
        b = upload_file(client, "patrols.csv", patrols, "lat,lon", layer="true", time="when")["upload_id"]
        run = engine.ws.load_batch(cross_run(client, a, b).json()["batch_id"])
        listed = next(x for x in client.get("/api/batches").json() if x["batch_id"] == run["batch_id"])
        found = [p["id"] for p in engine.ws.patterns() if p["dataset_id"] == run["dataset_id"]]
        assert found, run.get("error")
        drawer = client.get(f"/api/nodes/{found[0]}").json()
    (entry,) = run["context"]
    assert 0.49 < entry["overlap"] < 0.51 and entry["target_span"][0].startswith("2025-01-01")
    (line,) = listed["time_words"]
    assert "50 %" in line and "незмінн" in line and "2025-07-01" in line  # the card's words: its span, the share, the static reading
    assert "50 %" in drawer["link"]["time"]
    props = engine.graph().nodes[dataset_node_id(run["dataset_id"])]["props"]
    assert props["layer_time"][0]["overlap"] == entry["overlap"]
    prompt = engine.search(engine.graph().nodes[found[0]]["label"]).payload().evidence_prompt  # a question its insight answers
    assert "covers 50% of the target's span" in prompt


def test_an_aligned_run_with_no_layer_event_in_the_targets_span_is_refused_before_it_runs(engine, tmp_path):
    """``align=span`` with a layer whose events all fall outside the target's span counts nothing: the request is
    refused, naming both spans — never a run that fails on «no valid point»."""
    target, patrols = timed(tmp_path, ("2024-01-01", "2024-12-31"))
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}", time="when")["upload_id"]
        b = upload_file(client, "patrols.csv", patrols, "lat,lon", layer="true", time="when")["upload_id"]
        refused = cross_run(client, a, b, align="span")
    assert refused.status_code == 400 and refused.json()["code"] == "invalid_options"
    detail = refused.json()["detail"]
    assert "2024-01-01" in detail and "2025-01-01" in detail and "no event" in detail
    assert not [r for r in engine.ws.list_batches() if r.get("cross")]


def test_a_layer_cohort_refused_only_by_the_correction_has_its_own_reason_and_its_metric():
    """The layer's null refuses a cohort two ways: the dealt layer reproduces its shift (``context_confounded``, raw
    ``p_context`` above the gate), or it rarely does but not rarely enough once multiplied by the run's layer cohorts
    (``context_family``). Each refusal names the metric it tested."""
    config = MinerConfig()
    reproduced = select_insights([context_insight(0.3)], config).rejections
    corrected = select_insights([context_insight(0.01, family=10)], config).rejections
    assert [(r.reason, r.metric) for r in reproduced] == [("context_confounded", "points")]
    assert [(r.reason, r.metric) for r in corrected] == [("context_family", "points")] and "10" in corrected[0].detail
    assert wording.TERMS["context_family"][0] and "поправк" in wording.TERMS["context_family"][1]


def test_a_runs_refusals_read_in_words(tmp_path):
    """``GET /api/datasets/{id}/rejections``: each refused cohort of a run with its conditions and its metric in the
    console's words, the raw expression kept beside them."""
    target, context = planted(tmp_path)
    engine = make_engine(make_config(tmp_path / "ws", context_permutations=19))
    with TestClient(create_app(engine)) as client:
        a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
        b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
        run = engine.ws.load_batch(cross_run(client, a, b).json()["batch_id"])
        rows = client.get(f"/api/datasets/{run['dataset_id']}/rejections").json()
    layered = [r for r in rows if "context_schools_band" in r["expression"]]
    assert layered and all(r["where"] and "schools" in r["where"] and "context_schools_band" not in r["where"] for r in layered)
    assert all(r["metric_words"] for r in rows if r.get("metric"))
