"""The console reads a layer link in words (the cross-layer experiments of 2026-11-04, X-7, X-9, X-14): a condition and a
metric in the words every reading uses, not the column's name; one effect found under several conditions as one row with
its variants; the «no layer nearby» mirror of a finding marked as its complement; the bands of a layer in the concept's
words; a zoned value read as relative to its zone; the drivers in Ukrainian; the layer's points beside a focused insight."""

from __future__ import annotations

from conftest import make_config, make_engine, toy_insight
from fastapi.testclient import TestClient
from test_crosstype import RES, cross_run, planted, upload_file

from insight_contracts import Condition, PhenomenonThresholds, Shift, wording
from insight_graph_service.server.app import create_app
from insight_graph_service.server.views import variant_groups


def run_of(tmp_path):
    target, context = planted(tmp_path)
    engine = make_engine(make_config(tmp_path / "ws", context_permutations=19))
    client = TestClient(create_app(engine))
    a = upload_file(client, "target.csv", target, f"lat,lon@{RES}")["upload_id"]
    b = upload_file(client, "schools.csv", context, "lat,lon", layer="true")["upload_id"]
    return engine, client, engine.ws.load_batch(cross_run(client, a, b).json()["batch_id"])


def test_a_runs_insights_carry_their_conditions_and_metric_in_words(tmp_path):
    """``GET /api/graph``: a Pattern node holds ``words`` (its conditions as the drawer reads them) and ``metric_words``,
    so the table and its chips never print a column's name; the batch card's dimensions come in words too."""
    engine, client, run = run_of(tmp_path)
    with client:
        nodes = [n["data"] for n in client.get("/api/graph").json()["nodes"] if n["data"].get("dataset") == run["dataset_id"]]
        listed = next(x for x in client.get("/api/batches").json() if x["batch_id"] == run["batch_id"])
    assert nodes and all(d["words"] and d["metric_words"] for d in nodes)
    assert not any("_band" in w or "context_" in w for d in nodes for w in d["words"])
    assert any("schools" in w for d in nodes for w in d["words"])  # the layer named by its file, not by its column
    words = listed["dimension_words"]
    assert words["context_schools_band"] and "_" not in words["context_schools_band"]


def test_one_effect_under_several_conditions_is_one_row_and_a_mirror_is_its_complement():
    """Within a run, insights of one metric and one direction whose places nest or overlap by half are one effect: the
    strongest leads, the others are its variants; an insight on «none nearby» shifting a metric the other way than a
    leader on a band of the layer is that leader's complement. Another direction, or no shared place, stays apart."""
    items = [  # (id, target, rises, the layer's band, weight)
        ("a", "count_x", True, "q1", 0.8),
        ("b", "count_x", True, "q2", 0.6),
        ("c", "count_x", True, "q1", 0.5),
        ("g", "count_x", False, "q1", 0.6),  # the other direction: its own row
        ("f", "count_x", True, "q3", 0.3),  # no shared place: its own row
        ("e", "share_y", True, "q1", 0.7),
        ("d", "share_y", False, "none nearby", 0.4),  # the mirror of e
    ]
    linked = [("a", "b"), ("b", "c"), ("a", "g")]
    groups = variant_groups(items, linked)
    assert groups == {"b": {"variant_of": "a"}, "c": {"variant_of": "a"}, "d": {"complement_of": "e"}}


def test_a_layers_bands_and_a_zoned_value_read_in_the_concepts_words():
    """A context band reads *немає / мало / нижче середнього / вище середнього / найбільше* over its disk (its top band
    *найбільше* whatever the number of bands); a zoned value is read against its zone, not as a count."""
    band = "context_bars_band"
    four, three = {band: 5}, {band: 4}  # the Dimension's cardinality: the positive bands and «none nearby»
    read = lambda value, bands: wording.condition_words(Condition(band, value), {}, True, bands, 0)  # noqa: E731
    assert [read(v, four) for v in ("q1", "q2", "q3", "q4")] == [
        "кількість bars у самій комірці: мало",
        "кількість bars у самій комірці: нижче середнього",
        "кількість bars у самій комірці: вище середнього",
        "кількість bars у самій комірці: найбільше",
    ]
    assert read("q3", three).endswith(": найбільше") and read("none nearby", four) == "немає bars у самій комірці"
    zoned = toy_insight([(band, "q2")], [Shift("count_crime_a", 0.6, 0.29, 0.0, 0.4)], provenance={"spatial": {"baseline": "zone", "resolution": 9}})
    what = wording.reading(zoned, PhenomenonThresholds())["what"]
    assert "відносно своєї зони" in what and "медіана" not in what


def test_the_drivers_read_in_ukrainian():
    """The EDA's drivers (stored English, embedded and signed) are read in words: a categorical the cohort leans to, a
    hidden shift of another metric."""
    assert wording.driver_words("[district_mode] heavily skewed to 'district 16' (JS: 0.52)", {}, True) == "district: переважає «district 16»"
    assert wording.driver_words("[load_median] hidden shift (+2.1 robust sigma)", {}, True).endswith("прихований зсув +2.1 sd")
    assert wording.driver_words("anything else", {}, True) == "anything else"


def test_a_focused_run_insight_brings_its_layers_points(tmp_path):
    """``GET /api/geo/{ds}/patterns/{pid}`` of a run's insight holds its layer's points in and around its cells, so the
    map draws the two layers side by side; an insight of no layer holds none."""
    engine, client, run = run_of(tmp_path)
    found = [p["id"] for p in engine.ws.patterns() if p["dataset_id"] == run["dataset_id"]]
    with client:
        drawn = client.get(f"/api/geo/{run['dataset_id']}/patterns/{found[0]}").json()
    (layer,) = drawn["layer_points"]
    assert layer["name"] == "schools" and layer["points"] and all(len(p) == 2 for p in layer["points"])
