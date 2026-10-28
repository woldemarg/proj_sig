"""The shipped datasets the console offers in its list (docs/10_verification.md §10.2): the files are the pinned tables
of ``scripts/demo_datasets.py``, each readable with its options, and one uploads by its key."""

from __future__ import annotations

import csv
import hashlib
import time

from fastapi.testclient import TestClient

from insight_graph_service.core.demo import DEMO_DIR, demo_datasets
from insight_graph_service.core.engine import TERMINAL
from insight_graph_service.server.app import create_app


def test_every_shipped_dataset_is_its_pinned_file_and_names_columns_it_has():
    shipped = demo_datasets()
    assert len({d["key"] for d in shipped}) == len({d["file"] for d in shipped}) == len(shipped) >= 11
    assert {d["key"] for d in shipped} >= {"retail_synthetic", "california_housing", "sf_fire_calibration", "sf_fire_holdout_a", "sf_fire_holdout_b"}
    for d in shipped:
        path = DEMO_DIR / d["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == d["sha256"] and path.stat().st_size == d["bytes"] < 30 * 2**20, d["key"]
        with path.open(encoding="utf-8", newline="") as f:
            header, rows = next(csv.reader(f)), sum(1 for _ in f)
        assert rows == d["rows"] and d["title"] and d["description"] and d["license"] and d["source"], d["key"]
        options = d["options"]
        assert options.get("time") is None or options["time"] in header, d["key"]
        locator = options.get("geo", "").partition("@")[0]
        assert locator in ("", "auto") or set(locator.split(",")) <= set(header), d["key"]
        assert len(options.get("source", "")) <= 120  # the form's field
    assert (DEMO_DIR / "SOURCES.md").read_text(encoding="utf-8").count("## ") == len(shipped)


def test_a_shipped_dataset_uploads_by_its_key_with_the_forms_options(engine):
    with TestClient(create_app(engine)) as client:
        listed = client.get("/api/demo/datasets").json()
        assert [d["key"] for d in listed] == [d["key"] for d in demo_datasets()]
        accepted = client.post("/api/upload", data={"demo": "retail_synthetic", "bins": "", "categories": "", "source": "SIG"})
        assert accepted.status_code == 202 and (accepted.json()["filename"], accepted.json()["source"]) == ("retail_synthetic.csv", "SIG")
        for _ in range(600):  # the batch runs on the app's worker: it ends before the test does, or it runs on through the next tests
            if client.get(f"/api/batches/{accepted.json()['batch_id']}").json()["status"] in TERMINAL:
                break
            time.sleep(0.5)
        with (DEMO_DIR / "retail_synthetic.csv").open("rb") as f:
            both = client.post("/api/upload", files={"file": ("x.csv", f, "text/csv")}, data={"demo": "retail_synthetic"})
        neither, unknown = client.post("/api/upload", data={"bins": ""}), client.post("/api/upload", data={"demo": "../etc"})
    assert both.status_code == neither.status_code == unknown.status_code == 400
