"""The performance benchmark (docs/10_verification.md §10.4): the gauge of every optimisation in epics #83 and #84.

One upload through the graph service's HTTP API on a scratch workspace (the configured model, Neo4j off): each level's
stage times; then what the console asks of the result — the graph, the widest insight's node, each geo level's cells,
overlay, projected insights and window, three questions — once right after the commit (cold) and again (warm), with
the response sizes, raw and gzipped, and the stored snapshot's and cell files' sizes.

    python scripts/perf_benchmark.py <file | synthetic | california> [--geo auto@auto] [--table] [--console] [--browser]
                                     [--label NAME]

``synthetic``: ``SYNTHETIC_POINTS`` synthetic deliveries over California (``write_synthetic_points``); ``california``:
the map demo's table. ``--console``: while the levels process, a second client asks what an open console asks (the
batch list every 0.9 s, the reads above after every new READY). ``--browser``: the console in headless Chromium, as
``tests/test_ui_smoke.py`` drives it (skipped without one): the time to the graph and to the map, the map's time to
idle after each step of a zoom sequence, the most DOM markers on the map, frames per second during an animated zoom,
long tasks. Writes ``.scratch/results/perf_benchmark_<label>.json`` and prints a table.
"""

from __future__ import annotations

import argparse
import gzip
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the packages live one level up

from fastapi.testclient import TestClient  # noqa: E402

from insight_graph_service.core.engine import Engine  # noqa: E402
from insight_graph_service.core.settings import PROJECT_ROOT  # noqa: E402
from insight_graph_service.server.app import create_app  # noqa: E402

TERMINAL = {"READY", "FAILED", "SKIPPED"}
POLL_S = 0.9  # the console's batch poll
SYNTHETIC_POINTS = 300_000
# The synthetic set's own territory and calendar, chosen without regard to any real data (AGENTS.md, §Boundaries):
CALIFORNIA = (32.6, -124.3, 41.9, -114.2)  # south, west, north, east (the map demo's state)
EPOCH = "2019-03-04"  # the first day of its calendar
METROS = ((34.05, -118.25, 0.30), (37.77, -122.42, 0.20), (32.72, -117.16, 0.15), (38.58, -121.49, 0.15),
          (36.74, -119.79, 0.18), (37.34, -121.89, 0.12), (35.37, -119.02, 0.15), (33.95, -117.40, 0.20))  # fmt: skip
ZOOMS = (8, 10, 12, 9, 7)  # the audit's zoom sequence


def write_synthetic_points(folder: Path, n: int = SYNTHETIC_POINTS, seed: int = 5) -> Path:
    """Synthetic deliveries over California: 70 % of the points around eight metros (``METROS``: lat, lon, spread in
    degrees), the rest over the state's box, named by area (the nearest metro within 0.6°, else outside), with planted places for the miner to find: business customers twice as
    common around two metros, phone orders three times as common in the north, bikes in the metro cores and trucks
    outside the metros, a basket half as high again around two metros, two more items around one, a wait growing with
    the distance from the nearest metro centre and ten minutes longer around one, ratings falling with the wait, refunds
    four times as common around one. Its own generator and schema; the same ``n`` and ``seed`` write the same file."""
    path = folder / f"deliveries_synthetic_{n}_{seed}.csv"
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    south, west, north, east = CALIFORNIA
    metros = np.array(METROS)
    near = int(0.7 * n)
    home = rng.integers(0, len(metros), near)
    lat = np.clip(np.r_[rng.normal(metros[home, 0], metros[home, 2]), rng.uniform(south, north, n - near)], south, north)
    lon = np.clip(np.r_[rng.normal(metros[home, 1], 1.2 * metros[home, 2]), rng.uniform(west, east, n - near)], west, east)
    nearest = np.hypot(lat[:, None] - metros[:, 0], lon[:, None] - metros[:, 1]).min(axis=1)
    metro = np.r_[home, np.full(n - near, -1)]  # the metro a point was drawn around, -1 for the state-wide rest
    u = rng.random((5, n))
    business = np.where(np.isin(metro, [1, 5]), 0.5, 0.25)
    wait = 20 + 15 * nearest + 10 * (metro == 6) + rng.normal(0, 5, n)
    frame = pd.DataFrame(
        {
            "lat": lat.round(5),
            "lon": lon.round(5),
            "ordered_at": np.datetime64(EPOCH) + rng.integers(0, 60, n).astype("timedelta64[D]"),
            "segment": np.where(u[0] < business, "business", np.where(u[0] < business + 0.1, "public", "consumer")),
            "channel": np.where(u[1] < np.where(lat > 37.0, 0.3, 0.1), "phone", np.where(u[2] < 0.55, "web", "store")),
            "area": np.where(
                nearest < 0.6,
                np.array([f"metro {i + 1}" for i in range(len(metros))])[
                    np.argmin(np.hypot(lat[:, None] - metros[:, 0], lon[:, None] - metros[:, 1]), axis=1)
                ],
                "outside",
            ),
            "vehicle": np.where(nearest < 0.08, np.where(u[3] < 0.4, "bike", "van"), np.where((metro < 0) & (u[3] < 0.5), "truck", "van")),
            "refunded": np.where(u[4] < np.where(metro == 7, 0.2, 0.05), "yes", "no"),
            "basket": (rng.gamma(2.0, 20.0, n) * np.where(np.isin(metro, [1, 4]), 1.5, 1.0)).round(2),
            "items": rng.poisson(3, n) + 1 + 2 * (metro == 2),
            "wait_min": wait.round(1),
            "rating": np.clip(np.round(5.2 - 0.04 * wait + rng.normal(0, 0.6, n)), 1, 5).astype(int),
        }
    )
    frame.to_csv(path, index=False)
    return path


def timed(client: TestClient, method: str, path: str, **kw: Any) -> dict[str, Any]:
    start = time.perf_counter()
    resp = client.request(method, path, **kw)
    seconds = time.perf_counter() - start
    return {
        "path": path,
        "status": resp.status_code,
        "s": round(seconds, 3),
        "bytes": len(resp.content),
        "gzip_bytes": len(gzip.compress(resp.content, 6)),
    }


def widest_insight(client: TestClient) -> dict[str, Any] | None:
    nodes = [n["data"] for n in client.get("/api/graph").json()["nodes"] if n["data"]["kind"] == "Pattern"]
    return max(nodes, key=lambda d: d["support"]) if nodes else None


def console_reads(client: TestClient) -> list[dict[str, Any]]:
    """What an open console asks after a READY: the graph, the geo datasets and each one's map reads, the widest node."""
    out = [timed(client, "GET", "/api/graph"), timed(client, "GET", "/api/geo")]
    for d in client.get("/api/geo").json():
        ds = d["dataset_id"]
        out += [timed(client, "GET", f"/api/geo/{ds}/{part}") for part in ("cells", "overlay", "projected")]
        out.append(timed(client, "GET", f"/api/geo/{ds}/window", params={"bbox": "-180,-90,180,90"}))
    if widest := widest_insight(client):
        out.append(timed(client, "GET", f"/api/nodes/{widest['id']}"))
    return out


def watch(app: Any, stop: threading.Event, log: list[dict[str, Any]]) -> None:
    """The open console: the batch list every ``POLL_S``, its reads after each new READY (no lifespan of its own: the
    benchmark's client owns the batch worker)."""
    client, seen = TestClient(app), set()
    while not stop.is_set():
        ready = {b["batch_id"] for b in client.get("/api/batches").json() if b["status"] == "READY"}
        if ready - seen:
            seen = ready
            log.append({"ready": len(ready), "reads": console_reads(client)})
        stop.wait(POLL_S)


def finished(record: dict[str, Any]) -> bool:
    """A batch at its end: a READY one once its last save, after the Neo4j sync, records the mirror's status."""
    return record["status"] in TERMINAL and (record["status"] != "READY" or "neo4j" in record)


def run_upload(client: TestClient, app: Any, path: Path, geo: str | None, table: bool, console: bool) -> dict[str, Any]:
    """The upload through the API and every level to its end; the levels' stage times and the console's reads meanwhile."""
    stop, log = threading.Event(), []
    watcher = threading.Thread(target=watch, args=(app, stop, log), daemon=True)
    if console:
        watcher.start()
    data = {"bins": ""} | ({"geo": geo} if geo else {}) | ({"table": "true"} if table else {})
    start = time.perf_counter()
    with path.open("rb") as f:
        accepted = client.post("/api/upload", files={"file": (path.name, f)}, data=data).json()
    records = []
    for bid in (accepted.get("levels") or {"table": accepted["batch_id"]}).values():
        while not finished(rec := client.get(f"/api/batches/{bid}").json()):
            time.sleep(0.2)
        records.append(rec)
    wall = time.perf_counter() - start
    stop.set()
    if console:
        watcher.join()
    levels = [
        {
            "level": r.get("level"),
            "status": r["status"],
            "insights": r["metrics"].get("validated_insights", 0),
            "s": round(r["metrics"].get("processing_duration_s", 0.0), 2),
            "timings": {k: round(v, 2) for k, v in r["metrics"].get("timings", {}).items()},
            "error": (r.get("error") or {}).get("code"),
        }
        for r in records
    ]
    return {"wall_s": round(wall, 2), "levels": levels, "console": log}


def reads_after(client: TestClient, workspace: Path) -> dict[str, Any]:
    """Every console read twice (cold, the first after the commit; warm), three questions, and the stored sizes."""
    cold, warm = console_reads(client), console_reads(client)
    widest = widest_insight(client)
    target = widest["target"].replace("_", " ") if widest else "anything"
    asked = [
        timed(client, "POST", "/api/search", json={"question": q}) | {"question": q}
        for q in (f"Where is {target} higher?", f"Where is {target} lower?", f"What is special about {target}?")
    ]
    sizes = {
        p.relative_to(workspace).as_posix(): p.stat().st_size
        for p in [workspace / "graph" / "snapshot.json", *workspace.glob("datasets/*/cells.geojson")]
        if p.exists()
    }
    return {"cold": cold, "warm": warm, "questions": asked, "bytes": sizes}


def browser(engine: Engine) -> dict[str, Any] | str:
    """The console in headless Chromium against this workspace: what the audit of 2026-10-16 measured by hand."""
    from dev import chromium, compose

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "skipped: playwright is not installed"
    if (exe := chromium()) is None:
        return "skipped: no local Chromium"
    import uvicorn

    from evidence_narrator_service.app import create_app as create_narrator
    from evidence_narrator_service.settings import NarratorSettings

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    narrator = create_narrator(NarratorSettings(insight_graph_url=f"http://127.0.0.1:{port}"))
    server = uvicorn.Server(uvicorn.Config(compose(create_app(engine), narrator), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    idle_after = "(z) => new Promise((done) => { const t = performance.now(); G.map.once('idle', () => done(performance.now() - t)); G.map.jumpTo({ zoom: z }); })"
    fps = """() => new Promise((done) => { let frames = 0; const tick = () => { frames++; if (G.map.isMoving()) requestAnimationFrame(tick); };
      G.map.once('moveend', () => done(frames / 2.5)); G.map.zoomTo(G.map.getZoom() + 2, { duration: 2500 }); requestAnimationFrame(tick); })"""
    out: dict[str, Any] = {}
    try:
        with sync_playwright() as p:
            page = p.chromium.launch(executable_path=exe).new_page(viewport={"width": 1500, "height": 900})
            page.add_init_script(
                "window.__long = []; new PerformanceObserver((l) => l.getEntries().forEach((e) => window.__long.push(e.duration))).observe({ type: 'longtask', buffered: true });"
            )
            start = time.perf_counter()
            page.goto(f"http://127.0.0.1:{port}/")
            page.wait_for_function("typeof S !== 'undefined' && S.cy && S.cy.nodes('[kind=\"Pattern\"]').length > 0", timeout=120000)
            out["graph_s"] = round(time.perf_counter() - start, 2)
            if not engine.geo_datasets():
                return out | {"map": "no geo dataset"}
            page.wait_for_function("!document.getElementById('map-tab').hidden", timeout=600000)  # after the geo list loads
            start = time.perf_counter()
            page.click("#view-switch button[data-view='map']")
            page.wait_for_function("G.data && G.map && G.map.loaded() && !G.map.isMoving()", timeout=120000)
            out["map_s"] = round(time.perf_counter() - start, 2)
            out["zoom_idle_ms"], markers = {}, []
            for z in ZOOMS:
                out["zoom_idle_ms"][z] = round(page.evaluate(idle_after, z))
                markers.append(page.evaluate("document.querySelectorAll('.maplibregl-marker').length"))
            out["dom_markers_max"] = max(markers)
            out["fps_zoom_in"] = round(page.evaluate(fps), 1)
            long = page.evaluate("window.__long")
            out["long_tasks"] = {"n": len(long), "max_ms": round(max(long, default=0)), "total_ms": round(sum(long))}
    finally:
        server.should_exit = True
    return out


def table(result: dict[str, Any]) -> str:
    up, reads = result["upload"], result["reads"]
    rows = [f"{result['source']}  geo={result['geo']}  table={result['table']}  upload {up['wall_s']} s"]
    for lv in up["levels"]:
        stages = " ".join(f"{k}={v}" for k, v in lv["timings"].items())
        rows.append(f"  level {lv['level']:>5}  {lv['status']:<7} {lv['insights']:>3} insights {lv['s']:>8} s  {stages}")
    for kind in ("cold", "warm"):
        rows += [
            f"  {kind:<4} {r['path'][:60]:<60} {r['s']:>7.3f} s {r['bytes'] / 1e6:>7.2f} MB (gzip {r['gzip_bytes'] / 1e6:.2f})" for r in reads[kind]
        ]
    rows += [f"  ask  {q['question'][:60]:<60} {q['s']:>7.3f} s" for q in reads["questions"]]
    rows += [f"  {k}: {v / 1e6:.2f} MB" for k, v in reads["bytes"].items()]
    if up["console"]:
        slowest = max((r for rnd in up["console"] for r in rnd["reads"]), key=lambda r: r["s"])
        rows.append(f"  console: {len(up['console'])} rounds of reads while processing, slowest {slowest['path']} {slowest['s']} s")
    if "browser" in result:
        rows.append(f"  browser: {json.dumps(result['browser'])}")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="a file, 'synthetic' or 'california'")
    parser.add_argument("--geo", default=None, help="the geo option, e.g. auto@auto or lat,lon@7,5")
    parser.add_argument("--table", action="store_true", help="the table level too")
    parser.add_argument("--console", action="store_true", help="a console asking while the levels process")
    parser.add_argument("--browser", action="store_true", help="the console in headless Chromium afterwards")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    from scratch import build_engine, save_result, scratch_dir  # loads .env, as the services do

    root = scratch_dir(f"perf_benchmark_{args.label}", fresh=True)
    path = {
        "synthetic": lambda: write_synthetic_points(scratch_dir("perf_benchmark_data")),
        "california": lambda: PROJECT_ROOT / "data" / "demo" / "california_housing.csv",
    }.get(args.source, lambda: Path(args.source))()
    engine = build_engine(root / "ws")
    app = create_app(engine)
    result: dict[str, Any] = {"source": args.source, "geo": args.geo, "table": args.table}
    with TestClient(app) as client:  # its lifespan owns the batch worker
        result["upload"] = run_upload(client, app, path, args.geo, args.table, args.console)
        result["reads"] = reads_after(client, root / "ws")
    if args.browser:
        result["browser"] = browser(engine)
    print(table(result))
    print(save_result("perf_benchmark", args.label, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
