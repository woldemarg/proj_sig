"""Docker verification of the SIG service suite (docs/10_verification.md §10.5, docs/12_architecture.md).

    python scripts/compose_check.py               build, start on fresh volumes in dependency order, the workflow end to end, remove
    python scripts/compose_check.py --fresh       clean room: rebuild without the build cache and pull the base images
    python scripts/compose_check.py --tests       also run scripts/check.py inside the graph service's `test` image
    python scripts/compose_check.py --live-llm    also answer one question through the broker's real upstream (a paid call)
    python scripts/compose_check.py --keep        leave the stack running afterwards

Everything is reached the way a user reaches it: through the web console's nginx on one host port. The stack runs as
compose project ``sig-check`` on its own volumes (created for the run and removed after it), its own host ports and its
own Neo4j password, so it runs beside the user's own stack (project ``sig``) and never touches its data.
Needs Docker, ./models (the seed of the model volume; downloaded when it does not verify) and ./.env.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "sig-check"
PASSWORD = "sig-check-password"
WEB_PORT = 18765
CONSOLE = f"http://127.0.0.1:{WEB_PORT}"
QUESTION = "Why is margin lower for phones in the US?"
ENV = {k: v for k, v in os.environ.items() if not k.startswith("COMPOSE_")} | {
    "SIG_NEO4J_PASSWORD": PASSWORD,
    "SIG_WEB_PORT": str(WEB_PORT),
    "SIG_NEO4J_HTTP_PORT": "27474",
    "SIG_NEO4J_BOLT_PORT": "27687",
}
results: list[tuple[str, bool, str]] = []


def compose(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-p", PROJECT, "-f", str(ROOT / "compose.yaml"), *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=ENV, check=check, text=True, capture_output=capture)


def http(method: str, path: str, body: Any = None, timeout: float = 120, headers: dict[str, str] | None = None) -> Any:
    data = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
    hdrs = headers or ({"Content-Type": "application/json"} if data else {})
    req = urllib.request.Request(CONSOLE + path, data=data, method=method, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode()


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'ok' if ok else 'FAILED'}] {name}{': ' + detail if detail else ''}", flush=True)
    return ok


def cypher(query: str) -> list[str]:
    """Result rows of a query in the Neo4j container (plain format, header dropped)."""
    out = compose("exec", "-T", "neo4j", "cypher-shell", "-u", "neo4j", "-p", PASSWORD, "--format", "plain", query, capture=True)
    return out.stdout.strip().splitlines()[1:]


def mirror_counts() -> tuple[dict[str, int], int]:
    labels = {label.strip('"'): int(count) for label, count in (row.split(", ") for row in cypher("MATCH (n) RETURN labels(n)[0], count(*)"))}
    return labels, int(cypher("MATCH ()-[r]->() RETURN count(r)")[0])


def check_start_order(seconds: float) -> None:
    ids = compose("ps", "-a", "-q", capture=True).stdout.split()
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True))
    states = {c["Config"]["Labels"]["com.docker.compose.service"]: c["State"] for c in containers}
    started = {name: datetime.fromisoformat(s["StartedAt"]) for name, s in states.items()}
    held = sorted(
        f"{c['Config']['Labels']['com.docker.compose.service']}:{var.split('=')[0]}"
        for c in containers
        if c["Config"]["Labels"]["com.docker.compose.service"] != "llm-model-broker"
        for var in c["Config"]["Env"]
        if var.startswith(("GEMMA_", "LLM_API_KEY=")) and not var.endswith("=")
    )
    record("only the broker holds the provider settings and key", not held, ", ".join(held))
    init = states["model-init"]
    order = [
        init["ExitCode"] == 0 and datetime.fromisoformat(init["FinishedAt"]) <= started["insight-graph"],
        started["neo4j"] <= started["insight-graph"] <= started["evidence-narrator"],
        started["llm-model-broker"] <= started["evidence-narrator"] <= started["sig-web-console"],
    ]
    record("services started in dependency order", all(order), f"{seconds:.0f} s from up to all healthy")
    logs = compose("logs", "model-init", capture=True).stdout
    record(
        "model provisioned and verified",
        init["ExitCode"] == 0 and ("present" in logs or "copied from" in logs or "downloaded" in logs),
        logs.strip().splitlines()[-1] if logs.strip() else "",
    )


def check_services() -> None:
    page = http("GET", "/")
    record("console serves the UI", 'id="cy"' in page and "/vendor/plotly-gl3d.min.js" in page)
    record("graph service answers through the console", "graph" in http("GET", "/api/health"))
    chat = http("GET", "/api/chat/health")
    llm = chat["llm"]
    record(
        "narrator reaches the graph service and the broker",
        chat["insight_graph"]["reachable"] and llm["reachable"] and llm["model_available"],
        str(llm.get("model")),
    )
    record("Neo4j answers", cypher("RETURN 1") == ["1"])
    empty = http("POST", "/api/chat/query", {"question": QUESTION, "use_llm": False})
    record("an empty graph is an ordinary answer", empty["answer_mode"] == "empty")


def upload(path: Path) -> dict[str, Any]:
    """POST /api/upload as the browser sends it (multipart), through nginx."""
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: text/csv\r\n\r\n'.encode()
        + path.read_bytes()
        + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="bins"\r\n\r\n\r\n--{boundary}--\r\n'.encode()
    )
    return http("POST", "/api/upload", body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})


def wait_for_batch(batch_id: str, timeout: float = 1200) -> tuple[dict[str, Any], float]:
    """Poll the batch until it is final and mirrored; meanwhile, time /api/health (the service must stay responsive)."""
    slowest, deadline = 0.0, time.time() + timeout
    while time.time() < deadline:
        start = time.perf_counter()
        http("GET", "/api/health", timeout=30)
        slowest = max(slowest, time.perf_counter() - start)
        rec = http("GET", f"/api/batches/{batch_id}")
        if rec["status"] in {"FAILED", "SKIPPED"} or (rec["status"] == "READY" and "neo4j" in rec):  # the mirror syncs after READY
            return rec, slowest
        time.sleep(0.5)
    raise TimeoutError(f"batch {batch_id} not finished after {timeout:.0f} s")


def check_workflow(live_llm: bool) -> None:
    from scratch import scratch_dir, write_demo_csv

    csv = write_demo_csv(scratch_dir("compose_check"))
    rec, slowest = wait_for_batch(upload(csv)["batch_id"])
    m = rec.get("metrics", {})
    ok = rec["status"] == "READY"  # fresh volumes: a SKIPPED upload would mean old state
    detail = f"{m.get('validated_insights')} insights, {m.get('attractors_total')} anchors in {m.get('processing_duration_s', 0):.0f} s"
    record("upload through the console committed", ok, detail if ok else str(rec.get("error") or rec["status"]))
    record("health stays responsive during the ingest", slowest < 2.0, f"slowest /api/health {slowest * 1000:.0f} ms")
    graph = http("GET", "/api/graph")
    kinds = Counter(n["data"]["kind"] for n in graph["nodes"])
    mirrored, relationships = mirror_counts()
    same = mirrored == dict(kinds) and relationships == len(graph["edges"]) and rec.get("neo4j", {}).get("status") == "ok"
    record("Neo4j mirror equals the snapshot", same, f"{mirrored}, {relationships} relationships")
    qa = http("POST", "/api/chat/query", {"question": QUESTION, "use_llm": False}, timeout=300)
    items, cited = qa["view"]["evidence"].get("items", []), qa["citations"]
    ok = qa["answer_mode"] == "fallback" and items and cited["cited"] and not cited["unknown"]
    record("evidence narrated with a valid citation manifest", bool(ok), f"{len(items)} items, {len(cited['cited'])} cited")
    sphere = http("GET", "/api/sphere")
    record("sphere served as data", len(sphere["points"]) == kinds.get("Pattern", 0) + kinds.get("Attractor", 0), f"{len(sphere['points'])} points")
    if live_llm:
        qa = http("POST", "/api/chat/query", {"question": QUESTION, "use_llm": True}, timeout=300)
        detail = f"{qa['llm'].get('model')}, {qa['llm'].get('latency_s')} s{', ' + str(qa['llm'].get('error')) if qa['llm'].get('error') else ''}"
        record("LLM answer through the broker", qa["answer_mode"] == "llm" and qa["citations"]["grounded"], detail)
    reset = http("POST", "/api/reset")
    mirrored, relationships = mirror_counts()
    after = http("POST", "/api/chat/query", {"question": QUESTION, "use_llm": False})
    ok = (
        reset["status"] == "reset"
        and not http("GET", "/api/graph")["nodes"]
        and not mirrored
        and not relationships
        and after["answer_mode"] == "empty"
    )
    record("reset empties the graph and the mirror", ok)


def run_tests(fresh: bool) -> None:
    fresh_stage = ["--no-cache-filter", "test"] if fresh else []  # the runtime stage below it was just rebuilt by compose
    cmd = ["docker", "build", "-f", "insight_graph_service/Dockerfile", "--target", "test", *fresh_stage, "-t", "sig/insight-graph:test", "."]
    subprocess.run(cmd, cwd=ROOT, check=True)
    code = subprocess.call(["docker", "run", "--rm", "-v", f"{ROOT / 'models'}:/app/models:ro", "sig/insight-graph:test"], cwd=ROOT)
    record("test suite inside the graph service image", code == 0, f"exit {code}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fresh", action="store_true", help="rebuild without the build cache and pull the base images")
    parser.add_argument("--tests", action="store_true", help="also run the test suite inside the graph service image")
    parser.add_argument("--live-llm", action="store_true", help="also ask the real upstream model once (a paid call)")
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    args = parser.parse_args(argv)
    compose("down", "-v", "--remove-orphans")  # a run always starts from new volumes
    try:
        compose("build", *(["--no-cache", "--pull"] if args.fresh else []))
        start = time.perf_counter()
        compose("up", "-d", "--wait", "--wait-timeout", "900")
        check_start_order(time.perf_counter() - start)
        check_services()
        check_workflow(args.live_llm)
        if args.tests:
            run_tests(args.fresh)
    except Exception as exc:  # report and clean up whatever broke
        record("run", False, f"{type(exc).__name__}: {exc}")
        compose("logs", "--tail", "40", check=False)
    finally:
        if not args.keep:
            compose("down", "-v", "--remove-orphans", check=False)
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
