"""Docker verification of SIG's three services (docs/10_verification.md §10.5, docs/12_architecture.md §12.5).

    python scripts/compose_check.py               build, start on fresh volumes, health checks, the demo end to end, remove
    python scripts/compose_check.py --fresh       clean room: rebuild without the build cache and pull the base images
    python scripts/compose_check.py --tests       also run scripts/check.py inside the backend's `test` image
    python scripts/compose_check.py --live-llm    also answer one question through the gateway's real upstream (a paid call)
    python scripts/compose_check.py --keep        leave the stack running afterwards

The stack runs as compose project ``sig-check`` on its own volumes (created for the run and removed
after it), its own host ports and its own Neo4j password, so it runs beside the user's own stack
(project ``sig``) and never touches its data. Needs Docker, ./models and ./.env.gemma.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "sig-check"
PASSWORD = "sig-check-password"
WEB_PORT = 18765
BACKEND = f"http://127.0.0.1:{WEB_PORT}"
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


def http(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 120) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'ok' if ok else 'FAILED'}] {name}{': ' + detail if detail else ''}", flush=True)
    return ok


def cypher(query: str) -> list[str]:
    """Result rows of a query in the Neo4j container (plain format, header dropped)."""
    out = compose("exec", "-T", "neo4j", "cypher-shell", "-u", "neo4j", "-p", PASSWORD, "--format", "plain", query, capture=True)
    return out.stdout.strip().splitlines()[1:]


def gateway(path: str) -> Any:
    """GET on the gateway from inside the stack (it publishes no host port)."""
    code = f"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080{path}', timeout=10).read().decode())"
    return json.loads(compose("exec", "-T", "llm", "python", "-c", code, capture=True).stdout)


def wait_for_batch(batch_id: str, timeout: float = 1200) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = http("GET", f"{BACKEND}/api/batches/{batch_id}")
        if rec["status"] in {"FAILED", "SKIPPED"} or (rec["status"] == "READY" and "neo4j" in rec):  # the mirror syncs after READY
            return rec
        time.sleep(3)
    raise TimeoutError(f"batch {batch_id} not finished after {timeout:.0f} s")


def check_services() -> None:
    record("backend answers", "embedding_model" in http("GET", f"{BACKEND}/api/config"))
    health = gateway("/health")
    record("gateway configured", health["status"] == "ok", f"model {health['model']} via {health['upstream']}")
    models = [m["id"] for m in gateway("/v1/models")["data"]]
    llm = http("GET", f"{BACKEND}/api/health")["llm"]
    record("backend reaches the gateway", llm["reachable"] and llm["model_available"] and llm["model"] in models, llm.get("base_url", ""))
    record("Neo4j answers", cypher("RETURN 1") == ["1"])


def check_workflow(live_llm: bool) -> None:
    rec = wait_for_batch(http("POST", f"{BACKEND}/api/demo")["batch_id"])
    m = rec.get("metrics", {})
    ok = rec["status"] == "READY"  # fresh volumes: a SKIPPED demo would mean old state
    detail = f"{m.get('validated_insights')} insights, {m.get('attractors_total')} anchors in {m.get('processing_duration_s', 0):.0f} s"
    record("demo batch committed", ok, detail if ok else str(rec.get("error") or rec["status"]))
    graph = http("GET", f"{BACKEND}/api/graph")
    kinds: dict[str, int] = {}
    for n in graph["nodes"]:
        kinds[n["data"]["kind"]] = kinds.get(n["data"]["kind"], 0) + 1
    record("graph served", kinds.get("Pattern", 0) > 0 and kinds.get("Attractor", 0) > 0, f"{kinds}, {len(graph['edges'])} edges")
    mirrored = {label.strip('"'): int(count) for label, count in (row.split(", ") for row in cypher("MATCH (n) RETURN labels(n)[0], count(*)"))}
    relationships = int(cypher("MATCH ()-[r]->() RETURN count(r)")[0])
    same = mirrored == kinds and relationships == len(graph["edges"]) and rec.get("neo4j", {}).get("status") == "ok"
    record("Neo4j mirror equals the snapshot", same, f"{mirrored}, {relationships} relationships")
    qa = http("POST", f"{BACKEND}/api/query", {"question": QUESTION, "use_llm": False})
    items = qa["evidence"].get("items", [])
    record("evidence for a question", qa["answer_mode"] == "fallback" and bool(items) and bool(qa["highlight"]["seeds"]), f"{len(items)} items")
    if live_llm:
        qa = http("POST", f"{BACKEND}/api/query", {"question": QUESTION, "use_llm": True}, timeout=300)
        detail = f"{qa['llm'].get('model')}, {qa['llm'].get('latency_s')} s{', ' + str(qa['llm'].get('error')) if qa['llm'].get('error') else ''}"
        record("LLM answer through the gateway", qa["answer_mode"] == "llm" and qa["citations"]["grounded"], detail)


def run_tests(fresh: bool) -> None:
    fresh_stage = ["--no-cache-filter", "test"] if fresh else []  # the runtime stage below it was just rebuilt by compose
    subprocess.run(["docker", "build", "--target", "test", *fresh_stage, "-t", "sig-backend-test", "."], cwd=ROOT, check=True)
    code = subprocess.call(["docker", "run", "--rm", "-v", f"{ROOT / 'models'}:/app/models:ro", "sig-backend-test"], cwd=ROOT)
    record("test suite inside the backend image", code == 0, f"exit {code}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fresh", action="store_true", help="rebuild without the build cache and pull the base images")
    parser.add_argument("--tests", action="store_true", help="also run the test suite inside the backend image")
    parser.add_argument("--live-llm", action="store_true", help="also ask the real upstream model once (a paid call)")
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    args = parser.parse_args(argv)
    compose("down", "-v", "--remove-orphans")  # a run always starts from new volumes
    try:
        compose("build", *(["--no-cache", "--pull"] if args.fresh else []))
        start = time.perf_counter()
        compose("up", "-d", "--wait", "--wait-timeout", "900")
        record("stack up", True, f"{time.perf_counter() - start:.0f} s to healthy")
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
