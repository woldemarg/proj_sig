"""Embedder comparison (SDD 06, SDD 15): representation contract, benchmark, cost.

    python scripts/compare_embedders.py --models minilm,qwen3 --k 5

Per candidate: (a) contract cosines on synthetic insights — direction (same scope/target,
opposite shift; must be < 0), cross-scope (same phenomenon, disjoint scopes; high) and
entity (same scope, different phenomenon; must be below cross-scope), plus the raw label
cosine ``E(discount)·E(margin)``; (b) the demo hypothesis benchmark (``ltir.experiment``);
(c) load time, ingest embedding time, peak CUDA memory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np
from scratch import build_engine, ingest_quietly, save_result, scratch_dir, write_demo_csv

from ltir.canonical import canonicalize
from ltir.config import load_config
from ltir.encoder import InsightEncoder, make_text_embedder
from ltir.experiment import run_experiment
from ltir.models import Condition, Insight, Shift

CANDIDATES: dict[str, dict] = {
    "minilm": {"embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"},
}


def synthetic_insight(scope: dict[str, str], shifts: list[tuple[str, float]]) -> Insight:
    """Minimal shift insight for the contract cosines."""
    conds = tuple(sorted(Condition(k, v) for k, v in scope.items()))
    sh = tuple(Shift(m, z, 10.0 + z, 10.0, 1.0) for m, z in shifts)
    return Insight(
        id="P-x",
        dataset_id="d",
        batch_id="b",
        conditions=conds,
        expression="x",
        target=sh[0].metric,
        shifts=sh,
        support=300,
        support_fraction=0.1,
        baseline=10.0,
        local=10.0 + sh[0].robust_z,
        effect_size=sh[0].robust_z,
        sd_score=1.0,
        sd_raw_score=1.0,
        emm_score=0.0,
        volume_utility=0.3,
        stability=0.9,
        integrated_index=1.0,
        p_value=1e-9,
        p_adjusted=1e-6,
        drivers=(),
        row_hash="x",
    )


def contract_cosines(encoder: InsightEncoder) -> dict[str, float]:
    """Direction, cross-scope and entity cosines of the composed insight vectors."""
    cfg = load_config()
    a = synthetic_insight({"region": "EU", "category": "laptops"}, [("margin", 2.0)])
    b = synthetic_insight({"region": "EU", "category": "laptops"}, [("margin", -2.0)])
    c = synthetic_insight({"region": "US", "category": "phones"}, [("discount", 2.2), ("margin", -1.1)])
    d = synthetic_insight({"region": "APAC", "category": "tablets"}, [("discount", 2.1), ("margin", -1.0)])
    e = synthetic_insight({"region": "US", "category": "phones"}, [("return_rate", 2.0)])
    v = encoder.encode([canonicalize(i, cfg, 5000) for i in (a, b, c, d, e)])["vector"].astype(np.float64)
    labels = encoder.embedder.embed(["discount", "margin"]).astype(np.float64)
    return {
        "direction": float(v[0] @ v[1]),
        "cross_scope": float(v[2] @ v[3]),
        "entity": float(v[2] @ v[4]),
        "label_discount_margin": float(labels[0] @ labels[1]),
    }


def evaluate(name: str, overrides: dict, k: int) -> dict:
    """Contract, benchmark and cost figures for one embedder configuration."""
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        cuda = False
    cfg = load_config(**overrides)
    start = time.perf_counter()
    embedder = make_text_embedder(cfg)
    encoder = InsightEncoder(embedder, cfg)
    contract = contract_cosines(encoder)
    load_s = time.perf_counter() - start
    root = scratch_dir(f"compare_{name}", fresh=True)
    engine = build_engine(root / "ws", embedder=embedder, **overrides)  # reuses the loaded model
    record = ingest_quietly(engine, write_demo_csv(root))
    with contextlib.redirect_stdout(io.StringIO()):
        bench = run_experiment(engine, k=k)
    row = {
        **{f"cos_{key}": round(val, 3) for key, val in contract.items()},
        **{f"{m}_mrr": bench["summary"][m]["mrr"] for m in ("transversal", "vector_nn", "text_nn")},
        **{f"{m}_recall@{k}": bench["summary"][m]["recall"] for m in ("transversal", "vector_nn", "text_nn")},
        "cases": bench["cases"],
        "insights": record["metrics"]["validated_insights"],
        "attractors": record["metrics"]["attractors_total"],
        "load_s": round(load_s, 1),
        "embed_s": round(record["metrics"]["timings"]["embed_s"], 2),
        "peak_cuda_mb": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
        "model_id": encoder.spec.model_id,
        "dim": encoder.spec.dim,
    }
    row["contract_ok"] = row["cos_direction"] < 0 and row["cos_cross_scope"] > 0.7 and row["cos_entity"] < row["cos_cross_scope"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(CANDIDATES), help=f"comma-separated subset of {sorted(CANDIDATES)}")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    results = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"== {name}", flush=True)
        results[name] = evaluate(name, {"embedding_backend": "sentence-transformers", **CANDIDATES[name]}, args.k)
    keys = list(next(iter(results.values())))
    print(f"\n{'metric':<26}" + "".join(f"{n:>26}" for n in results))
    for key in keys:
        print(f"{key:<26}" + "".join(f"{str(results[n][key]):>26}" for n in results))
    print("saved:", save_result("compare_embedders", args.label, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
