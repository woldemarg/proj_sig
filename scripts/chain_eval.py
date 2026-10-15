"""Evidence-chain connectivity of the dual graph on the demo's planted sets (docs/10_verification.md §10.3).

    python scripts/chain_eval.py [--label NAME]

The query-free gate shared with the lac project's graph-construction record: the hop rule of its ``evalkit/chains.py``
(``hetero``, ``distances``) is copied verbatim (this repository imports nothing outside itself); the aggregates differ
from that record's: the null is exhaustive (every non-set pair, no sampling or degree matching), completeness is over
all sets, and there is no hub-free or component reading. Each planted mechanism with two or more patterns is a
set the graph should connect. A path leaves a pattern through its anchors (ACTIVATES, cost 0), moves between anchors
along RELATED_TO / CO_OCCURS (cost 1), arrives at a pattern from an anchor and may continue only along a lattice edge
(cost 1): a pattern never bridges two anchors for free. Layers: ``structural`` (lattice edges only), ``latent`` (anchors
and their links), ``union``. Per layer and hop limit h in 0..4: the share of the sets' pairs within h (all pairs, and the
scope-disjoint pairs the hypothesis benchmark calls analogues), the share of sets complete within h, the share of the
non-set pattern pairs within h (the null) with the lift over it, and the share of the patterns within h of a random
pattern (the ball). Results: ``.scratch/results/chain_eval_<label>.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: the packages live one level up

from experiment import phenomenon_of, planted_scopes  # noqa: E402

from insight_contracts.graph import LATENT_EDGES, STRUCTURAL_EDGES  # noqa: E402
from insight_graph_service.core.engine import Engine  # noqa: E402
from insight_graph_service.core.workspace import utc_now  # noqa: E402

HOPS = (0, 1, 2, 3, 4)


def hetero(
    attach: sp.csr_matrix, anchor_links: sp.csr_matrix | None = None, node_links: sp.csr_matrix | None = None, *, semantic: bool = True
) -> sp.csr_matrix:
    """A directed graph over (content out, content in, anchors) = (n, n, A) nodes. A path leaves a content node through
    its anchors, moves between anchors along semantic edges (cost 1), arrives at a content node from an anchor and may
    continue only along a content–content semantic edge (cost 1): a pattern never bridges two anchors for free (that is
    what a CO_OCCURS edge states, at cost 1). Attachment costs 0 (``semantic``) or 1 (full path length)."""
    n, a = attach.shape
    out, inn, anc = 0, n, 2 * n
    att = attach.tocoo()
    r, c = att.row[att.data != 0], att.col[att.data != 0]  # stored zeros are no attachment
    w = 0.0 if semantic else 1.0
    rows, cols, data = [out + r, anc + c], [anc + c, inn + r], [np.full(len(r), w)] * 2
    if anchor_links is not None:
        upper = sp.triu(anchor_links + anchor_links.T, 1).tocoo()
        keep = upper.data != 0
        rows += [anc + upper.row[keep], anc + upper.col[keep]]
        cols += [anc + upper.col[keep], anc + upper.row[keep]]
        data += [np.ones(int(keep.sum()))] * 2
    if node_links is not None:
        link = (node_links + node_links.T).tocoo()  # both orientations, from either copy of the source to the target's out copy
        keep = link.data != 0
        for origin in (out, inn):
            rows.append(origin + link.row[keep])
            cols.append(out + link.col[keep])
            data.append(np.ones(int(keep.sum())))
    graph = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n + a, 2 * n + a))
    return graph.tocsr()  # explicit zeros stay: scipy's csgraph reads them as zero-weight edges


def distances(graph: sp.csr_matrix, n: int) -> np.ndarray:
    """(n x n) shortest distances from content nodes to content nodes, inf beyond the last hop limit (inclusive)."""
    d = dijkstra(graph, directed=True, indices=np.arange(n), limit=HOPS[-1])
    return np.minimum(d[:, :n], d[:, n : 2 * n])


def layer_measures(d: np.ndarray, sets: list[list[int]], disjoint: list[tuple[int, int]]) -> dict[str, float]:
    n = d.shape[0]
    pairs = [(a, b) for s in sets for i, a in enumerate(s) for b in s[i + 1 :]]
    in_set = set(pairs) | {(b, a) for a, b in pairs}
    null = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in in_set]
    out: dict[str, float] = {}
    for h in HOPS:
        reach = np.mean([d[a, b] <= h for a, b in pairs])
        null_share = float(np.mean([d[a, b] <= h for a, b in null])) if null else float("nan")
        out[f"pairs@{h}"] = float(reach)
        out[f"disjoint@{h}"] = float(np.mean([d[a, b] <= h for a, b in disjoint])) if disjoint else float("nan")
        out[f"complete@{h}"] = float(np.mean([all(d[a, b] <= h for i, a in enumerate(s) for b in s[i + 1 :]) for s in sets]))
        out[f"null@{h}"] = null_share
        out[f"lift@{h}"] = float(reach / null_share) if null_share else float("nan")
        out[f"ball@{h}"] = float(np.median((d <= h).sum(axis=1) / n))
    return out


def run_chain_eval(engine: Engine) -> dict[str, Any]:
    """The measures per layer on the committed graph's planted sets."""
    g = engine.committed().graph
    patterns = [node["id"] for node in g.of_kind("Pattern")]
    index = {p: i for i, p in enumerate(patterns)}
    anchors = sorted({e["target"] for e in g.edges if e["type"] == "ACTIVATES"})
    aidx = {a: i for i, a in enumerate(anchors)}
    n, a = len(patterns), len(anchors)
    structural, latent = {t.value for t in STRUCTURAL_EDGES}, {t.value for t in LATENT_EDGES}
    att = [(index[e["source"]], aidx[e["target"]]) for e in g.edges if e["type"] == "ACTIVATES" and not e.get("props", {}).get("weak")]
    links = [(aidx[e["source"]], aidx[e["target"]]) for e in g.edges if e["type"] in latent and e["source"] in aidx and e["target"] in aidx]
    lattice = [(index[e["source"]], index[e["target"]]) for e in g.edges if e["type"] in structural and e["source"] in index and e["target"] in index]

    def matrix(edges: list[tuple[int, int]], shape: tuple[int, int]) -> sp.csr_matrix:
        return sp.csr_matrix((np.ones(len(edges)), ([s for s, _ in edges], [t for _, t in edges])), shape=shape)

    attach, anchor_links, node_links = matrix(att, (n, a)), matrix(links, (a, a)), matrix(lattice, (n, n))
    empty_a = sp.csr_matrix((n, a))
    truth = planted_scopes()
    label = {p: phenomenon_of(g.insight(p), truth) for p in patterns}
    conds = {p: {c.expr for c in g.insight(p).conditions} for p in patterns}
    sets = [[index[p] for p in patterns if label[p] == m] for m in sorted(truth) if sum(label[p] == m for p in patterns) >= 2]
    disjoint = [(a_, b_) for s in sets for i, a_ in enumerate(s) for b_ in s[i + 1 :] if not (conds[patterns[a_]] & conds[patterns[b_]])]
    layers = {
        "structural": layer_measures(distances(hetero(empty_a, None, node_links), n), sets, disjoint),
        "latent": layer_measures(distances(hetero(attach, anchor_links, None), n), sets, disjoint),
        "union": layer_measures(distances(hetero(attach, anchor_links, node_links), n), sets, disjoint),
    }
    return {
        "at": utc_now(),
        "patterns": n,
        "anchors": a,
        "sets": len(sets),
        "set_sizes": [len(s) for s in sets],
        "disjoint_pairs": len(disjoint),
        "layers": layers,
    }


def format_table(result: dict[str, Any]) -> str:
    lines = [
        f"evidence chains — {result['sets']} planted sets {result['set_sizes']} over {result['patterns']} patterns, {result['anchors']} anchors; "
        f"{result['disjoint_pairs']} scope-disjoint pairs",
        f"{'layer':<11} {'pairs@1':>8} {'pairs@2':>8} {'disjoint@2':>11} {'complete@2':>11} {'null@2':>7} {'lift@2':>7} {'ball@2':>7}",
    ]
    for name, m in result["layers"].items():
        lines.append(
            f"{name:<11} {m['pairs@1']:>8.3f} {m['pairs@2']:>8.3f} {m['disjoint@2']:>11.3f} {m['complete@2']:>11.3f} {m['null@2']:>7.3f} {m['lift@2']:>7.2f} {m['ball@2']:>7.3f}"
        )
    return "\n".join(lines)


def main() -> int:
    """The chain eval on a scratch workspace holding the synthetic demo; the result is saved under .scratch/results."""
    import argparse

    from scratch import demo_engine, save_result

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    result = run_chain_eval(demo_engine("chain_eval"))
    print(format_table(result))
    print(save_result("chain_eval", args.label, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
