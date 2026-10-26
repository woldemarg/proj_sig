"""Time windows in the evidence (docs/07_question_answering.md §7.4): a window's insight carries its time line — its
window, its trajectory's state, episode and windows, the last transition, the latest admitted window and the peak — and a
window's dataset names its window (``evidence`` adds them, and the NOTE). The seeds and the walk keep one card per
trajectory (``seeds.representatives``)."""

from __future__ import annotations

from typing import Any

from graph_query_engine.graph import DualGraph
from graph_query_engine.seeds import trajectory
from insight_contracts import dataset_node_id


def when(iso: str) -> str:
    """A window bound as read: the date at midnight, else the date and the time (a window may be as short as a second)."""
    return iso[:10] if iso[11:19] == "00:00:00" else iso[:19].replace("T", " ")


def span(window: dict[str, Any]) -> str:
    """``W<index> [t_start, t_end)``."""
    return f"W{window['index']} [{when(window['t_start'])}, {when(window['t_end'])})"


def time_facts(graph: DualGraph, pid: str) -> dict[str, Any]:
    """What a window's insight says of time: its window, its trajectory's state (the closing reason once closed), episode,
    windows active out of those followed, the transition into its latest admitted window, that window and the peak (the
    admitted window with the largest effect); a window whose node fails its digest is not counted admitted (no answer
    shows it). Empty for an insight of no time window."""
    key = trajectory(graph, pid)
    window = graph.nodes.get(dataset_node_id(graph.insight(pid).dataset_id), {"props": {}})["props"].get("window")
    t = graph.trajectories.get(key or "")
    if not (window and t):
        return {}
    active = [w for w in t["windows"] if w["active"] and not graph.tampered(w["pattern_id"])]
    latest, peak = active[-1], max(active, key=lambda w: abs(w["effect_size"] or 0.0))
    into = [
        e["props"]["transition"] for e in graph.inc.get(latest["pattern_id"], []) if e["type"] == "EVOLVES_TO" and e["props"]["trajectory"] == key
    ]
    return {
        "window": window,
        "trajectory": key,
        "state": t["closed_reason"] or t["state"],
        "episode": t["episode"],
        "active": len(active),
        "followed": len(t["windows"]),
        "transition": into[0] if into else None,
        "latest": latest["index"],
        "peak": {"index": peak["index"], "effect_size": peak["effect_size"]},
    }


def time_line(facts: dict[str, Any]) -> str:
    """The ``time:`` line of an evidence item."""
    move = f"last transition: {facts['transition']}; " if facts["transition"] else ""
    peak, effect = facts["peak"], facts["peak"]["effect_size"]
    return (
        f"window {span(facts['window'])} | trajectory {facts['state']} (episode {facts['episode']}, active in {facts['active']}/"
        f"{facts['followed']} windows; {move}latest: W{facts['latest']}, peak: W{peak['index']}{f' {effect:+.2f} sd' if effect is not None else ''})"
    )
