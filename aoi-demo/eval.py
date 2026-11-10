# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi", "httpx", "pandas", "shapely", "jinja2", "h3>=4,<5"]
# ///
"""The hand-off evaluation (issue #335): every scenario of app.SCENARIOS on a fixed set of questions over one fixed
zone of the demo data, through the running demo's HTTP API. Raw runs go to .scratch/eval/ (not committed).

    uv run eval.py run --scenario A B --runs 3     one JSON per scenario x question x run; existing files are kept
    uv run eval.py facts                           the zone's reference figures, computed with pandas
    uv run eval.py judge                           blind LLM scores and the recompute check, written into each run
    uv run eval.py report                          the scenario x question table (markdown)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / ".scratch" / "eval"
DEMO = "http://127.0.0.1:8090"
LLM = "http://127.0.0.1:8080"  # ADS's gateway: the judge runs on the route the final answer uses
COUNT_VALUES = ["theft"]

# The Loop, River North and the Near North Side: the densest bars and thefts of the demo data
ZONE = {"type": "Polygon", "coordinates": [[[-87.650, 41.874], [-87.616, 41.874], [-87.616, 41.912], [-87.650, 41.912], [-87.650, 41.874]]]}

QUESTIONS = {
    "time": "Як змінювалась кількість злочинів у цій зоні протягом року: які місяці й дні тижня найактивніші?",
    "space": "Де в цій зоні зосереджені крадіжки (theft) і чим ці місця відрізняються від решти зони?",
    "compare": "Чим структура типів злочинів у зоні відрізняється від території поруч і від решти міста?",
    "bars": "Чи пов'язана кількість крадіжок із барами в цій зоні?",
    "open": "Що найважливіше варто знати про злочинність у цій зоні?",
    "nodata": "Як рівень доходів мешканців впливає на злочинність у цій зоні?",  # no such column: the answer is "no data"
}

# Fixed before the first run. Every axis 0-3; the judge sees neither the scenario nor the other answers.
RUBRIC = """Score the answer on four axes, each an integer 0-3.

correctness: figures and statements agree with the data and with each other.
  A figure is right when it matches <Reference_Facts> or <Computed_Outputs> (what the analysis code printed), rounding
  aside. Another valid definition of the same thing (Pearson instead of Spearman, a mean instead of a median, cells of
  the whole map instead of the zone) is not a mismatch. A figure found in neither is unverified, not wrong.
  Wrong is: a figure that contradicts both, a figure attached to the wrong thing or unit (a share told as a percent
  of a percent, a map-wide number told as the zone's), a conclusion the reference contradicts.
  3 = nothing wrong, no internal contradiction; 2 = one minor error or several unverified figures;
  1 = several errors or one that changes a conclusion; 0 = the main conclusion is wrong.
  When the question asks for something the data do not hold, 3 requires saying so plainly.
insights_used: the answer uses the established insights that bear on the question and the zone.
  3 = the relevant ones are woven into the answer with their numbers; 2 = used but generic or without numbers;
  1 = only mentioned in passing or irrelevant ones listed; 0 = none used.
no_contradiction: the answer does not contradict the established insights or misstate them.
  3 = no conflict; 2 = an insight is restated loosely; 1 = an insight is misread (wrong direction, wrong place, a
  finding outside the zone told as the zone's); 0 = the answer denies or reverses an insight without saying so.
usefulness: specific and useful to the person who asked.
  3 = answers the question directly with concrete figures, places or periods and a clear takeaway;
  2 = answers it but buries or pads the point; 1 = mostly generic or off the question; 0 = does not answer."""

JUDGE = """You grade an analytical answer. Be strict and literal; do not reward length.

{rubric}

<Question>
{question}
</Question>

<Reference_Facts>
Computed directly from the tables for the zone (ground truth for the figures they cover):
{facts}
</Reference_Facts>

<Computed_Outputs>
What the analysis steps behind this answer printed (truncated):
{outputs}
</Computed_Outputs>

<Established_Insights>
Validated findings of a statistical insight miner about this dataset (ground truth):
{insights}
</Established_Insights>

<Answer>
{answer}
</Answer>

Reply with JSON only:
{{"correctness": n, "insights_used": n, "no_contradiction": n, "usefulness": n, "issues": ["short, concrete; at most 4"]}}"""

# Fixed before the first run: what counts as ADS recomputing an insight
RECOMPUTE = """An agent was given established insights about a dataset and then ran analysis steps. Decide which steps
recompute or verify an insight.

A step counts when its stated goal or its code tests, confirms, refutes or reproduces an insight's own claim: it
rebuilds the insight's cell condition from the data (neighbour bands, LISA classes, per-cell bands of a context layer),
or recomputes the insight's per-cell metric for that subgroup against the rest or the whole map to see whether the
stated shift holds. A step does not count when it only uses an insight's event type, place or condition to compute
something else the question asks (a time series, a count in the zone, a breakdown), or when it merely quotes the insight.
Read literally: comparing the zone with its surroundings, or cells with and without context points, to answer a question
about exactly that is the question's own work, not a recomputation, unless the step re-measures the insight's stated
metric on the insight's own subgroup (its bands, its LISA class, its sig_o column).

<Established_Insights>
{insights}
</Established_Insights>

<Steps>
{steps}
</Steps>

Reply with JSON only: {{"recomputed": [{{"step": n, "insight": "O#", "why": "one short sentence"}}]}} (an empty list when none)."""

http = httpx.Client(timeout=httpx.Timeout(900.0, connect=10.0))


def state() -> dict[str, Any]:
    return http.get(f"{DEMO}/state").json()


def wait(done: Any, every: float = 3, limit: float = 1800) -> dict[str, Any]:
    deadline = time.time() + limit
    while True:
        s = state()
        if done(s):
            return s
        if time.time() > deadline:
            raise TimeoutError("the demo did not finish in time")
        time.sleep(every)


def prepare() -> dict[str, Any]:
    """The demo's datasets and the fixed zone, built once per demo process (its state is in memory)."""
    s = state()
    if not s["datasets"] or s["datasets"]["stage"] != "ready":
        http.post(f"{DEMO}/datasets", json={"count_values": COUNT_VALUES})
        s = wait(lambda s: s["datasets"]["stage"] in ("ready", "error"))
        assert s["datasets"]["stage"] == "ready", s["datasets"]
    if not s["zone"] or s["zone"]["polygon"] != ZONE:
        r = http.post(f"{DEMO}/zone", json={"polygon": ZONE})
        r.raise_for_status()
    return wait(lambda s: s["zone"]["sig"]["stage"] != "mining")


def llm_calls(a: dict[str, Any]) -> dict[str, int]:
    trace = a.get("trace") or {}
    ads = (trace.get("ads") or {}).get("counters") or {}
    return {
        "sig": int(bool((trace.get("search") or {}).get("llm"))),
        "ads": int(ads.get("total_llm_calls") or 0),
        "joint": int("joint" in trace),
    }


def tokens(a: dict[str, Any]) -> dict[str, int]:
    trace = a.get("trace") or {}
    ads = (trace.get("ads") or {}).get("counters") or {}
    use = lambda u: int((u or {}).get("total_tokens") or 0)  # noqa: E731
    return {
        "sig": use(((trace.get("search") or {}).get("llm") or {}).get("usage")),
        "ads": int(ads.get("total_prompt_tokens") or 0) + int(ads.get("total_completion_tokens") or 0),
        "joint": use((trace.get("joint") or {}).get("usage")),
    }


def run(scenarios: list[str], questions: list[str], runs: int) -> None:
    prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    for n in range(1, runs + 1):  # a whole round per run number: a partial matrix still compares every scenario
        for qid in questions:
            for sc in scenarios:
                path = OUT / f"{sc}_{qid}_{n}.json"
                if path.exists():
                    continue
                t0 = time.time()
                wait(lambda s: not s["ask"] or s["ask"]["stage"] in ("done", "error"))  # an ask a stopped run left behind
                r = http.post(f"{DEMO}/ask", json={"question": QUESTIONS[qid], "scenario": sc})
                r.raise_for_status()
                a = wait(lambda s: s["ask"]["stage"] in ("done", "error"))["ask"]
                rec = {
                    "scenario": sc,
                    "qid": qid,
                    "run": n,
                    "question": QUESTIONS[qid],
                    "stage": a["stage"],
                    "error": a.get("error"),
                    "failed_stage": a.get("failed_stage"),
                    "seconds": round(time.time() - t0, 1),
                    "timings": a.get("timings"),
                    "llm_calls": llm_calls(a),
                    "tokens": tokens(a),
                    "sig_entry": a.get("sig_entry"),
                    "observations": a.get("observations", ""),
                    "ads_query": a.get("ads_query", ""),
                    "ads_report": a.get("ads_report", ""),
                    "final": a.get("joint_report", ""),
                    "final_raw": ((a.get("trace") or {}).get("joint") or {}).get("raw", ""),  # an answer the gate refused
                    "joint_error": a.get("joint_error"),
                    "ads_result": ((a.get("trace") or {}).get("ads") or {}).get("result"),
                    "ads_trace": http.get(f"{DEMO}/trace/ads").json() if a.get("ads_report") else {},
                }
                path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{path.name}: {rec['stage']} {rec['seconds']} s, calls {rec['llm_calls']}, {rec['error'] or ''}", flush=True)


def facts() -> dict[str, Any]:
    """The zone's reference figures, from the same rule the demo applies (app.flag_in_aoi, geo.aoi_zone)."""
    import h3
    from app import BASE, CONTEXT, DEMO_DIR, flag_in_aoi
    from geo import aoi_zone, enrich_h3, zone_h3_cells

    res = state()["datasets"]["resolution"]
    frames = {}
    for name, spec in (("base", BASE), ("context", CONTEXT)):
        f = enrich_h3(pd.read_csv(DEMO_DIR / spec["file"]), res)
        f["in_aoi"] = flag_in_aoi(f, ZONE)
        frames[name] = f
    col = f"h3_r{res}"
    cells = zone_h3_cells(ZONE, res, sorted(set(frames["base"][col]) - {""}))
    for f in frames.values():
        f["aoi_zone"] = aoi_zone(f, cells, col)
    b, c = frames["base"], frames["context"]
    inside = b[b["in_aoi"] == 1].assign(day=lambda d: pd.to_datetime(d["crime_date"]))
    daily = inside.groupby(inside["day"].dt.date).size()
    shares = {z: b[b["aoi_zone"] == z]["crime"].value_counts(normalize=True).round(3).to_dict() for z in ("inside", "adjacent", "far")}
    per_cell = pd.DataFrame({"events": inside.groupby(col).size(), "theft": inside[inside["crime"] == "theft"].groupby(col).size()})
    per_cell["bars"] = c[c["in_aoi"] == 1].groupby(col).size()
    per_cell = per_cell.fillna(0)
    top = per_cell.sort_values("theft", ascending=False).head(5)
    with_bars, without = per_cell[per_cell["bars"] > 0], per_cell[per_cell["bars"] == 0]
    by_zone = {}
    for z in ("inside", "adjacent", "far"):  # cells as ADS usually builds them: grouped by cell and the rows' class
        ev = b[b["aoi_zone"] == z]
        pc = pd.DataFrame({"events": ev.groupby(col).size(), "theft": ev[ev["crime"] == "theft"].groupby(col).size()})
        pc["bars"] = c.groupby(col).size()
        pc = pc.fillna(0)
        th = pc[pc["theft"] > 0]
        by_zone[z] = {
            "cells_with_events": len(pc),
            "cells_with_theft": len(th),
            "mean_events_per_cell": round(float(pc["events"].mean()), 2),
            "mean_theft_per_cell": round(float(pc["theft"].mean()), 2),
            "mean_theft_per_theft_cell": round(float(th["theft"].mean()), 2),
            "mean_bars_per_cell": round(float(pc["bars"].mean()), 2),
            "mean_bars_per_theft_cell": round(float(th["bars"].mean()), 2),
            "pearson_bars_theft(theft cells)": round(float(th["bars"].corr(th["theft"])), 3),
        }
    out = {
        "columns_base": list(pd.read_csv(DEMO_DIR / BASE["file"], nrows=0).columns),
        "columns_context": list(pd.read_csv(DEMO_DIR / CONTEXT["file"], nrows=0).columns),
        "zone_cells": len(cells),
        "base_rows": b["aoi_zone"].value_counts().to_dict(),
        "context_rows(bars)": c["aoi_zone"].value_counts().to_dict(),
        "period": [str(daily.index.min()), str(daily.index.max())],
        "daily_inside": {"mean": round(daily.mean(), 1), "median": float(daily.median()), "min": int(daily.min()), "max": int(daily.max())},
        "busiest_days_inside": {str(k): int(v) for k, v in daily.sort_values(ascending=False).head(5).items()},
        "monthly_inside": {int(k): int(v) for k, v in inside.groupby(inside["day"].dt.month).size().items()},
        "weekday_inside(Mon=0)": {int(k): int(v) for k, v in inside.groupby(inside["day"].dt.day_of_week).size().items()},
        "hour_band_inside": inside["hour_band"].value_counts().to_dict(),
        "crime_type_shares": shares,
        "crime_type_counts_inside": inside["crime"].value_counts().to_dict(),
        "theft_per_cell_inside": {
            "cells_with_events": len(per_cell),
            "median": float(per_cell["theft"].median()),
            "max": int(per_cell["theft"].max()),
        },
        "top_theft_cells_inside": {
            k: {
                "theft": int(v["theft"]),
                "events": int(v["events"]),
                "bars": int(v["bars"]),
                "centre_lat_lon": [round(x, 4) for x in h3.cell_to_latlng(k)],
            }
            for k, v in top.iterrows()
        },
        "theft_share_in_top5_cells": round(float(top["theft"].sum() / per_cell["theft"].sum()), 3),
        "bars_vs_theft_inside": {
            "cells_with_bars": len(with_bars),
            "cells_without_bars": len(without),
            "mean_theft_per_cell_with_bars": round(float(with_bars["theft"].mean()), 2),
            "mean_theft_per_cell_without_bars": round(float(without["theft"].mean()), 2),
            "median_theft_with_bars": float(with_bars["theft"].median()),
            "median_theft_without_bars": float(without["theft"].median()),
            "spearman_bars_theft_per_cell": round(float(per_cell["bars"].rank().corr(per_cell["theft"].rank())), 3),
            "spearman_bars_events_per_cell": round(float(per_cell["bars"].rank().corr(per_cell["events"].rank())), 3),
            "pearson_bars_theft_per_cell": round(float(per_cell["bars"].corr(per_cell["theft"])), 3),
        },
        "per_cell_by_aoi_zone(cells of all rows, by the rows' class)": by_zone,
        "note": "per-cell figures depend on which cells are counted (cells with any event, with a theft, with a bar); a figure "
        "within about 15% of one above, under another such choice, is consistent",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "facts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def ask_json(prompt: str) -> dict[str, Any]:
    body = {"model": "synthesis", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 4000}
    for _ in range(3):
        r = http.post(f"{LLM}/v1/chat/completions", json=body)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(text[text.index("{") : text.rindex("}") + 1])
        except ValueError:
            continue
    raise RuntimeError(f"the judge returned no JSON: {text[:300]}")


NUMBER = re.compile(r"(?<![\w.])\d[\d  ]*(?:[.,]\d+)?")


def numbers(text: str) -> set[str]:
    """The figures of a text, to three significant digits, so 0.5105123 and 0.51 meet and 1 358 and 1358 meet."""
    out = set()
    for m in NUMBER.findall(text):
        v = float(re.sub(r"[  ]", "", m).replace(",", "."))
        out |= {f"{v:.3g}", f"{v:.2g}", f"{v * 100:.3g}", f"{v * 100:.2g}", f"{v / 100:.3g}"} if v else set()
    return out


def grounding(rec: dict[str, Any]) -> float | None:
    """The share of the final answer's figures found in what stood behind it: the ADS report, the steps' output, SIG's
    findings and the task. A figure found nowhere is one the final call made up or recomputed."""
    answer = rec["final"] or rec.get("final_raw") or ""
    mine = {f"{float(re.sub(r'[  ]', '', m).replace(',', '.')):.3g}" for m in NUMBER.findall(answer)}
    mine = {n for n in mine if float(n) > 12 or "." in n}  # small integers are list numbers, months, O# ids
    steps = " ".join(str(s.get("stdout_tail") or "") + json.dumps(s.get("observation") or {}) for s in rec["ads_trace"].get("steps") or [])
    known = numbers(" ".join([rec["ads_report"], steps, rec["observations"], rec["ads_query"]]))
    return round(len(mine & known) / len(mine), 2) if mine else None


VERIFY = re.compile(
    r"\b(verif\w*|validat\w*|confirm\w*|re-?comput\w*|reproduc\w*|test(?:ing)? (?:the |whether |if )?(?:sig|insight|hypothes|claim|o\d))", re.I
)


def steps_text(trace: dict[str, Any]) -> str:
    out = []
    for n, s in enumerate(trace.get("steps") or [], 1):
        goal, code = s.get("step_summary") or "", str(s.get("code") or "")[:2500]
        out.append(f"Step {n} [{s.get('tool_name') or '?'}] {goal}\n{code}")
    return "\n\n".join(out) or "(no steps)"


def outputs_text(trace: dict[str, Any]) -> str:
    return "\n".join(f"Step {n}: {str(s.get('stdout_tail') or '')[:1500]}" for n, s in enumerate(trace.get("steps") or [], 1)) or "(none)"


def judge() -> None:
    """Blind scores of the final answer (and of the ADS report alone where the final is built from it), in a shuffled
    order, and the recompute check over the ADS steps."""
    ref = json.dumps(json.loads((OUT / "facts.json").read_text(encoding="utf-8")), ensure_ascii=False)
    paths = [p for p in sorted(OUT.glob("?_*_*.json"))]
    random.Random(335).shuffle(paths)
    for p in paths:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if "judge" in rec or rec["stage"] != "done":
            continue
        insights = rec["observations"] or "none"
        grade = lambda answer: ask_json(  # noqa: E731, B023
            JUDGE.format(rubric=RUBRIC, question=rec["question"], facts=ref, outputs=outputs_text(rec["ads_trace"]), insights=insights, answer=answer)
        )  # noqa: E731, B023
        answer = rec["final"] or rec.get("final_raw")  # a refused answer is still graded; the report counts the refusal
        rec["judge"] = grade(answer) if answer else {"error": rec.get("joint_error")}
        if rec["scenario"] in "BC":  # the sub-variant "the ADS report as is, no final call"
            rec["judge_ads_report"] = grade(rec["ads_report"])
        rec["grounding"] = grounding(rec)
        rec["verify_words"] = sorted(
            {m.group(0).lower() for s in rec["ads_trace"].get("steps") or [] for m in VERIFY.finditer(s.get("step_summary") or "")}
        )
        rec["recompute"] = ask_json(RECOMPUTE.format(insights=insights, steps=steps_text(rec["ads_trace"])))["recomputed"]
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(p.name, rec["judge"], len(rec["recompute"]), flush=True)


AXES = ("correctness", "insights_used", "no_contradiction", "usefulness")


def report() -> str:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("?_*_*.json"))]
    med = lambda xs: round(statistics.median(xs), 1) if xs else None  # noqa: E731
    spread = lambda xs: f"{med(xs)} ({min(xs)}–{max(xs)})" if xs else "—"  # noqa: E731
    lines = [
        "| Сценарій | Запитання | n | збої + відмови гейта | якість Σ/12 | правильність | інсайти | без суперечностей | корисність | числа з джерел | перерахунок (суддя), запусків | слова перевірки в кроках, запусків | кроків ADS | час, с | SIG / ADS / фінал, с | виклики LLM | токени |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sc in sorted({r["scenario"] for r in recs}):
        for qid in [*QUESTIONS, "усі"]:
            rs = [r for r in recs if r["scenario"] == sc and (qid == "усі" or r["qid"] == qid)]
            ok = [r for r in rs if r["stage"] == "done" and "correctness" in r.get("judge", {})]
            if not rs:
                continue
            total = [sum(r["judge"][a] for a in AXES) for r in ok]
            stage = lambda k: med([r["timings"].get(k, 0) for r in ok])  # noqa: E731, B023
            lines.append(
                f"| {sc} | {qid} | {len(rs)} | {len(rs) - len(ok)}+{sum(not r['final'] for r in ok)} | {spread(total)} | "
                + " | ".join(str(med([r["judge"][a] for r in ok])) for a in AXES)
                + f" | {med([r['grounding'] for r in ok if r['grounding'] is not None])} | {sum(bool(r.get('recompute')) for r in ok)}/{len(ok)} | "
                f"{sum(bool(r.get('verify_words')) for r in ok)}/{len(ok)} | {med([len(r['ads_trace'].get('steps') or []) for r in ok])} | {spread([r['seconds'] for r in ok])} | "
                f"{stage('search')} / {stage('ads')} / {stage('joint')} | {med([sum(r['llm_calls'].values()) for r in ok])} | "
                f"{med([sum(r['tokens'].values()) for r in ok])} |"
            )
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "facts", "judge", "report"])
    ap.add_argument("--scenario", nargs="+", default=list("ABC"))
    ap.add_argument("--questions", nargs="+", default=list(QUESTIONS))
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.scenario, args.questions, args.runs)
    elif args.cmd == "facts":
        print(json.dumps(facts(), ensure_ascii=False, indent=1))
    elif args.cmd == "judge":
        judge()
    else:
        print(report())
