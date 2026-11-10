# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi", "uvicorn", "httpx", "pandas", "shapely", "jinja2", "h3>=4,<5"]
# ///
"""AOI demo: polygon on a map -> SIG insights in the zone + ADS analysis of the in_aoi tables -> one joint report
rendered through Copilot's final_response.jinja. One process, one user, state in memory (see README.md).

Run:  uv run app.py          the server on http://127.0.0.1:8090
      uv run app.py check    the in_aoi self-check
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import h3
import httpx
import pandas as pd
import shapely
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from geo import H3_LEVELS, aoi_zone, enrich_h3, h3_columns, zone_h3_cells
from jinja2 import Template
from shapely.geometry import shape

log = logging.getLogger("aoi")
HERE = Path(__file__).resolve().parent

SIG_URL = os.environ.get("SIG_URL", "http://127.0.0.1:8765")  # SIG's graph service (or a console edge, which routes both)
SIG_CHAT_URL = os.environ.get("SIG_CHAT_URL", SIG_URL)  # SIG's narrator: /api/chat/query
ADS_URL = os.environ.get("ADS_URL", "http://127.0.0.1:8000")
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080")
DEMO_DIR = Path(os.environ.get("DEMO_DIR", HERE.parent / "data" / "demo"))  # the repo's shipped datasets (data/demo/SOURCES.md)
# Copilot's final_response.jinja adapted for a joint ADS + SIG report; its header names the source and the changes
JOINT_TEMPLATE = Path(os.environ.get("JOINT_TEMPLATE", HERE / "prompts/joint_response.jinja"))

# the happy path is Chicago: the shipped SIG datasets, with the options data/demo/datasets.json reads them with
BASE = {
    "file": "chicago_crimes_2024.csv",
    "label": "злочини 2024",
    "geo": "lat,lon@auto",
    "time": "crime_date",
    "source": "City of Chicago, Crimes",
    "category": "crime",
}
CONTEXT = {"file": "chicago_bars.csv", "label": "бари", "geo": "lat,lon", "layer": "true"}
RING = 0  # the layer counted in the cell alone: SIG's default since 2026-11-05; an older image defaults to 1 and the bars admit nothing
# BASE["category"]: the column whose values the page offers to count per cell (SIG's count_values -> count_<column>_<value>,
# the counters a cross run with the layer reads); SIG counts a value with at least COUNT_MIN located rows and marks a
# rarer one `sparse` without a counter (insight_graph_service/core/uploads.py)
COUNT_MIN = 250
COUNT_DEFAULT = ("theft", "assault")  # the demo's preselected types
MAX_INSIGHTS = 10  # findings drawn on the map, the zone's own first: SIG's evidence holds about ten
SAMPLE_POINTS = 3000
# How SIG's knowledge reaches ADS (issue #335): SIG's narrator LLM, what the ADS task carries of SIG, SIG in the final prompt
SCENARIOS = {
    "A": {"sig_llm": True, "task": "answer", "final_sig": True},  # SIG's answer and O# lines; hypotheses as claims to test
    "B": {"sig_llm": False, "task": "facts", "final_sig": False},  # the insights alone, as settled facts; the answer from ADS alone
    "C": {"sig_llm": False, "task": "columns", "final_sig": False},  # the insights as sig_o# flag columns of the tables
    "D": {"sig_llm": True, "task": "none", "final_sig": True},  # ADS without SIG; SIG joins in the final prompt
    "E": {"sig_llm": False, "task": "plan", "final_sig": False},  # the insights turned into an analysis plan for ADS
}
HANDOFF = os.environ.get("HANDOFF", "A").upper()

http = httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))

# ponytail: single-user in-memory state; a second browser tab shares it, per-session state when a second user needs their
# own zone. DATA holds the frames as the build left them; STATE what the page polls (JSON, except the zone's "files": the
# tables' bytes taken when the polygon closed, which the ask uploads and /state leaves out)
STATE: dict[str, Any] = {"datasets": None, "zone": None, "ask": None}
DATA: dict[str, pd.DataFrame] = {}
GRID: dict[str, Any] = {}  # the base layer's map cells in SIG: h3 ids, rings, centres, and SIG's resolution
PATTERN_CELLS: dict[str, list[str]] = {}  # pattern id -> its h3 cells, kept out of STATE (the page draws the rings)


@lru_cache(maxsize=1)
def category_options() -> list[dict[str, Any]]:
    """The base file's category values with their located rows (valid coordinates): the ones SIG would count per cell
    and the rarer ones it marks `sparse` and leaves uncounted."""
    df = pd.read_csv(DEMO_DIR / BASE["file"], usecols=[BASE["category"], "lat", "lon"])
    located = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    counts = located[BASE["category"]].astype(str).value_counts()
    return [{"value": v, "rows": int(n), "status": "ok" if n >= COUNT_MIN else "sparse"} for v, n in counts.items()]


def base_spec(count_values: list[str]) -> dict[str, str]:
    """The base layer's SIG upload options: the manifest's, plus count_values=<column>=v1,v2 for the chosen types."""
    spec = {k: v for k, v in BASE.items() if k != "category"}
    if count_values:
        spec["count_values"] = ",".join(f"{BASE['category']}={v}" for v in count_values)  # one column=value per type
    return spec


# ----------------------------------------------------------------------------- SIG


def sig_upload(spec: dict[str, str]) -> dict[str, Any]:
    """A shipped layer as a file, with the options the SIG manifest reads it with (data/demo/datasets.json):
    the finest level's record + levels."""
    options = {k: v for k, v in spec.items() if k not in ("file", "label")}
    r = http.post(f"{SIG_URL}/api/upload", files={"file": (spec["file"], (DEMO_DIR / spec["file"]).read_bytes())}, data=options)
    if r.status_code >= 300:
        raise RuntimeError(f"SIG upload {spec['file']}: {r.status_code} {r.text[:300]}")
    return r.json()


NO_FINDINGS = ("no_viable_insights", "no_candidates")  # a run SIG read to the end that admitted nothing: an answer, not a fault


def sig_wait(batch_ids: list[str], limit: float = 3600, empty_ok: bool = False) -> list[dict[str, Any]]:
    deadline = time.time() + limit
    while True:
        records = [http.get(f"{SIG_URL}/api/batches/{b}").json() for b in batch_ids]
        status = {r["batch_id"]: r.get("status") for r in records}
        failed = [r for r in records if r.get("status") == "FAILED" and not (empty_ok and (r.get("error") or {}).get("code") in NO_FINDINGS)]
        if failed:
            raise RuntimeError(f"SIG batch {failed[0]['batch_id']} FAILED: {failed[0].get('error')}")
        if all(s in ("READY", "SKIPPED", "LAYER", "FAILED") for s in status.values()):  # LAYER: a context layer, never mined
            return records
        if time.time() > deadline:
            raise TimeoutError(f"SIG batches not ready: {status}")
        time.sleep(3)


def sig_crosstype(target: str, layer: str) -> list[dict[str, Any]]:
    """One run per level of the target with the layer as its context; SIG skips a repeat of the same pair itself."""
    r = http.post(f"{SIG_URL}/api/crosstype", json={"target": target, "layers": [layer], "ring": RING})
    if r.status_code >= 300:
        raise RuntimeError(f"SIG crosstype: {r.status_code} {r.text[:300]}")
    data = r.json()
    return data if isinstance(data, list) else [data]


def sig_resolution(dataset_id: str) -> int:
    """The H3 resolution SIG chose for a geo dataset (GET /api/geo lists every geo dataset with its resolution): the one
    the demo's h3_r* columns are built at, never chosen here."""
    r = http.get(f"{SIG_URL}/api/geo")
    if r.status_code >= 300:
        raise RuntimeError(f"SIG geo: {r.status_code} {r.text[:300]}")
    return int(next(g["resolution"] for g in r.json() if g.get("dataset_id") == dataset_id))


def cell_grid(pack: dict[str, Any]) -> dict[str, Any]:
    """The map's cells from SIG's cell package (geo_views.geo_cells_pack), for drawing: each ring is its first vertex and
    integer offsets in 10**-scale degrees. Which cells a polygon selects is geo.zone_h3_cells, SIG's own rule."""
    step = 10.0 ** -pack["scale"]
    rings = []
    for r in pack["rings"]:
        x0, y0 = r[0], r[1]
        ring = [[x0, y0]] + [[round(x0 + dx * step, 6), round(y0 + dy * step, 6)] for dx, dy in zip(r[2::2], r[3::2], strict=True)]
        rings.append(ring + [ring[0]])
    return {"h3": pack["columns"]["h3"], "rings": rings}


def sig_chat(cells: list[str], question: str, use_llm: bool = True) -> dict[str, Any]:
    """SIG's reading of the area exactly as its console's chat asks it: the selected map cells as the zone, with the
    user's question or without one (the console's zone-only ask: "no text: the map area below is the question"). With
    both, the graph scores its seeds by the text and the place together; its LLM then reads the evidence by the
    project's rules: observations, then hypotheses. Without use_llm the narrator returns the evidence alone, no model."""
    r = http.post(f"{SIG_CHAT_URL}/api/chat/query", json={"question": question, "cells": cells, "use_llm": use_llm})
    if r.status_code >= 300:
        raise RuntimeError(f"SIG chat: {r.status_code} {r.text[:300]}")
    return r.json()


def sig_pattern(dataset_id: str, pattern_id: str) -> dict[str, Any]:
    """A finding's place on the map: its cells' ids and rings."""
    r = http.get(f"{SIG_URL}/api/geo/{dataset_id}/patterns/{pattern_id}", params={"rings": "true"})
    return r.json() if r.status_code < 300 else {}


CRED_RANK = {1: 0, 2: 1, 6: 2, 5: 3}  # SIG's credibility, best first: confirmed, unconfirmed, no other source, contradicted


def sig_findings(qa: dict[str, Any]) -> list[dict[str, Any]]:
    """The chat's evidence items with their cells' rings for the map: the ones the zone weighed as inside (place > 0)
    first, the most credible first, then the ones outside it that SIG's answer also tells; at most MAX_INSIGHTS."""
    evidence = (qa.get("view") or {}).get("evidence") or {}
    area = evidence.get("area") or {}
    source = {p["key"]: p for p in evidence.get("provenance") or []}  # key -> dataset_id, filename
    layer_of = {it["key"]: next((c for c in it.get("scope") or [] if c.startswith("context_")), "") for it in evidence.get("items", [])}
    # the zone's own first, then SIG's credibility order; a cross-layer finding (a context_<layer>_band condition: the base
    # layer's events read against the context layer) is tagged for the task and the joint prompt, not promoted
    items = sorted(evidence.get("items", []), key=lambda it: (-((area.get(it["key"]) or 0) > 0), CRED_RANK.get(it.get("credibility"), 9)))
    out = []
    for it in items[:MAX_INSIGHTS]:  # SIG's own text, in its own language, as is
        dataset_id = (source.get(it["key"]) or {}).get("dataset_id", "")
        place = sig_pattern(dataset_id, it["pattern_id"]) if dataset_id else {}
        PATTERN_CELLS[it["pattern_id"]] = place.get("cells") or []
        out.append(
            {
                "key": it["key"],
                "pattern_id": it["pattern_id"],
                "dataset_id": dataset_id,
                "filename": (source.get(it["key"]) or {}).get("filename", ""),
                "role": it.get("role", ""),
                "inside": (area.get(it["key"]) or 0) > 0,
                "layer": layer_of[it["key"]],  # the cross-layer condition, e.g. context_chicago_bars_band=q2, else ""
                "headline": it.get("headline", ""),
                "scope": it.get("scope_text", ""),
                "shift": it.get("shift_text") or [],
                "validation": it.get("validation", ""),
                "trust": it.get("trust", ""),
                "credibility": it.get("credibility"),
                "place": area.get(it["key"]),
                "rings": place.get("rings") or [],
            }
        )
    return out


def sig_zone_answer(cells: list[str], question: str, use_llm: bool = True) -> dict[str, Any]:
    """One SIG reading of the zone, with the user's question or without it: the findings for the map, SIG's answer, the
    evidence its model read, the files the findings come from, and the call's trace — the same shape for the preview on
    polygon close (no question) and for the ask (the question), so the ask can fall back on the preview."""
    t0 = time.time()
    trace = {"call": f"POST {SIG_CHAT_URL}/api/chat/query", "question": question, "cells_count": len(cells)}
    qa = sig_chat(cells, question, use_llm)
    insights = sig_findings(qa)
    metrics, mode, model = qa.get("metrics") or {}, qa.get("answer_mode"), (qa.get("llm") or {}).get("model")
    trace["result"] = {
        "відповідь": f"{mode} · {model or 'без LLM'} · {(qa.get('llm') or {}).get('latency_s')} с LLM",
        "граф": f"точок входу {metrics.get('seed_count')} · обійдено станів {metrics.get('visited_states')} · глибина "
        f"{metrics.get('traversal_depth')} · якорів {metrics.get('anchors_visited')} · свідчень {metrics.get('retrieved_evidence')}",
        "знахідки": f"у зоні {sum(it['inside'] for it in insights)} · поза зоною {sum(not it['inside'] for it in insights)}",
        "свідчення графа": f"{len(qa.get('prompt', ''))} символів",
    }
    return {
        "insights": insights,
        "sig_answer": qa.get("answer", ""),
        "sig_evidence": qa.get("prompt", ""),
        "sig_mode": mode,
        "sig_model": model,
        "llm": qa.get("llm") or {},
        "sig_sources": qa.get("provenance_footer", ""),
        "files": finding_files(qa),
        "trace": trace,
        "seconds": round(time.time() - t0, 1),
    }


def in_zone(reading: dict[str, Any]) -> bool:
    return any(it["inside"] for it in reading.get("insights") or [])


# ----------------------------------------------------------------------------- the zone and the files


def flag_in_aoi(df: pd.DataFrame, polygon: dict[str, Any]) -> pd.Series:
    """1 for the rows whose (lon, lat) lies inside the GeoJSON polygon, else 0. Deterministic, no model."""
    poly = shape(polygon)
    inside = shapely.contains_xy(poly, df["lon"].to_numpy(dtype=float), df["lat"].to_numpy(dtype=float))
    return pd.Series(inside.astype(int), index=df.index, name="in_aoi")


def zone_frames(polygon: dict[str, Any], cells: list[str], cell_col: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    """Copies of the layers with in_aoi and aoi_zone for this polygon and its cells, and the rows per class; the layers
    themselves stay as the build left them."""
    frames, counts = {}, {}
    for name, df in DATA.items():
        f = df.assign(in_aoi=flag_in_aoi(df, polygon))
        f["aoi_zone"] = aoi_zone(f, cells, cell_col)
        frames[name] = f
        counts[name] = {"total": int(len(f)), **{k: int((f["aoi_zone"] == k).sum()) for k in ("inside", "adjacent", "far")}}
    return frames, counts


# What SIG's own words mean (subgroup_miner/spatial.py LISA_WORDS, insight_contracts/wording.py), for ADS and the joint prompt
SIG_GLOSSARY = """## Словник термінів графа інсайтів SIG

Знахідки SIG записані мовою графа, англійською. Терміни означають таке.

- Комірка мапи (H3): шестикутна комірка сітки H3. У SIG рядок даних є коміркою з подіями, а не окремою подією. Підгрупа (`scope`, subgroup) є набором комірок, що відповідають умовам `<колонка> is <значення>`.
- `points`: кількість подій у комірці.
- `share <колонка> <значення>`: частка подій комірки з цим значенням. Наприклад, `share crime robbery` означає частку пограбувань серед подій комірки.
- `<колонка>_mode`: найчастіше значення серед подій комірки. `district_mode is district 10` означає, що найбільше подій комірки належить округу `district 10`.
- `neighbours_<лічильник>_band is q1…q4`: наскільки жваві шість сусідніх комірок за цим лічильником, у чвертях усіх комірок. `q1` найтихіша чверть, `q4` найжвавіша. Сама комірка тут не враховується.
- `lisa_points`: локальний кластер LISA (локальний індекс Морана), тобто чи утворюють комірка і її сусіди скупчення за кількістю подій.
  - `hot spot`: гаряча точка, жвава комірка серед жвавих сусідів.
  - `cold spot`: холодна точка, тиха комірка серед тихих сусідів.
  - `high outlier`: жвава комірка серед тихих сусідів.
  - `low outlier`: тиха комірка серед жвавих сусідів.
  - `no cluster`: комірка не входить ні в жваве, ні в тихе скупчення. Це звичайне місто без вираженого скупчення, а не відсутність подій.
- `context_<шар>_band`: скільки точок контекстного шару, наприклад барів, у самій комірці. `none nearby` означає жодної, `q1…q4` чверті серед комірок, де вони є.
- Зсув `<метрика>: mild|moderate|strong increase|decrease, ±z sd (median a vs b overall)`: медіана метрики в комірках підгрупи (a) проти медіани всієї мапи (b). `sd` є робастним стандартним відхиленням, тобто різницею медіан, поділеною на MAD. Слова сили: mild означає слабкий, moderate помірний, strong сильний.
- `bootstrap stability`: частка повторних вибірок, у яких підгрупа відтворюється; 1.0 означає завжди.
- `adjusted p`: p-значення після поправки на множинні порівняння; що менше, то менш імовірно, що знахідка випадкова.
- `credibility`, рівень довіри: 1 підтверджено іншим джерелом; 2 не підтверджено, бо інше джерело має події в цьому місці, але такої знахідки не має; 5 суперечить іншому джерелу; 6 немає іншого джерела. Це окремо від статистичної перевірки: сильне p з одного джерела все одно не є підтвердженням.
- `source F`: надійність джерела не оцінена тим, хто завантажив дані.
- Роль знахідки: `seed` знайдена прямо за питанням і зоною; `transversal` досягнута через латентний якір, тобто схожа картина в іншій частині даних; лише за місцем означає, що знахідка лежить поруч, і це не доводить зв'язку.
- `AREA`, місце в зоні: наскільки комірки знахідки збігаються з виділеною зоною, 100% означає ті самі комірки. Знахідки поза зоною (`outside`) описують інші місця.
- Граф SIG спільний для всіх завантажених датасетів. Знахідка з іншого файлу, наприклад іншого року, є контекстом, а не фактом про базовий шар цього аналізу.
"""


def finding_files(qa: dict[str, Any]) -> dict[str, list[str]]:
    """file -> the keys of the chat's findings that come from it (the evidence's provenance)."""
    out: dict[str, list[str]] = {}
    for p in ((qa.get("view") or {}).get("evidence") or {}).get("provenance") or []:
        out.setdefault(p.get("filename") or p.get("dataset_id", "?"), []).append(p["key"])
    return out


CREDIBILITY_WORDS = {1: "confirmed by another source", 2: "not confirmed", 5: "contradicted by another source", 6: "no other source"}


def observation_lines(items: list[dict[str, Any]]) -> list[str]:
    """SIG's findings as O# lines for the ADS task and the joint prompt, SIG's fields verbatim, in sig_findings' order
    (the zone's first, the most credible first); a finding from another file of the graph is context only."""
    out = []
    for n, it in enumerate(items, 1):
        other = "" if it["filename"] == BASE["file"] else " (other file, context only)"
        layer = f" · cross-layer: {it['layer']} (the context layer's points in the cell)" if it.get("layer") else ""
        out.append(
            f"O{n} [{it['key']}] {it['headline']} · where: {it['scope'] or '—'} · shift: {'; '.join(it['shift']) or '—'} · "
            f"in zone: {'yes' if it['inside'] else 'no'} · credibility: {CREDIBILITY_WORDS.get(it['credibility'], 'not rated')} · "
            f"file: {it['filename'] or '?'}{other}{layer}"
        )
    return out


def zone_facts(zone: dict[str, Any]) -> tuple[int, str, str, dict[str, Any], dict[str, Any]]:
    """What both markdowns say about the zone: its vertex count, the lon and lat spans, the two layers' counts."""
    coords = zone["polygon"]["coordinates"][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    return len(coords) - 1, f"{min(lons):.4f}…{max(lons):.4f}", f"{min(lats):.4f}…{max(lats):.4f}", zone["base"], zone["context"]


def aoi_md(zone: dict[str, Any], data: dict[str, Any], files: dict[str, list[str]]) -> str:
    """ADS's domain knowledge for the zone: a short English dictionary of the tables and columns, declarative. ADS's
    guardrail reads every uploaded .md and its grounder keeps 800 tokens of them, so nothing that belongs in the task
    (the question, SIG's answer, the findings, the guidance) goes here."""
    vertices, lon_span, lat_span, b, c = zone_facts(zone)
    r = data["resolution"]
    cols = h3_columns(r)
    levels = "; ".join(
        f"{col}: resolution {r - k} (≈ {h3.average_hexagon_area(r - k, unit='km^2'):.2f} km² a cell)" for col, k in zip(cols, H3_LEVELS, strict=True)
    )
    sources = "; ".join(f"{name}: {', '.join(keys)}" for name, keys in files.items()) or "none in this answer"
    return "\n".join(
        [
            "# Area of interest: tables and columns",
            f"- Zone: the user's polygon on the map, {vertices} vertices, lon {lon_span}, lat {lat_span}. "
            f"Base events: {b['inside']} of {b['total']} inside; context points: {c['inside']} of {c['total']} inside.",
            f"- base_aoi.csv ({BASE['file']}, {BASE['label']}): the events. context_aoi.csv ({CONTEXT['file']}, {CONTEXT['label']}): the context points. "
            f"Both carry {', '.join(cols)}, in_aoi, aoi_zone.",
            f"- H3 columns: {levels}; a coarser id is the parent of the finer one; the finest is the level SIG analysed.",
            "- in_aoi: 1 = the point lies inside the polygon (exact point-in-polygon), 0 = outside.",
            "- aoi_zone: inside (in_aoi 1) | adjacent (outside the polygon, in a zone cell or one of its six neighbours) | far.",
            f"- SIG's cells: one row of SIG's table = one {cols[0]} cell with its events; shares and medians per cell exist for cells with 5 or more events; "
            f"per-type counters SIG built: {', '.join(data.get('counted') or []) or 'none'}.",
            f"- SIG's terms: {SIG_TERMS}; credibility 1 confirmed by another source, 2 not confirmed, 5 contradicted, 6 no other source.",
            f"- SIG's findings come from: {sources}. The graph holds other datasets too; a finding from another file is context, not a fact about the base layer.",
        ]
    )


def domain_md(zone: dict[str, Any], files: dict[str, list[str]] | None = None) -> str:
    """The joint prompt's domain knowledge, in the report's language: how the zone was applied to the data, where the
    findings come from, and the SIG glossary."""
    vertices, lon_span, lat_span, b, c = zone_facts(zone)
    by_file = [
        f"- {name}: {', '.join(keys)}, "
        + ("базовий шар цього аналізу." if name == BASE["file"] else "інший датасет графа, не базовий шар; його числа описують інші дані.")
        for name, keys in (files or {}).items()
    ]
    sources = [
        "## Джерела знахідок SIG",
        "",
        f"Базовий шар цього аналізу: {BASE['file']}. Граф SIG містить і інші датасети, наприклад інший рік того самого типу подій. "
        "Знахідка з іншого файлу не є фактом про базовий шар, і її числа не можна поєднувати з числами ADS.",
        "",
        *by_file,
        "",
    ]
    return "\n".join(
        [
            "# Зона інтересу (AOI): методика",
            "",
            f"- Зона задана користувачем як полігон на мапі: {vertices} вершин, довгота {lon_span}, широта {lat_span}.",
            f"- Файл base_aoi.csv: базовий шар подій, файл {BASE['file']} ({BASE['label']}), {b['total']} рядків, з них {b['inside']} у зоні.",
            f"- Файл context_aoi.csv: контекстний шар, файл {CONTEXT['file']} ({CONTEXT['label']}), {c['total']} рядків, з них {c['inside']} у зоні.",
            "- Колонка in_aoi в обох файлах: 1 означає, що точка лежить усередині полігона, 0 поза ним. "
            "Обчислено детерміновано (point-in-polygon), без моделі. Колонка aoi_zone: inside у полігоні, adjacent поза ним у комірках "
            "зони чи їх шести сусідах, far решта.",
            f"- Граф інсайтів SIG побудовано з тих самих шарів: базовий шар розбито на комірки H3, контекстний шар {CONTEXT['label']} "
            "враховано як окрему ознаку комірки в крос-прогоні.",
            f"- Зону передано в SIG як {len(zone.get('cells') or [])} комірок H3 базового шару, чиї центри лежать у полігоні, разом із "
            "питанням користувача (або без нього, якщо відповідь із питанням не мала знахідок у зоні). Від цих комірок ретривер SIG "
            "увійшов у граф, обійшов його і зібрав знахідки в зоні та пов'язані з ними; мовна модель SIG описала спостереження і гіпотези.",
            "",
            *(sources if by_file else []),
            SIG_GLOSSARY,
        ]
    )


SIG_TERMS = (
    "points = events in the cell; share_<column>_<value> = share of the cell's events with that value; count_<column>_<value> = "
    "their number; <column>_mode = the cell's most frequent value; neighbours_<counter>_band q1…q4 = quartile of the counter summed over the six "
    "neighbours; lisa_points = local Moran class of event counts (hot spot, cold spot, high outlier, low outlier, no cluster); "
    "context_<layer>_band = none nearby, or the quantile of context points in the cell; shift = the subgroup's median against the map's median "
    "in robust sd"
)

# Scenario B: what an insight is and how to read it, ahead of the user's question (the query is the part ADS never trims)
INSIGHT_PRIMER = f"""Read this first: established insights about this dataset.

Before this task a separate statistical system (SIG) mined these very tables and validated what it found (bootstrap stability and a multiple-testing correction). Each insight O# below is a fact already computed for this dataset, not a suggestion.

How to read an insight:
- `cells where` is the condition that selects a set of H3 map cells of the base layer (the finest h3_r column); SIG's row is a cell with its events, not an event.
- `established` is how a per-cell metric in those cells differs from the whole map: the subgroup's median against the map's median, in robust standard deviations (sd).
- `in zone` says whether those cells overlap the user's area of interest.
- SIG's terms: {SIG_TERMS}.

Rules for the insights:
1. They are ground truth. Do not verify, test, reproduce, re-derive or recompute an insight, its medians, its sd or its cell condition; no step of your plan may have that purpose. The per-cell features behind them (neighbour bands, LISA classes, per-cell shares) are not columns of the tables, so a recomputation would only produce a different, wrong number.
2. Use them as navigation: they tell you which event types, places and conditions deserve attention. Spend your steps on what the question asks and the insights do not already state.
3. In the report cite an insight by its id and quote its numbers exactly as given, as an established fact, next to your own figures.

"""


def insight_blocks(items: list[dict[str, Any]], zone_cells: list[str]) -> list[str]:
    """Scenario B's one template per insight: the id and its place, the cell condition, each shift, the checks it passed."""
    zone = set(zone_cells)
    out = []
    for n, it in enumerate(items, 1):
        cells = PATTERN_CELLS.get(it.get("pattern_id", ""), [])
        place = f" ({len(zone & set(cells))} of its {len(cells)} cells are zone cells)" if cells else ""
        lines = [f"O{n} · in zone: {'yes' if it['inside'] else 'no'}{place}", f"  cells where: {it['scope'] or '—'}"]
        lines += [f"  established: {s}" for s in it["shift"]]
        if it.get("validation"):
            lines.append(f"  checks passed: {it['validation']}")
        if it.get("layer"):
            lines.append(f"  layers: links the base layer's events to the context layer's points in the cell ({it['layer']})")
        if it["filename"] != BASE["file"]:
            lines.append(f"  file: {it['filename'] or '?'} (another dataset of the graph: context only, not a fact about the base layer)")
        out.append("\n".join(lines))
    return out


def sig_flag_files(files: dict[str, bytes], items: list[dict[str, Any]], cell_col: str) -> tuple[dict[str, bytes], list[str]]:
    """Scenario C: the zone's tables with one sig_o<n> column per finding of the base file (1 = the row's cell is one of
    the finding's cells), and each column's dictionary line for the task."""
    frames = {name: pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False) for name, data in files.items()}
    notes = []
    for n, it in enumerate(items, 1):
        cells = set(PATTERN_CELLS.get(it.get("pattern_id", ""), []))
        if not cells or it["filename"] != BASE["file"]:
            continue
        for f in frames.values():
            f[f"sig_o{n}"] = f[cell_col].isin(cells).astype(int)
        base = frames["base_aoi.csv"]
        inside = int(((base[f"sig_o{n}"] == 1) & (base["in_aoi"] == "1")).sum())
        layer = f"; it links the base layer to the context layer ({it['layer']})" if it.get("layer") else ""
        notes.append(
            f"- sig_o{n}: cells where {it['scope'] or '—'}; there {'; '.join(it['shift']) or '—'}{layer}; {inside} base rows inside the AOI carry 1"
        )
    return {name: f.to_csv(index=False).encode("utf-8") for name, f in frames.items()}, notes


PLAN_PROMPT = """You plan the work of a data-science agent that answers a user's question about an area of interest (AOI) on two tables.

Question: {question}
{data}

Established insights about this dataset (already computed and validated by a separate system; facts, not claims):
{insights}

Write the analysis plan: 3 to 5 numbered steps, one line each. A step says what to compute, on which table, columns and cut, and ends with the insight ids that motivated it in brackets, or (question) when the question alone did.
- At least one step answers the question directly.
- Let the insights that concern the zone and the question steer where the steps look: the event types, the conditions, the layers they name.
- No step verifies or recomputes an insight; a step is computable from the listed columns alone.
- Output only the numbered steps, in English."""


def task_parts(
    task: str, head: list[dict[str, str]], sig: dict[str, Any] | None, zone: dict[str, Any] | None, cell: str, plan: str, columns: list[str] | None
) -> list[dict[str, str]]:
    """The ADS task of scenarios B–E after the shared head (frame, question, data): what of SIG it carries and the guidance."""
    items = (sig or {}).get("insights") or []
    pending = "(after the ask)" if sig is None else ""
    compare = (
        "- Where the question points to it, compare inside vs adjacent vs far at the same level (rows per cell for counts, shares for categories).\n"
    )
    causes = "- Describe differences as shifts and ratios, not causes; proximity is not causation."
    if task == "none":
        return [*head, {"kind": "guidance", "text": "\n\nGuidance:\n" + compare + causes}]
    if task == "facts":
        blocks = pending or "\n\n".join(insight_blocks(items, (zone or {}).get("cells") or [])) or "SIG has no insights for this zone."
        guidance = (
            "\n\nGuidance:\n- Answer the question with your own computations on the tables. " + compare[2:] + "- The insights are settled: "
            "quote those that bear on the question by id with their numbers, and add what they do not state (dates, counts in the zone, "
            "breakdowns).\n" + causes
        )
        return [
            {"kind": "primer", "text": INSIGHT_PRIMER},
            *head,
            {"kind": "insights", "text": "\n\nEstablished insights (facts; do not recompute):\n" + blocks},
            {"kind": "guidance", "text": guidance},
        ]
    if task == "columns":
        notes = pending or "\n".join(columns or []) or "SIG has no findings for this zone: the tables carry no sig_o columns."
        text = (
            "\n\nSIG columns (both tables): SIG, a statistical insight miner, was run on this dataset before this task. Each sig_o<n> is 1 when "
            f"the row's {cell} cell belongs to a set of cells that SIG found to differ from the whole map, else 0:\n" + notes
        )
        guidance = (
            "\n\nGuidance:\n" + compare + "- The sig_o columns are ready-made cuts: where one bears on the question, split your figures by it "
            "(1 vs 0) inside the zone and name the column.\n" + causes
        )
        return [*head, {"kind": "sig_columns", "text": text}, {"kind": "guidance", "text": guidance}]
    lines = pending or "\n".join(observation_lines(items)) or "SIG has no findings for this zone."
    guidance = (
        "\n\nGuidance:\n- Follow the plan step by step; add at most one step of your own when the data call for it.\n"
        "- The insights are settled facts: quote them by O# where a step rests on one; do not recompute them.\n" + causes
    )
    return [
        *head,
        {"kind": "observations", "text": "\n\nEstablished insights about this dataset (O#; facts computed by SIG before this task):\n" + lines},
        {"kind": "plan", "text": "\n\nAnalysis plan:\n" + (pending or plan.strip() or "(no plan: answer the question directly)")},
        {"kind": "guidance", "text": guidance},
    ]


def ads_query_parts(
    question: str,
    zone: dict[str, Any] | None,
    data: dict[str, Any] | None,
    sig: dict[str, Any] | None,
    task: str = "answer",
    plan: str = "",
    columns: list[str] | None = None,
) -> list[dict[str, str]]:
    """The ADS task, part by part, as the page shows it: the frame, the user's words, the data with its geo columns and
    zone counts, SIG's answer verbatim, SIG's findings as O# lines, and the guidance. The frame is English (ADS's own
    prose is English and Cyrillic costs its budgets two to three times more); the question and SIG's text stay verbatim.
    Before the zone and the ask the figures and the SIG parts are placeholders. Joined, the parts are the query.
    ``task`` is the scenario's hand-off (SCENARIOS): "answer" is this layout, the others are task_parts."""
    if zone and data:
        r, b, c = data["resolution"], zone["base"], zone["context"]
        cols = h3_columns(r)
        geo = (
            f"\n\nData: base_aoi.csv → df_base_aoi ({BASE['label']}, {b['total']} rows, {b['inside']} inside the AOI); "
            f"context_aoi.csv → df_context_aoi ({CONTEXT['label']}, {c['total']} rows, {c['inside']} inside).\n"
            f"Geo columns (both tables): {', '.join(cols)} are H3 cell ids (strings) at resolutions {', '.join(str(r - k) for k in H3_LEVELS)}; "
            "a coarser id is the parent of the finer one; group by them, never edit them. in_aoi is 1 for points inside the polygon (exact). "
            "aoi_zone is inside (= in_aoi 1) | adjacent (outside the polygon but in the zone's cells or their ring-1 neighbours) | far. "
            "A point in the same cell as the zone is not inside the zone.\n"
            f"Zone (base rows): inside={b['inside']}, adjacent={b['adjacent']}, far={b['far']}; {cols[0]} cells in the zone: {len(zone['cells'])}. "
            f"SIG counted these types per cell: {', '.join(data.get('counted') or []) or 'none'}."
        )
        cell, coarser = cols[0], f"h3_r{r - 1}"
    else:
        geo = "\n\nData: base_aoi.csv → df_base_aoi and context_aoi.csv → df_context_aoi with the H3 columns h3_r*, in_aoi and aoi_zone (the figures appear once the zone is drawn)."
        cell, coarser = "the finest h3_r", "the coarser h3_r column"
    if sig is None:
        answer = "\n\nSIG's answer for this zone (verbatim): (after the ask: SIG reads the zone with this question)"
        obs = "\n\nSIG findings (O#): (after the ask)"
    else:
        text = (sig.get("sig_answer") or "").strip()
        answer = "\n\nSIG's answer for this zone (verbatim):\n" + (text or "SIG gave no answer for this zone.")
        lines = observation_lines(sig.get("insights") or [])
        obs = (
            "\n\nSIG findings (O#; 'in zone' says whether the finding's cells lie in the AOI — check each that concerns the question on the data):\n"
            + ("\n".join(lines) or "SIG has no findings for this zone.")
        )
    head = [
        {"kind": "frame", "text": "Analyse the area of interest (AOI): "},
        {"kind": "user", "text": question},
        {"kind": "data", "text": geo},
    ]
    if task != "answer":
        return task_parts(task, head, sig, zone, cell, plan, columns)
    cross = (  # only when a finding links the layers: the join is work ADS should not do for a question about a count trend
        f"- A finding marked cross-layer links the base layer to the context layer: reproduce that relation on the tables — count the context "
        f"rows per {cell} cell by joining the two tables on the cell id, then compare the counted type's rows per cell between cells with many, "
        "few and no context points, inside the zone and around it — and report it under its O#.\n"
        if sig and any(it.get("layer") for it in sig.get("insights") or [])
        else ""
    )
    guidance = (
        "\n\nGuidance:\n"
        f"- Where the question or SIG's answer points to it, compare inside vs adjacent vs far at the same level (rows per cell for counts, "
        f"shares for categories) and check on {coarser} whether the difference holds.\n"
        "- Treat SIG's conditions as extra cuts of the same computation and SIG's hypotheses as claims to test: confirmed / not confirmed / "
        "not testable with these columns. Do not re-mine spatial clusters.\n" + cross + "- Describe differences as shifts and ratios, not causes; "
        "proximity is not causation; label O# explicitly in the report."
    )
    return [*head, {"kind": "sig_answer", "text": answer}, {"kind": "observations", "text": obs}, {"kind": "guidance", "text": guidance}]


def ads_query(question: str, zone: dict[str, Any] | None, data: dict[str, Any] | None, sig: dict[str, Any] | None, task: str = "answer") -> str:
    return "".join(p["text"] for p in ads_query_parts(question, zone, data, sig, task))


# ----------------------------------------------------------------------------- ADS


def ads_session() -> str:
    return http.post(f"{ADS_URL}/sessions").json()["session_id"]


def ads_upload(sid: str, name: str, data: bytes) -> None:
    r = http.post(f"{ADS_URL}/sessions/{sid}/datasets", files={"file": (name, data)})
    if r.status_code >= 300:
        raise RuntimeError(f"ADS upload {name}: {r.status_code} {r.text[:300]}")


def ads_job(sid: str, query: str) -> str:
    r = http.post(f"{ADS_URL}/sessions/{sid}/jobs", json={"user_query": query})
    if r.status_code >= 300:
        raise RuntimeError(f"ADS job: {r.status_code} {r.text[:300]}")
    return r.json()["job_id"]


def ads_wait(sid: str, every: float = 5, limit: float = 1200) -> dict[str, Any]:
    deadline = time.time() + limit
    while True:
        r = http.get(f"{ADS_URL}/sessions/{sid}/status")
        if r.status_code == 200:
            return r.json()
        if r.status_code != 202:
            raise RuntimeError(f"ADS status: {r.status_code} {r.text[:300]}")
        if time.time() > deadline:
            raise TimeoutError("ADS job still running after the limit")
        time.sleep(every)


_WORKSPACE_REFS = re.compile(r"<iframe[^>]*>.*?</iframe>|!?\[([^\]]*)\]\(/workspace/[^)]*\)", re.S)


def ads_report(status: dict[str, Any]) -> str:
    last = (status.get("result") or {}).get("last_execute_response") or {}
    text = last.get("report") or ""
    if not text and last.get("answer") is not None:
        text = json.dumps(last["answer"], ensure_ascii=False, indent=2)
    return _WORKSPACE_REFS.sub(lambda m: m.group(1) or "", text).strip()


JOINT_MAX_TOKENS = 16384  # Copilot's final_response_max_tokens: the prompt keeps every row of the report's tables (a year of days)
PROMPTS: dict[str, str] = {}  # the last joint prompt, served apart (GET /trace/prompt) so /state stays light to poll
TRACES: dict[str, Any] = {}  # the last ADS job's counters and steps, served apart (GET /trace/ads)


def llm_chat(prompt: str) -> dict[str, Any]:
    body = {"model": "synthesis", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": JOINT_MAX_TOKENS}
    r = http.post(f"{LLM_URL}/v1/chat/completions", json=body)
    if r.status_code >= 300:
        raise RuntimeError(f"LLM gateway: {r.status_code} {r.text[:300]}")
    return r.json()


def joint_report(
    question: str, report: str, sig_answer: str, sig_evidence: str, domain: str, observations: str, ads_saw_sig: bool = True
) -> tuple[str, dict[str, Any]]:
    """Copilot's final response prompt adapted for two sources (prompts/joint_response.jinja): the ADS report, SIG's
    answer, the graph evidence it read and its findings as the O# lines, verbatim; the zone's methodology and the SIG
    glossary as domain knowledge. -> (the report, "" without its required sections; the call's trace)."""
    template = Template(JOINT_TEMPLATE.read_text(encoding="utf-8"))
    prompt = template.render(
        user_query=question,
        report=report,
        insight_analysis=sig_answer,
        insight_evidence=sig_evidence,
        observations=observations,
        domain_knowledge=domain,
        ads_saw_sig=ads_saw_sig,
        query_language="uk",
    )
    PROMPTS["joint"] = prompt
    resp = llm_chat(prompt)
    out = resp["choices"][0]["message"]["content"]
    trace = {
        "call": f"POST {LLM_URL}/v1/chat/completions",
        "route": "synthesis",
        "model": resp.get("model"),
        "max_tokens": JOINT_MAX_TOKENS,
        "template": f"prompts/{JOINT_TEMPLATE.name} (Copilot final_response.jinja, правки «AOI demo»)",
        "inputs": {
            "user_query: питання користувача": len(question),
            "Analytical_Report: звіт ADS": len(report),
            "Insight_Analysis: відповідь SIG": len(sig_answer),
            "Insight_Evidence: свідчення графа SIG": len(sig_evidence),
            "Insight_Observations: рядки O#": len(observations),
            "User_Domain_Knowledge: методика і словник": len(domain),
        },
        "prompt_chars": len(prompt),
        "usage": resp.get("usage"),
        "answer_chars": len(out),
    }
    if "## Результати аналізу" in out and "## Методологія" in out:
        return out, trace
    return "", {**trace, "raw": out}  # the answer that missed its required sections, for the trace


# ----------------------------------------------------------------------------- the flows


def run_datasets(count_values: list[str]) -> None:
    d = STATE["datasets"]
    t0 = time.time()
    try:
        d["stage"] = "uploading"
        base = sig_upload(base_spec(count_values))
        ctx = sig_upload(CONTEXT)
        d["base"] = {"upload_id": base["upload_id"], "batches": list(base.get("levels", {}).values()) or [base["batch_id"]]}
        d["context"] = {"upload_id": ctx["upload_id"], "batches": list(ctx.get("levels", {}).values()) or [ctx["batch_id"]]}
        d["stage"] = "waiting"
        records = sig_wait(d["base"]["batches"] + d["context"]["batches"])
        d["base"]["dataset_id"] = records[0]["dataset_id"]  # the base's finest level: the map whose cells a zone selects
        d["seconds_layers"] = round(time.time() - t0, 1)
        d["stage"] = "crosstype"
        d["cross"] = [b["batch_id"] for b in sig_crosstype(base["upload_id"], ctx["upload_id"])]
        runs = sig_wait(d["cross"], empty_ok=True)
        runs = [sig_wait([r["duplicate_of"]])[0] if r.get("duplicate_of") else r for r in runs]  # a repeat: the run it repeats
        d["cross_insights"] = sum(int((r.get("metrics") or {}).get("validated_insights") or 0) for r in runs if r.get("status") == "READY")
        d["cross_note"] = "; ".join((r.get("error") or {}).get("message", "") for r in runs if r.get("status") == "FAILED")
        pack = http.get(f"{SIG_URL}/api/geo/{d['base']['dataset_id']}/cells")
        if pack.status_code >= 300:
            raise RuntimeError(f"SIG cells of {d['base']['dataset_id']}: {pack.status_code} {pack.text[:300]}")
        cells = pack.json()
        GRID.update(cell_grid(cells))
        GRID["resolution"] = sig_resolution(d["base"]["dataset_id"])
        cols = h3_columns(GRID["resolution"])  # h3_r{R}, h3_r{R-1}, h3_r{R-2}: SIG's own cells and their parents, on both layers
        DATA["base"] = enrich_h3(pd.read_csv(DEMO_DIR / BASE["file"]), GRID["resolution"])
        DATA["context"] = enrich_h3(pd.read_csv(DEMO_DIR / CONTEXT["file"]), GRID["resolution"])
        d["rows"] = {"base": int(len(DATA["base"])), "context": int(len(DATA["context"])), "cells": len(GRID["h3"])}
        d["resolution"], d["levels"] = GRID["resolution"], [int(c[4:]) for c in cols]
        d["counted"] = sorted(c for c in cells["columns"] if c.startswith("count_"))  # the counters SIG really built
        d["h3_mismatch"] = len((set(DATA["base"][cols[0]]) - {""}) ^ set(GRID["h3"]))  # the demo's cells vs SIG's: a warning, never a stop
        d["seconds"] = round(time.time() - t0, 1)
        d["stage"] = "ready"
    except Exception as exc:  # the happy path shows the text and stops
        log.exception("datasets failed")
        d["stage"], d["error"] = "error", f"{type(exc).__name__}: {exc}"


def run_mine(z: dict[str, Any]) -> None:
    """The preview, as soon as the zone is drawn: SIG's reading of the place before any question (the cells alone).
    The ask repeats the call with the question and falls back on this reading when the text finds nothing in the zone."""
    s = z["sig"]
    try:
        s.update(sig_zone_answer(z["cells"], "", SCENARIOS[HANDOFF]["sig_llm"]))
        s["stage"] = "ready"
    except Exception as exc:  # the question can still go to ADS, with whatever the ask's own SIG call brings
        log.exception("zone preview failed")
        s["stage"], s["error"] = "error", f"{type(exc).__name__}: {exc}"
        s.setdefault("seconds", 0)
        s.update({"insights": [], "sig_answer": "", "sig_evidence": "", "files": {}})


def run_ask(question: str, scenario: str) -> None:
    a, zone, mode = STATE["ask"], STATE["zone"], SCENARIOS[scenario]
    preview = zone["sig"]  # SIG's reading of the place alone, done when the zone was drawn
    timings: dict[str, float] = {}
    t = time.time()

    def mark(stage: str) -> None:
        nonlocal t
        timings[a["stage"]] = round(time.time() - t, 1)
        t = time.time()
        a["stage"] = stage

    trace = a["trace"] = {}  # per stage: what went to the service and what came back
    try:
        combined = sig_zone_answer(zone["cells"], question, mode["sig_llm"])  # the question and the zone together: the graph scores both
        fallback = not in_zone(combined) and in_zone(preview)  # the text pulled the seeds out of the zone: the preview stands
        chosen = preview if fallback else combined
        a["sig_entry"] = "preview" if fallback else "combined"
        for k in ("insights", "sig_answer", "sig_evidence", "sig_mode", "sig_model", "sig_sources"):
            a[k] = chosen.get(k)
        files_of = chosen.get("files") or {}
        a["aoi_md"], a["domain_md"] = aoi_md(zone, STATE["datasets"], files_of), domain_md(zone, files_of)
        a["observations"] = "\n".join(observation_lines(a["insights"]))
        trace["search"] = {
            **chosen.get("trace", {}),
            "entry": a["sig_entry"],
            "llm": combined.get("llm") if mode["sig_llm"] else {},  # this ask's narrator call: model, latency, usage
            "combined": combined["trace"].get("result"),
            "preview": preview.get("trace", {}).get("result"),
        }
        tables, columns, plan = zone["files"], None, ""  # this zone's tables as taken when its polygon closed
        if mode["task"] == "columns":
            tables, columns = sig_flag_files(tables, a["insights"], h3_columns(GRID["resolution"])[0])
        if mode["task"] == "plan":
            mark("plan")
            head = ads_query_parts(question, zone, STATE["datasets"], chosen, "none")[2]["text"].strip()
            resp = llm_chat(PLAN_PROMPT.format(question=question, data=head, insights=a["observations"] or "none"))
            plan = a["plan"] = resp["choices"][0]["message"]["content"]
            trace["plan"] = {"call": f"POST {LLM_URL}/v1/chat/completions", "model": resp.get("model"), "usage": resp.get("usage")}
        mark("ads")
        sid = ads_session()
        a["ads_session"] = sid
        parts = ads_query_parts(question, zone, STATE["datasets"], chosen, mode["task"], plan, columns)
        a["ads_query"] = "".join(p["text"] for p in parts)
        files = {**tables, "aoi.md": a["aoi_md"].encode("utf-8")}  # the tables and the column dictionary
        trace["ads"] = {
            "call": f"POST {ADS_URL}/sessions/{sid}/jobs",
            "files": {name: len(data) for name, data in files.items()},
            "query_parts": parts,
        }
        for name, data in files.items():
            ads_upload(sid, name, data)
        a["ads_job"] = ads_job(sid, a["ads_query"])
        status = ads_wait(sid)
        a["ads_outcome"] = status.get("outcome")
        a["ads_report"] = ads_report(status)
        last = (status.get("result") or {}).get("last_execute_response") or {}
        counters = (status.get("result") or {}).get("system_counters") or {}
        steps = (status.get("result") or {}).get("execution_trace") or []
        TRACES["ads"] = {"counters": counters, "steps": steps}
        trace["ads"]["counters"] = counters
        trace["ads"]["result"] = {
            "outcome": f"{a['ads_outcome']} · {last.get('completion_status')}",
            "кроків": counters.get("total_steps_taken"),
            "інструменти": ", ".join(dict.fromkeys(str(s.get("tool_name") or "?") for s in steps)) or "—",
            "domain_knowledge_blocked": int(counters.get("domain_knowledge_blocked") or 0),  # 1: ADS dropped aoi.md (its guardrail)
            "звіт": f"{len(a['ads_report'])} символів",
            "файли ADS": ", ".join(o.get("filename", "") for o in last.get("outputs") or []) or "—",
        }
        http.post(f"{ADS_URL}/sessions/{sid}/close")
        if not a["ads_report"]:
            raise RuntimeError(f"ADS returned no report (outcome {a['ads_outcome']})")
        mark("joint")
        if mode["final_sig"]:  # the two reports joined; otherwise the answer is built from the ADS report alone
            sig_parts = (a["sig_answer"], a["sig_evidence"], a["domain_md"], a["observations"])
        else:
            sig_parts = ("", "", a["domain_md"], "")
        a["joint_report"], trace["joint"] = joint_report(question, a["ads_report"], *sig_parts, ads_saw_sig=mode["task"] != "none")
        if not a["joint_report"]:
            a["joint_error"] = "спільний звіт не зібрано: відповідь без обов'язкових розділів"
        mark("done")
    except Exception as exc:
        log.exception("ask failed")
        a["failed_stage"] = a["stage"]
        a["stage"], a["error"] = "error", f"{type(exc).__name__}: {exc}"
    a["timings"] = timings


# ----------------------------------------------------------------------------- HTTP

app = FastAPI(title="AOI demo")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static/index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    out = {}
    for name, url in (("sig", f"{SIG_URL}/api/health"), ("ads", f"{ADS_URL}/health"), ("llm", f"{LLM_URL}/health")):
        try:
            out[name] = "ok" if httpx.get(url, timeout=5).status_code == 200 else "down"
        except httpx.HTTPError as exc:
            out[name] = f"down: {type(exc).__name__}"
    out["template"] = "ok" if JOINT_TEMPLATE.is_file() else f"missing: {JOINT_TEMPLATE}"
    return out


@app.get("/state")
def state() -> dict[str, Any]:
    zone = STATE["zone"]
    return {**STATE, "zone": {k: v for k, v in zone.items() if k != "files"} if zone else None}  # the tables' bytes are not for the page


@app.get("/ads-query")
def ads_query_preview(question: str = "") -> dict[str, Any]:
    """The ADS task for this question, part by part, as the run will build it: the figures once the zone is drawn, the
    SIG parts (its answer with this question, the O# lines) only when the ask runs, so the preview is never exact."""
    return {"parts": ads_query_parts(question.strip(), STATE["zone"], STATE["datasets"], None, SCENARIOS[HANDOFF]["task"]), "exact": False}


@app.get("/trace/prompt")
def trace_prompt() -> PlainTextResponse:
    """The last joint prompt as sent to the LLM: long, so the page asks for it only when opened."""
    return PlainTextResponse(PROMPTS.get("joint", ""))


@app.get("/trace/ads")
def trace_ads() -> dict[str, Any]:
    """The last ADS job's counters (steps, LLM calls, tokens) and its execution trace, step by step."""
    return TRACES.get("ads", {})


@app.get("/datasets/options")
def datasets_options() -> dict[str, Any]:
    """The base file's category column and its values with located rows: what the page offers to count per cell."""
    return {
        "column": BASE["category"],
        "values": category_options(),
        "min_rows": COUNT_MIN,
        "default": list(COUNT_DEFAULT),
        "labels": {"base": BASE["label"], "context": CONTEXT["label"]},  # the page's words for the layers
    }


@app.post("/datasets")
def datasets(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """The build: the chosen category values go to SIG as count_values with the base layer; none means no counters."""
    if STATE["datasets"] and STATE["datasets"]["stage"] not in ("ready", "error"):
        return STATE["datasets"]
    countable = {o["value"] for o in category_options() if o["status"] == "ok"}
    count_values = [v for v in (body or {}).get("count_values") or [] if v in countable]
    STATE["datasets"] = {"stage": "starting", "base": None, "context": None, "cross": [], "error": None, "count_values": count_values}
    STATE["zone"], STATE["ask"] = None, None
    threading.Thread(target=run_datasets, args=(count_values,), daemon=True).start()
    return STATE["datasets"]


@app.post("/zone")
def zone(body: dict[str, Any]) -> dict[str, Any]:
    if "base" not in DATA:
        raise HTTPException(409, "датасети ще не готові")
    if STATE["ask"] and STATE["ask"]["stage"] not in ("done", "error"):  # the running ask keeps its zone; a new one waits
        raise HTTPException(409, "питання ще виконується: зону можна перемалювати після відповіді")
    polygon = body.get("polygon") or {}
    if polygon.get("type") != "Polygon" or len((polygon.get("coordinates") or [[]])[0]) < 4:
        raise HTTPException(400, "polygon: GeoJSON Polygon з щонайменше трьома вершинами")
    cells = zone_h3_cells(polygon, GRID["resolution"], GRID["h3"])  # the zone as SIG reads a geometry
    if not cells:
        raise HTTPException(400, "у полігоні немає жодної комірки мапи SIG: обведіть ділянку з подіями")
    frames, counts = zone_frames(polygon, cells, h3_columns(GRID["resolution"])[0])
    samples = {}
    for name, f in frames.items():  # the points in the zone the map draws, at most SAMPLE_POINTS a layer
        hit = f[f["in_aoi"] == 1]
        samples[name] = hit.sample(min(SAMPLE_POINTS, len(hit)), random_state=0)[["lon", "lat"]].round(5).values.tolist()
    ring_of = dict(zip(GRID["h3"], GRID["rings"], strict=True))
    STATE["zone"] = {
        "polygon": polygon,
        **counts,
        "cells": cells,
        "cell_rings": [ring_of[c] for c in cells],
        "sample": samples["base"],
        "sample_context": samples["context"],
        "files": {f"{name}_aoi.csv": f.to_csv(index=False).encode("utf-8") for name, f in frames.items()},  # what the ask uploads
        "sig": {"stage": "mining"},  # the preview starts now, before any question
    }
    STATE["ask"] = None
    threading.Thread(target=run_mine, args=(STATE["zone"],), daemon=True).start()
    return STATE["zone"]


@app.post("/ask")
def ask(body: dict[str, Any]) -> dict[str, Any]:
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question: порожнє")
    if not STATE["zone"]:
        raise HTTPException(409, "спочатку зона")
    if STATE["ask"] and STATE["ask"]["stage"] not in ("done", "error"):
        raise HTTPException(409, "попереднє питання ще виконується")
    if STATE["zone"]["sig"]["stage"] == "mining":
        raise HTTPException(409, "SIG ще аналізує зону")
    scenario = str(body.get("scenario") or HANDOFF).upper()
    if scenario not in SCENARIOS:
        raise HTTPException(400, f"scenario: один із {', '.join(SCENARIOS)}")
    STATE["ask"] = {
        "question": question,
        "scenario": scenario,
        "stage": "search",
        "insights": [],
        "ads_report": "",
        "joint_report": "",
        "error": None,
    }
    threading.Thread(target=run_ask, args=(question, scenario), daemon=True).start()
    return STATE["ask"]


# ----------------------------------------------------------------------------- entry


def check() -> None:
    df = pd.DataFrame({"lon": [0.5, 0.2, 2.0], "lat": [0.5, 0.8, 2.0]})
    square = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    assert flag_in_aoi(df, square).tolist() == [1, 1, 0], flag_in_aoi(df, square).tolist()
    # the ADS task: English frame, the question and SIG's text verbatim, O# lines, the guidance; placeholders before the ask
    zone_fx = {
        "polygon": square,
        "base": {"total": 2, "inside": 1, "adjacent": 1, "far": 0},
        "context": {"total": 1, "inside": 0, "adjacent": 0, "far": 1},
        "cells": ["a"],
    }
    data_fx = {"resolution": 9, "counted": ["count_crime_theft"]}
    sig_fx = {
        "sig_answer": "Спостереження: A [P1]",
        "insights": [
            {
                "key": "P1",
                "headline": "H",
                "scope": "s",
                "shift": ["m: up"],
                "inside": True,
                "credibility": 6,
                "filename": BASE["file"],
                "layer": "context_x_band=q2",
            },
            {"key": "P2", "headline": "G", "scope": "", "shift": [], "inside": False, "credibility": None, "filename": "other.csv", "layer": ""},
        ],
    }
    parts = ads_query_parts("як змінювалась кількість?", zone_fx, data_fx, sig_fx)
    assert [p["kind"] for p in parts] == ["frame", "user", "data", "sig_answer", "observations", "guidance"], parts
    q = "".join(p["text"] for p in parts)
    assert q.startswith("Analyse the area of interest (AOI): як змінювалась") and "h3_r9, h3_r8, h3_r7" in q and "inside=1, adjacent=1, far=0" in q, q
    assert (
        f"O1 [P1] H · where: s · shift: m: up · in zone: yes · credibility: no other source · file: {BASE['file']} · cross-layer: context_x_band=q2"
        in q
    ), q
    assert "\nO2 [P2] G" in q and "A finding marked cross-layer" in q and "joining the two tables on the cell id" in q, q
    plain = ads_query("q", zone_fx, data_fx, {"sig_answer": "A", "insights": [{**sig_fx["insights"][1], "inside": True}]})
    assert "A finding marked cross-layer" not in plain and "proximity is not causation" in plain  # the join bullet only with a layer finding
    assert "(other file, context only)" in q and "Спостереження: A [P1]" in q and "proximity is not causation" in q and "h3_r8 whether" in q, q
    assert "(after the ask)" in ads_query("q", None, None, None) and "(after the ask)" not in q
    # the hand-off scenarios (SCENARIOS): what each task carries of SIG
    PATTERN_CELLS["x1"] = ["a", "b"]
    fx = {"sig_answer": "A", "insights": [{**sig_fx["insights"][0], "pattern_id": "x1", "validation": "p < 0.001"}, sig_fx["insights"][1]]}
    kinds = {
        t: [p["kind"] for p in ads_query_parts("q", zone_fx, data_fx, fx, t, "1. step (O1)", ["- sig_o1: c"])]
        for t in ("facts", "columns", "none", "plan")
    }
    assert kinds == {
        "facts": ["primer", "frame", "user", "data", "insights", "guidance"],
        "columns": ["frame", "user", "data", "sig_columns", "guidance"],
        "none": ["frame", "user", "data", "guidance"],
        "plan": ["frame", "user", "data", "observations", "plan", "guidance"],
    }, kinds
    facts = ads_query("q", zone_fx, data_fx, fx, "facts")
    assert facts.startswith("Read this first") and facts.index("Do not verify") < facts.index("Analyse the area of interest (AOI): q"), facts
    assert "O1 · in zone: yes (1 of its 2 cells are zone cells)\n  cells where: s\n  established: m: up\n  checks passed: p < 0.001" in facts, facts
    assert "claims to test" not in facts and "[P1]" not in facts and "Спостереження" not in facts, facts
    blind = ads_query("q", zone_fx, data_fx, fx, "none")
    assert "O1" not in blind and "SIG" not in blind.split("Guidance:")[1], blind
    assert "1. step (O1)" in ads_query_parts("q", zone_fx, data_fx, fx, "plan", "1. step (O1)")[4]["text"]
    csv = {"base_aoi.csv": b"h3_r9,in_aoi\na,1\nc,1\nb,0\n", "context_aoi.csv": b"h3_r9,in_aoi\nb,0\n"}
    flagged, notes = sig_flag_files(csv, fx["insights"], "h3_r9")
    assert flagged["base_aoi.csv"] == b"h3_r9,in_aoi,sig_o1\na,1,1\nc,1,0\nb,0,1\n" and flagged["context_aoi.csv"].endswith(b"b,0,1\n"), flagged
    assert len(notes) == 1 and notes[0].startswith("- sig_o1: cells where s; there m: up") and "1 base rows inside" in notes[0], notes
    PATTERN_CELLS.clear()
    md = aoi_md(zone_fx, data_fx, {"other.csv": ["P1"]})
    assert len(md) <= 2000 and "lisa_points" in md and "other.csv: P1" in md and "h3_r9" in md and "count_crime_theft" in md, len(md)
    assert not any(w in md.lower() for w in ("must", "always", "never", "ignore")), md  # declarative: ADS's guardrail reads it
    dom = domain_md(zone_fx, {"other.csv": ["P1"]})
    assert "lisa_points" in dom and "other.csv: P1, інший датасет" in dom and "питанням користувача" in dom, dom
    assert (DEMO_DIR / BASE["file"]).is_file(), f"demo data missing: {DEMO_DIR}"
    prompt = Template(JOINT_TEMPLATE.read_text(encoding="utf-8")).render(
        user_query="q",
        report="R",
        insight_analysis="A [P1]",
        insight_evidence="E",
        observations="O1 [P1] H",
        domain_knowledge="D",
        query_language="uk",
    )
    for tag in (
        "<Insight_Analysis>",
        "<Insight_Evidence>",
        "<Insight_Observations>",
        "<insight_integration>",
        "<Analytical_Report>",
        "<User_Domain_Knowledge>",
    ):
        assert tag in prompt, tag
    assert "O1 [P1] H" in prompt and "area only, never" not in prompt, "the prompt says SIG got the query and the area"
    assert "SIG's hypotheses as claims to test" in prompt
    render = Template(JOINT_TEMPLATE.read_text(encoding="utf-8")).render
    late = render(user_query="q", report="R", insight_analysis="A", insight_evidence="E", ads_saw_sig=False, query_language="uk")
    assert "ADS did not see SIG's findings" in late and "claims to test" not in late
    alone = render(user_query="q", report="R", insight_analysis="", insight_evidence="", observations="", domain_knowledge="D", query_language="uk")
    assert "<insight_integration>" not in alone and "\n<Insight_Analysis>\n" not in alone and "<Analytical_Report>" in alone  # the ADS report alone
    # a cell package of two cells, offsets in 10**-1 degrees: a unit square at (0, 0) and one at (5, 5)
    grid = cell_grid({"scale": 1, "columns": {"h3": ["a", "b"]}, "rings": [[0, 0, 10, 0, 10, 10, 0, 10], [5, 5, 10, 0, 10, 10, 0, 10]]})
    assert grid["rings"][0] == [[0, 0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0, 0]], grid["rings"]
    # SIG's credibility order (1 confirmed, 2 unconfirmed, 6 no other source, 5 contradicted), the zone's items first
    items = [{"key": f"P{i}", "pattern_id": "x", "credibility": c, "scope": []} for i, c in enumerate((5, 6, 2, 1, 1), 1)]
    items[1]["scope"] = ["context_bars_band=q2", "lisa_points=no cluster"]  # a cross-layer finding: tagged, not promoted
    found = sig_findings({"view": {"evidence": {"items": items, "area": {"P1": 0.4, "P2": 0.4, "P3": 0.4, "P4": 0.4}, "provenance": []}}})
    assert [it["key"] for it in found] == ["P4", "P3", "P2", "P1", "P5"], [it["key"] for it in found]
    assert found[2]["layer"] == "context_bars_band=q2" and found[0]["layer"] == ""
    # geo.py: SIG's own indexing, parents consistent, a cell's own boundary polygon is exactly that cell, the rows' place
    cell = h3.latlng_to_cell(41.885, -87.625, 9)
    e = enrich_h3(pd.DataFrame({"lon": [-87.625, -87.6251, None], "lat": [41.885, 41.885, 41.9]}), 9)
    assert e["h3_r9"].tolist() == [cell, cell, ""] and e["h3_r7"].tolist()[2] == "", e
    assert all(h3.cell_to_parent(c, 8) == p for c, p in zip(e["h3_r9"][:2], e["h3_r8"][:2], strict=True)), e
    ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(cell)]
    assert zone_h3_cells({"type": "Polygon", "coordinates": [ring + ring[:1]]}, 9, [cell, h3.grid_ring(cell, 1)[0]]) == [cell]
    assert zone_h3_cells({"type": "Polygon", "coordinates": [[[-87.625, 41.885]] * 4]}, 9, [cell]) == []  # H3 refuses it: no cells, 400
    wrapped = [[lng + 360 if lng < 0 else lng, lat] for lng, lat in ring]  # a longitude past 180: folded back as SIG folds it
    assert zone_h3_cells({"type": "Polygon", "coordinates": [wrapped + wrapped[:1]]}, 9, [cell]) == [cell]
    nb, far = h3.grid_ring(cell, 1)[0], h3.latlng_to_cell(41.95, -87.70, 9)
    rows = pd.DataFrame({"in_aoi": [1, 0, 0, 0, 0], "h3_r9": [cell, cell, nb, far, ""]})
    assert aoi_zone(rows, [cell], "h3_r9").tolist() == ["inside", "adjacent", "adjacent", "far", "far"]  # the same cell is not inside
    # a zone's tables are copies: the build's frames stay as they were, and a second zone cannot touch the first's bytes
    DATA.update({"base": e, "context": e})
    first = {
        n: f.to_csv(index=False).encode() for n, f in zone_frames({"type": "Polygon", "coordinates": [ring + ring[:1]]}, [cell], "h3_r9")[0].items()
    }
    zone_frames(square, [], "h3_r9")
    assert (
        "in_aoi" not in DATA["base"].columns
        and first["base"] == DATA["base"].assign(in_aoi=[1, 1, 0], aoi_zone=["inside", "inside", "far"]).to_csv(index=False).encode()
    )
    DATA.clear()
    assert base_spec(["theft", "assault"])["count_values"] == "crime=theft,crime=assault" and "category" not in base_spec([])
    print("ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if "check" in sys.argv[1:]:
        check()
    else:
        import uvicorn

        uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8090")))
