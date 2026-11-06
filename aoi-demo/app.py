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

import httpx
import pandas as pd
import shapely
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from geo import aoi_zone, enrich_h3, h3_columns, zone_h3_cells
from jinja2 import Template
from shapely.geometry import shape

log = logging.getLogger("aoi")
HERE = Path(__file__).resolve().parent

SIG_URL = os.environ.get("SIG_URL", "http://127.0.0.1:8765")
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

http = httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))

# ponytail: single-user in-memory state; a second browser tab shares it, per-session state when a second user needs their
# own zone. DATA holds the frames as the build left them, FILES the zone's tables taken when the polygon closed (what the
# ask uploads, so a redraw cannot change a running ask), STATE what the page polls (JSON only)
STATE: dict[str, Any] = {"datasets": None, "zone": None, "ask": None}
DATA: dict[str, pd.DataFrame] = {}
FILES: dict[str, bytes] = {}
GRID: dict[str, Any] = {}  # the base layer's map cells in SIG: h3 ids, rings, centres, and SIG's resolution


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


def sig_chat(cells: list[str], question: str) -> dict[str, Any]:
    """SIG's reading of the area exactly as its console's chat asks it: the selected map cells as the zone, with the
    user's question or without one (the console's zone-only ask: "no text: the map area below is the question"). With
    both, the graph scores its seeds by the text and the place together; its LLM then reads the evidence by the
    project's rules: observations, then hypotheses."""
    r = http.post(f"{SIG_URL}/api/chat/query", json={"question": question, "cells": cells, "use_llm": True})
    if r.status_code >= 300:
        raise RuntimeError(f"SIG chat: {r.status_code} {r.text[:300]}")
    return r.json()


def sig_rings(dataset_id: str, pattern_id: str) -> list[list[list[float]]]:
    r = http.get(f"{SIG_URL}/api/geo/{dataset_id}/patterns/{pattern_id}", params={"rings": "true"})
    return r.json().get("rings", []) if r.status_code < 300 else []


CRED_RANK = {1: 0, 2: 1, 6: 2, 5: 3}  # SIG's credibility, best first: confirmed, unconfirmed, no other source, contradicted


def sig_findings(qa: dict[str, Any]) -> list[dict[str, Any]]:
    """The chat's evidence items with their cells' rings for the map: the ones the zone weighed as inside (place > 0)
    first, the most credible first, then the ones outside it that SIG's answer also tells; at most MAX_INSIGHTS."""
    evidence = (qa.get("view") or {}).get("evidence") or {}
    area = evidence.get("area") or {}
    source = {p["key"]: p for p in evidence.get("provenance") or []}  # key -> dataset_id, filename
    items = sorted(evidence.get("items", []), key=lambda it: (-((area.get(it["key"]) or 0) > 0), CRED_RANK.get(it.get("credibility"), 9)))
    out = []
    for it in items[:MAX_INSIGHTS]:  # SIG's own text, in its own language, as is
        dataset_id = (source.get(it["key"]) or {}).get("dataset_id", "")
        out.append(
            {
                "key": it["key"],
                "pattern_id": it["pattern_id"],
                "dataset_id": dataset_id,
                "filename": (source.get(it["key"]) or {}).get("filename", ""),
                "role": it.get("role", ""),
                "inside": (area.get(it["key"]) or 0) > 0,
                "headline": it.get("headline", ""),
                "scope": it.get("scope_text", ""),
                "shift": it.get("shift_text") or [],
                "validation": it.get("validation", ""),
                "trust": it.get("trust", ""),
                "credibility": it.get("credibility"),
                "place": area.get(it["key"]),
                "rings": sig_rings(dataset_id, it["pattern_id"]) if dataset_id else [],
            }
        )
    return out


def sig_zone_answer(cells: list[str], question: str) -> dict[str, Any]:
    """One SIG reading of the zone, with the user's question or without it: the findings for the map, SIG's answer, the
    evidence its model read, the files the findings come from, and the call's trace — the same shape for the preview on
    polygon close (no question) and for the ask (the question), so the ask can fall back on the preview."""
    t0 = time.time()
    trace = {"call": f"POST {SIG_URL}/api/chat/query", "question": question, "cells_count": len(cells)}
    qa = sig_chat(cells, question)
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


def aoi_md(zone: dict[str, Any], files: dict[str, list[str]] | None = None) -> str:
    coords = zone["polygon"]["coordinates"][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    b, c = zone["base"], zone["context"]
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
            f"- Зона задана користувачем як полігон на мапі: {len(coords) - 1} вершин, "
            f"довгота {min(lons):.4f}…{max(lons):.4f}, широта {min(lats):.4f}…{max(lats):.4f}.",
            f"- Файл base_aoi.csv: базовий шар подій, файл {BASE['file']} ({BASE['label']}), {b['total']} рядків, з них {b['inside']} у зоні.",
            f"- Файл context_aoi.csv: контекстний шар, файл {CONTEXT['file']} ({CONTEXT['label']}), {c['total']} рядків, з них {c['inside']} у зоні.",
            "- Колонка in_aoi в обох файлах: 1 означає, що точка лежить усередині полігона, 0 поза ним. "
            "Обчислено детерміновано (point-in-polygon), без моделі.",
            "- Рядки з in_aoi = 0 є базовою лінією для порівняння з зоною.",
            f"- Граф інсайтів SIG побудовано з тих самих шарів: базовий шар розбито на комірки H3, контекстний шар {CONTEXT['label']} "
            "враховано як окрему ознаку комірки в крос-прогоні.",
            f"- Зону передано в SIG так, як її задає виділення комірок у консолі SIG: {len(zone.get('cells') or [])} комірок H3 "
            "базового шару, чиї центри лежать у полігоні. Питання користувача в SIG не передавали: SIG описав, що відбувається "
            "у виділеній зоні загалом. Від цих комірок ретривер SIG увійшов у граф, обійшов його і зібрав знахідки в зоні та "
            "пов'язані з ними; мовна модель SIG описала спостереження і гіпотези. Ця відповідь у частині «--- insights.md ---».",
            "",
            *(sources if by_file else []),
            SIG_GLOSSARY,
        ]
    )


def insights_body(items: list[dict[str, Any]]) -> str:
    if not items:
        return "У зоні інтересу граф SIG не має знахідок, які пройшли статистичну перевірку.\n"
    lines = []  # SIG's fields verbatim; the joint prompt reads them (its <insight_integration> glossary)
    for n, it in enumerate(items, 1):
        lines.append(f"{n}. [{it['key']}] {it['headline']}")
        for label, value in (
            ("scope", it["scope"]),
            ("shifts", "; ".join(it["shift"])),
            ("validation", it["validation"]),
            ("credibility", "" if it["credibility"] is None else str(it["credibility"])),
            ("trust", it["trust"]),
        ):
            if value:
                lines.append(f"   - {label}: {value}")
    return "\n".join(lines) + "\n"


def insights_md(sig_answer: str, items: list[dict[str, Any]]) -> str:
    return (
        "# Аналіз графа інсайтів SIG для зони інтересу\n\n"
        "Нижче наведено опис виділеної зони від сервісу інсайтів SIG, якому передали лише комірки зони, без питання: "
        "спостереження і гіпотези з ключами знахідок [P#], а потім знахідки в зоні дослівно. Кожна знахідка пройшла статистичну перевірку "
        "(bootstrap, поправка на множинні порівняння). Пояснення термінів наведено в частині «--- aoi.md ---». Це довідник фактів "
        "про зону: значення колонок у рядках scope знахідок можна брати як розрізи власного аналізу.\n\n"
        "## Відповідь SIG\n\n" + sig_answer.strip() + "\n\n## Знахідки SIG у зоні, дослівно\n\n" + insights_body([it for it in items if it["inside"]])
    )


def ads_query_parts(question: str, enrich: bool) -> list[dict[str, str]]:
    """The user's question, framed deterministically, as the parts the page shows apart: the area of interest, the
    user's own words, the data, and — when SIG found anything in the zone — its findings' values as extra slices.
    Joined, they are the ADS query."""
    parts = [
        {"kind": "frame", "text": "Проаналізуй у зоні інтересу: "},
        {"kind": "user", "text": question},
        {
            "kind": "data",
            "text": "\n\nДані: base_aoi.csv (базовий шар подій) і context_aoi.csv (контекстний шар). В обох колонка in_aoi "
            "позначає зону інтересу: 1 всередині полігона, 0 поза ним. Питання стосується рядків in_aoi = 1; решта є базовою "
            "лінією для порівняння. Методика зони і пояснення термінів наведено в блоці User_Domain_Knowledge, у частині, "
            "що починається рядком «--- aoi.md ---».",
        },
    ]
    if enrich:
        # Fixed text, the same for any data and question, keyed to SIG's findings in the zone (not to how its answer is
        # worded). ADS reads every uploaded .md inside its <User_Domain_Knowledge> block under a "--- <file> ---" line
        # (profiler_script.py), so the reference names exactly that line and the section of the findings.
        parts.append(
            {
                "kind": "sig",
                "text": "\n\nДодаткові розрізи. Дай відповідь на питання. Значення колонок зі знахідок SIG у зоні (блок "
                "User_Domain_Knowledge, частина «--- insights.md ---», розділ «Знахідки SIG у зоні, дослівно», рядки scope) "
                "візьми як додаткові розрізи цього ж розрахунку: для кожного покажи показник із питання в зоні порівняно з рештою "
                "даних. Не переказуй знахідки [P#] і не рахуй просторові кластери заново; значення, якого в даних немає, пропусти.",
            }
        )
    return parts


def ads_query(question: str, enrich: bool) -> str:
    return "".join(p["text"] for p in ads_query_parts(question, enrich))


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


def llm_chat(prompt: str) -> dict[str, Any]:
    body = {"model": "synthesis", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": JOINT_MAX_TOKENS}
    r = http.post(f"{LLM_URL}/v1/chat/completions", json=body)
    if r.status_code >= 300:
        raise RuntimeError(f"LLM gateway: {r.status_code} {r.text[:300]}")
    return r.json()


def joint_report(question: str, report: str, sig_answer: str, sig_evidence: str, aoi: str) -> tuple[str, dict[str, Any]]:
    """Copilot's final response prompt adapted for two sources (prompts/joint_response.jinja): the ADS report, SIG's
    answer and the graph evidence it read, verbatim; the zone's methodology and the SIG glossary as domain knowledge.
    -> (the report, "" without its required sections; the call's trace)."""
    template = Template(JOINT_TEMPLATE.read_text(encoding="utf-8"))
    prompt = template.render(
        user_query=question, report=report, insight_analysis=sig_answer, insight_evidence=sig_evidence, domain_knowledge=aoi, query_language="uk"
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
            "User_Domain_Knowledge: aoi.md": len(aoi),
        },
        "prompt_chars": len(prompt),
        "usage": resp.get("usage"),
        "answer_chars": len(out),
    }
    return (out if "## Результати аналізу" in out and "## Методологія" in out else ""), trace


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
        s.update(sig_zone_answer(z["cells"], ""))
        s["aoi_md"], s["insights_md"] = aoi_md(z, s["files"]), insights_md(s["sig_answer"], s["insights"])
        s["enrich"] = in_zone(s)  # step 2 is enriched whenever the place has findings
        s["stage"] = "ready"
    except Exception as exc:  # the question can still go to ADS, unenriched
        log.exception("zone preview failed")
        s["stage"], s["error"] = "error", f"{type(exc).__name__}: {exc}"
        s.setdefault("aoi_md", aoi_md(z))
        s.setdefault("seconds", 0)
        s.update({"insights": [], "sig_answer": "", "sig_evidence": "", "insights_md": "", "enrich": False, "files": {}})


def run_ask(question: str) -> None:
    a, zone = STATE["ask"], STATE["zone"]
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
        combined = sig_zone_answer(zone["cells"], question)  # the question and the zone together: the graph scores both
        fallback = not in_zone(combined) and in_zone(preview)  # the text pulled the seeds out of the zone: the preview stands
        chosen = preview if fallback else combined
        a["sig_entry"] = "preview" if fallback else "combined"
        for k in ("insights", "sig_answer", "sig_evidence", "sig_mode", "sig_model", "sig_sources"):
            a[k] = chosen.get(k)
        a["aoi_md"], a["insights_md"] = aoi_md(zone, chosen.get("files") or {}), insights_md(a["sig_answer"], a["insights"])
        trace["search"] = {
            **chosen.get("trace", {}),
            "entry": a["sig_entry"],
            "combined": combined["trace"].get("result"),
            "preview": preview.get("trace", {}).get("result"),
        }
        mark("ads")
        sid = ads_session()
        a["ads_session"] = sid
        parts = ads_query_parts(question, in_zone(chosen))
        a["ads_query"] = "".join(p["text"] for p in parts)
        files = {
            **FILES,  # the zone's tables as taken when the polygon closed
            "aoi.md": a["aoi_md"].encode("utf-8"),
            **({"insights.md": a["insights_md"].encode("utf-8")} if a["insights_md"] else {}),
        }
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
        trace["ads"]["result"] = {
            "outcome": f"{a['ads_outcome']} · {last.get('completion_status')}",
            "кроків": counters.get("total_steps_taken"),
            "звіт": f"{len(a['ads_report'])} символів",
            "файли ADS": ", ".join(o.get("filename", "") for o in last.get("outputs") or []) or "—",
        }
        http.post(f"{ADS_URL}/sessions/{sid}/close")
        if not a["ads_report"]:
            raise RuntimeError(f"ADS returned no report (outcome {a['ads_outcome']})")
        mark("joint")
        a["joint_report"], trace["joint"] = joint_report(question, a["ads_report"], a["sig_answer"], a["sig_evidence"], a["aoi_md"])
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
    return STATE


@app.get("/ads-query")
def ads_query_preview(question: str = "") -> dict[str, Any]:
    """The ADS task for this question, part by part, as the run will send it. Exact once SIG has mined the drawn zone;
    before that the SIG slices are shown as pending (they join when the place has findings)."""
    s = (STATE["zone"] or {}).get("sig") or {}
    exact = s.get("stage") in ("ready", "error")
    return {"parts": ads_query_parts(question.strip(), s.get("enrich", False) if exact else True), "exact": exact}


@app.get("/trace/prompt")
def trace_prompt() -> PlainTextResponse:
    """The last joint prompt as sent to the LLM: long, so the page asks for it only when opened."""
    return PlainTextResponse(PROMPTS.get("joint", ""))


@app.get("/datasets/options")
def datasets_options() -> dict[str, Any]:
    """The base file's category column and its values with located rows: what the page offers to count per cell."""
    return {"column": BASE["category"], "values": category_options(), "min_rows": COUNT_MIN, "default": list(COUNT_DEFAULT)}


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
    polygon = body.get("polygon") or {}
    if polygon.get("type") != "Polygon" or len((polygon.get("coordinates") or [[]])[0]) < 4:
        raise HTTPException(400, "polygon: GeoJSON Polygon з щонайменше трьома вершинами")
    cells = zone_h3_cells(polygon, GRID["resolution"], GRID["h3"])  # the zone as SIG reads a geometry
    if not cells:
        raise HTTPException(400, "у полігоні немає жодної комірки мапи SIG: обведіть ділянку з подіями")
    frames, counts = zone_frames(polygon, cells, h3_columns(GRID["resolution"])[0])
    FILES.clear()
    FILES.update({f"{name}_aoi.csv": f.to_csv(index=False).encode("utf-8") for name, f in frames.items()})
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
    STATE["ask"] = {"question": question, "stage": "search", "insights": [], "ads_report": "", "joint_report": "", "error": None}
    threading.Thread(target=run_ask, args=(question,), daemon=True).start()
    return STATE["ask"]


# ----------------------------------------------------------------------------- entry


def check() -> None:
    df = pd.DataFrame({"lon": [0.5, 0.2, 2.0], "lat": [0.5, 0.8, 2.0]})
    square = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    assert flag_in_aoi(df, square).tolist() == [1, 1, 0], flag_in_aoi(df, square).tolist()
    q = ads_query("як змінювалась кількість?", enrich=True)
    assert q.startswith("Проаналізуй у зоні інтересу: як змінювалась") and "in_aoi" in q, q
    assert "«--- insights.md ---»" in q and "Знахідки SIG у зоні, дослівно" in q and "Не переказуй" in q, q  # slices, not an essay
    sig_block = ads_query_parts("q", enrich=True)[-1]["text"].lower()
    assert not any(w in sig_block for w in ("злочин", "бар", "район", "поясн")), "the SIG block: generic, no explaining"
    assert "insights.md" not in ads_query("q", enrich=False)  # no findings in the place: no enrichment
    assert "Знахідки SIG у зоні, дослівно" in insights_md("A", []) and "пояснити" not in insights_md("A", [])
    assert (DEMO_DIR / BASE["file"]).is_file(), f"demo data missing: {DEMO_DIR}"
    prompt = Template(JOINT_TEMPLATE.read_text(encoding="utf-8")).render(
        user_query="q", report="R", insight_analysis="A [P1]", insight_evidence="E", domain_knowledge="D", query_language="uk"
    )
    for tag in ("<Insight_Analysis>", "<Insight_Evidence>", "<insight_integration>", "<Analytical_Report>", "<User_Domain_Knowledge>"):
        assert tag in prompt, tag
    md = aoi_md({"polygon": square, "base": {"inside": 1, "total": 2}, "context": {"inside": 0, "total": 1}}, {"other.csv": ["P1"]})
    assert "lisa_points" in md and "other.csv: P1, інший датасет" in md, md
    # a cell package of two cells, offsets in 10**-1 degrees: a unit square at (0, 0) and one at (5, 5)
    grid = cell_grid({"scale": 1, "columns": {"h3": ["a", "b"]}, "rings": [[0, 0, 10, 0, 10, 10, 0, 10], [5, 5, 10, 0, 10, 10, 0, 10]]})
    assert grid["rings"][0] == [[0, 0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0, 0]], grid["rings"]
    # SIG's credibility order (1 confirmed, 2 unconfirmed, 6 no other source, 5 contradicted), the zone's items first
    items = [{"key": f"P{i}", "pattern_id": "x", "credibility": c} for i, c in enumerate((5, 6, 2, 1, 1), 1)]
    found = sig_findings({"view": {"evidence": {"items": items, "area": {"P1": 0.4, "P2": 0.4, "P3": 0.4, "P4": 0.4}, "provenance": []}}})
    assert [it["key"] for it in found] == ["P4", "P3", "P2", "P1", "P5"], [it["key"] for it in found]
    # geo.py: SIG's own indexing, parents consistent, a cell's own boundary polygon is exactly that cell, the rows' place
    import h3

    cell = h3.latlng_to_cell(41.885, -87.625, 9)
    e = enrich_h3(pd.DataFrame({"lon": [-87.625, -87.6251, None], "lat": [41.885, 41.885, 41.9]}), 9)
    assert e["h3_r9"].tolist() == [cell, cell, ""] and e["h3_r7"].tolist()[2] == "", e
    assert all(h3.cell_to_parent(c, 8) == p for c, p in zip(e["h3_r9"][:2], e["h3_r8"][:2], strict=True)), e
    ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(cell)]
    assert zone_h3_cells({"type": "Polygon", "coordinates": [ring + ring[:1]]}, 9, [cell, h3.grid_ring(cell, 1)[0]]) == [cell]
    nb, far = h3.grid_ring(cell, 1)[0], h3.latlng_to_cell(41.95, -87.70, 9)
    rows = pd.DataFrame({"in_aoi": [1, 0, 0, 0, 0], "h3_r9": [cell, cell, nb, far, ""]})
    assert aoi_zone(rows, [cell], "h3_r9").tolist() == ["inside", "adjacent", "adjacent", "far", "far"]  # the same cell is not inside
    assert base_spec(["theft", "assault"])["count_values"] == "crime=theft,crime=assault" and "category" not in base_spec([])
    print("ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if "check" in sys.argv[1:]:
        check()
    else:
        import uvicorn

        uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8090")))
