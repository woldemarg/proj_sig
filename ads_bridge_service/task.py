"""The texts of an ADS task: the zone's insights as established facts, the tables' roles and columns, the guidance, the
column dictionary ADS reads as domain knowledge and the final report's methodology. Pure functions, no I/O."""

from __future__ import annotations

from typing import Any

MAX_INSIGHTS = 10  # the graph's evidence holds about ten
CRED_RANK = {1: 0, 2: 1, 6: 2, 5: 3}  # credibility, best first: confirmed, unconfirmed, no other source, contradicted

SIG_TERMS = (
    "points = events in the cell; share_<column>_<value> = share of the cell's events with that value; count_<column>_<value> = "
    "their number; <column>_mode = the cell's most frequent value; neighbours_<counter>_band q1…q4 = quartile of the counter summed over the six "
    "neighbours; lisa_points = local Moran class of event counts (hot spot, cold spot, high outlier, low outlier, no cluster); "
    "context_<layer>_band = none nearby, or the quantile of context points in the cell; shift = the subgroup's median against the map's median "
    "in robust sd"
)

# What an insight is and how to read it, ahead of the user's question (the query is the part ADS never trims)
INSIGHT_PRIMER = f"""Read this first: established insights about this dataset.

Before this task a separate statistical system (SIG) mined these very tables and validated what it found (bootstrap stability and a multiple-testing correction). Each insight O# below is a result already computed and validated for this dataset, not a suggestion.

How to read an insight:
- `cells where` is the condition that selects a set of H3 map cells at the insight's own H3 level (for the base layer: the finest h3_r column); SIG's row is a cell with its events, not an event.
- `established` is how a per-cell metric in those cells differs from the whole map: the subgroup's median against the map's median, in robust standard deviations (sd).
- `in zone` says whether those cells overlap the user's area of interest.
- `trust` is how far the insight's source is confirmed: the source's reliability, whether another source confirms or contradicts the insight (credibility), and the record's integrity. It is separate from the statistics.
- SIG's terms: {SIG_TERMS}.

Rules for the insights:
1. They are computed and validated. Do not verify, test, reproduce, re-derive or recompute an insight, its medians, its sd or its cell condition; no step of your plan may have that purpose. The per-cell features behind them (neighbour bands, LISA classes, per-cell shares) are not columns of the tables, so a recomputation would only produce a different, wrong number.
2. Use them as navigation: they tell you which event types, places and conditions deserve attention. Spend your steps on what the question asks and the insights do not already state.
3. In the report cite an insight by its id and quote its numbers exactly as given, next to your own figures. An insight whose `trust` line says another source contradicts it is contested: say so when you cite it.

"""

GUIDANCE = (
    "\n\nGuidance:\n- Answer the question with your own computations on the tables. Where the question points to it, compare inside vs "
    "adjacent vs far at the same level (rows per cell for counts, shares for categories).\n- The insights are computed: quote those that "
    "bear on the question by id with their numbers, and add what they do not state (dates, counts in the zone, breakdowns).\n"
    "- End the report with one section 'Established insights'. List only the insights that bear on the question (their condition or "
    "metric is what the question asks about), the ones in the zone first. For each give its id and whether it is in the zone, copy its "
    "'established' lines word for word (a median pair reads 'median a vs b overall'; sd is the unit of the shift, not of a median), and "
    "add one sentence on what it adds to your own figures. When no insight bears on the question, say so in one sentence. An insight "
    "with 'in zone: no' describes other places: say so, never present it as a fact about the AOI.\n"
    "- Report figures for groups of cells (inside, adjacent, far; with and without layer points), not tables of raw cell ids.\n"
    "- Describe differences as shifts and ratios, not causes; proximity is not causation."
)

# What the graph's own words mean, for the final report
SIG_GLOSSARY = """## Словник термінів графа інсайтів SIG

Знахідки SIG записані мовою графа, англійською. Терміни означають таке.

- Комірка мапи (H3): шестикутна комірка сітки H3. У SIG рядок даних є коміркою з подіями, а не окремою подією. Підгрупа (`scope`, subgroup) є набором комірок, що відповідають умовам `<колонка> is <значення>`.
- `points`: кількість подій у комірці.
- `share <колонка> <значення>`: частка подій комірки з цим значенням колонки.
- `<колонка>_mode`: найчастіше значення колонки серед подій комірки.
- `neighbours_<лічильник>_band is q1…q4`: наскільки жваві шість сусідніх комірок за цим лічильником, у чвертях усіх комірок. `q1` найтихіша чверть, `q4` найжвавіша. Сама комірка тут не враховується.
- `lisa_points`: локальний кластер LISA (локальний індекс Морана), тобто чи утворюють комірка і її сусіди скупчення за кількістю подій.
  - `hot spot`: гаряча точка, жвава комірка серед жвавих сусідів.
  - `cold spot`: холодна точка, тиха комірка серед тихих сусідів.
  - `high outlier`: жвава комірка серед тихих сусідів.
  - `low outlier`: тиха комірка серед жвавих сусідів.
  - `no cluster`: комірка не входить ні в жваве, ні в тихе скупчення. Це звичайне місце без вираженого скупчення, а не відсутність подій.
- `context_<шар>_band`: скільки точок контекстного шару в самій комірці. `none nearby` означає жодної, `q1…q4` чверті серед комірок, де вони є.
- Зсув `<метрика>: mild|moderate|strong|extreme increase|decrease, ±z sd (median a vs b overall)`: медіана метрики в комірках підгрупи (a) проти медіани всієї мапи (b). `sd` є робастним стандартним відхиленням, тобто різницею медіан, поділеною на MAD. Слова сили: mild означає слабкий, moderate помірний, strong сильний, extreme надзвичайно сильний.
- `bootstrap stability`: частка повторних вибірок, у яких підгрупа відтворюється; 1.0 означає завжди.
- `adjusted p`: p-значення після поправки на множинні порівняння; що менше, то менш імовірно, що знахідка випадкова.
- `credibility`, рівень довіри: 1 підтверджено іншим джерелом; 2 не підтверджено, бо інше джерело має події в цьому місці, але такої знахідки не має; 5 суперечить іншому джерелу; 6 немає іншого джерела. Це окремо від статистичної перевірки: сильне p з одного джерела все одно не є підтвердженням.
- `source F`: надійність джерела не оцінена тим, хто завантажив дані.
- Граф SIG спільний для всіх завантажених датасетів. Знахідка з іншого файлу є контекстом, а не фактом про базовий шар цього аналізу.
"""


def name_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The picked datasets with the file each goes to ADS as: the first the base table, the rest its context layers."""
    layers = len(tables) - 1
    return [t | {"file": "base_aoi.csv" if n == 0 else "context_aoi.csv" if layers == 1 else f"context{n}_aoi.csv"} for n, t in enumerate(tables)]


def select_findings(qa: dict[str, Any]) -> list[dict[str, Any]]:
    """The answer's evidence items for the task: the ones the zone weighed as inside (place > 0) first, the most
    credible first, then the ones outside it the answer also tells; at most MAX_INSIGHTS, in the graph's own words. An
    item folded into another (the same finding at another level of its upload) is told by its host alone, as the
    graph's own prompt tells it."""
    evidence = (qa.get("view") or {}).get("evidence") or {}
    area = evidence.get("area") or {}
    source = {p["key"]: p for p in evidence.get("provenance") or []}
    shown = [it for it in evidence.get("items", []) if not it.get("folded_into")]
    items = sorted(shown, key=lambda it: (-((area.get(it["key"]) or 0) > 0), CRED_RANK.get(it.get("credibility"), 9)))

    def level(it: dict[str, Any]) -> str:
        resolution = ((it.get("provenance") or {}).get("spatial") or {}).get("resolution")
        return "the table level" if resolution is None else f"H3 resolution {resolution}"

    return [
        {
            "key": it["key"],
            "pattern_id": it["pattern_id"],
            "dataset_id": (it.get("provenance") or {}).get("dataset_id") or (source.get(it["key"]) or {}).get("dataset_id", ""),
            "level": level(it),
            "filename": (source.get(it["key"]) or {}).get("filename", ""),
            "inside": (area.get(it["key"]) or 0) > 0,
            "layer": next((c for c in it.get("scope") or [] if c.startswith("context_")), ""),  # a cross-layer condition
            "headline": it.get("headline", ""),
            "scope": it.get("scope_text", ""),
            "shift": it.get("shift_text") or [],
            "validation": it.get("validation", ""),
            "trust": it.get("trust", ""),
        }
        for it in items[:MAX_INSIGHTS]
    ]


def finding_files(findings: list[dict[str, Any]]) -> dict[str, list[str]]:
    """file -> the O# ids, as the task numbers them, of the findings that come from it."""
    out: dict[str, list[str]] = {}
    for n, it in enumerate(findings, 1):
        out.setdefault(it["filename"] or "?", []).append(f"O{n}")
    return out


def insight_blocks(findings: list[dict[str, Any]], base: dict[str, Any]) -> list[str]:
    """One block per insight: the id and its place, the cell condition, each shift, its validation and its trust; an
    insight of another dataset than the base table's says which: another level of its file, or another file."""
    out = []
    for n, it in enumerate(findings, 1):
        lines = [f"O{n} · in zone: {'yes' if it['inside'] else 'no'}", f"  cells where: {it['scope'] or '—'}"]
        lines += [f"  established: {s}" for s in it["shift"]]
        lines += [f"  {k}: {it[k]}" for k in ("validation", "trust") if it.get(k)]
        if it.get("layer"):
            lines.append(f"  layers: links the base layer's events to the context layer's points in the cell ({it['layer']})")
        if it["filename"] != base["filename"]:
            lines.append(f"  file: {it['filename'] or '?'} (another dataset of the graph: context only, not a fact about the base layer)")
        elif it["dataset_id"] != base["dataset_id"]:
            lines.append(f"  level: {it['level']} of the base table's file (its cells are of that level, not of the base table's h3_r columns)")
        out.append("\n".join(lines))
    return out


def _table_line(t: dict[str, Any]) -> str:
    counted = f", {t['total']} rows, {t['inside']} inside the AOI" if "total" in t else ""
    part = f", only its rows where {t['where']}" if t.get("where") else ""  # a filtered reading of the file
    return f"df_{t['file'].removesuffix('.csv')} ({t['file']}: {t['filename']}{part}{counted})"


def build_query_parts(question: str, tables: list[dict[str, Any]], findings: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """The ADS task, part by part, as the console shows it: the insight primer, the frame, the user's words, the tables
    with their geo columns and zone counts, the insights as computed results, and the guidance. The frame is English
    (ADS's own prose is; Cyrillic costs its budgets more), the question stays verbatim. ``tables`` without counts and
    ``findings`` None are the preview before the ask. Joined, the parts are the query."""
    base, layers = tables[0], tables[1:]
    data = (
        f"\n\nData and roles:\n- {_table_line(base)} is the main table, the subject of the analysis: one row is one event. Every count, "
        "share and trend the question asks for is computed on its rows."
    )
    for t in layers:
        data += f"\n- {_table_line(t)} is a context layer of the main table: one row is one point of another kind."
    if layers:
        data += (
            "\nA context layer adds no events and no metrics of its own; it describes the places of the main table's events. The tables "
            "share no key but the H3 cell ids: relate them by counting the layer's points per cell and joining on the cell id, then compare "
            "the main table's per-cell figures between cells with and without layer points (never sum the layer's count over event rows), "
            "and tell the result from the main table's side (the layer is the setting, not the result)."
        )
    if base.get("time"):
        data += (
            f"\n- Time: {base['time']} in df_base_aoi is the event's timestamp; for figures per day, week or month group by its calendar "
            "date, never by the raw value."
        )
    cells = "; ".join(f"{t['file']}: {', '.join(c for c in t['columns'] if c.startswith('h3_r'))}" for t in tables if t.get("columns"))
    data += (
        f"\nGeo columns: h3_r<resolution> are H3 cell ids (strings){f' ({cells})' if cells else ''}; a coarser id is the parent of the finer "
        "one; group by them, never edit them. in_aoi is 1 for points inside the polygon. aoi_zone is inside (= in_aoi 1) | adjacent "
        "(outside the polygon but in the zone's cells or their ring-1 neighbours) | far. A point in the same cell as the zone is not inside "
        "the zone. The AOI is the inside rows only; adjacent and far rows are comparison groups, never part of the AOI's own figures."
    )
    if "total" in base:
        data += (
            f"\nZone (base rows): inside={base['inside']}, adjacent={base['adjacent']}, far={base['far']}; cells in the zone: {base['zone_cells']}."
        )
    if findings is None:
        blocks = "(after the ask)"
    else:
        blocks = "\n\n".join(insight_blocks(findings, base)) or "SIG has no insights for this zone."
    return [
        {"kind": "primer", "text": INSIGHT_PRIMER},
        {"kind": "frame", "text": "Analyse the area of interest (AOI): "},
        {"kind": "user", "text": question},
        {"kind": "data", "text": data},
        {"kind": "insights", "text": "\n\nEstablished insights (computed and validated; do not recompute):\n" + blocks},
        {"kind": "guidance", "text": GUIDANCE},
    ]


def _zone_span(polygon: dict[str, Any]) -> tuple[int, str, str]:
    ring = polygon["coordinates"][0]
    west = (ring[0][0] + 180) % 360 - 180  # read around the first vertex, itself in range: a zone across 180° stays one span, its far end past ±180°
    lons, lats = [west + (c[0] - ring[0][0] + 540) % 360 - 180 for c in ring], [c[1] for c in ring]
    return len(ring) - 1, f"{min(lons):.4f}…{max(lons):.4f}", f"{min(lats):.4f}…{max(lats):.4f}"


def build_aoi_md(polygon: dict[str, Any], tables: list[dict[str, Any]], files: dict[str, list[str]]) -> str:
    """ADS's domain knowledge for the zone: a short English dictionary of the tables and columns, declarative. ADS's
    guardrail reads every uploaded .md and its grounder keeps little of them, so nothing that belongs in the task (the
    question, the insights, the guidance) goes here."""
    vertices, lons, lats = _zone_span(polygon)
    sources = "; ".join(f"{name}: {', '.join(keys)}" for name, keys in files.items()) or "none in this answer"
    return "\n".join(
        [
            "# Area of interest: tables and columns",
            f"- Zone: the user's polygon on the map, {vertices} vertices, lon {lons}, lat {lats}.",
            *(
                f"- {t['file']} ({t['filename']}): {'the main table, one row = one event' if n == 0 else 'a context layer, one row = one point'}; "
                f"{t['inside']} of {t['total']} rows inside; columns {', '.join(t['columns'])}; SIG analysed this table at H3 resolution {t['dataset_resolution']}."
                for n, t in enumerate(tables)
            ),
            "- in_aoi: 1 = the point lies inside the polygon (point-in-polygon, its edges straight in degrees), 0 = outside.",
            "- aoi_zone: inside (in_aoi 1) | adjacent (outside the polygon, in a zone cell or one of its six neighbours) | far.",
            "- SIG's cells: one row of SIG's table = one cell with its events; shares and medians per cell exist for cells with enough events.",
            f"- SIG's terms: {SIG_TERMS}; credibility 1 confirmed by another source, 2 not confirmed, 5 contradicted, 6 no other source.",
            f"- SIG's findings come from: {sources}. The graph holds other datasets too; a finding from another file is context, not a fact about the base layer.",
        ]
    )


def build_domain_md(polygon: dict[str, Any], tables: list[dict[str, Any]], zone_cells: int, files: dict[str, list[str]]) -> str:
    """The final prompt's domain knowledge, in the report's language: how the zone was applied to the data, where the
    findings come from, and the glossary."""
    vertices, lons, lats = _zone_span(polygon)
    base = tables[0]["filename"]
    by_file = [
        f"- {name}: {', '.join(keys)}, "
        + ("базовий шар цього аналізу." if name == base else "інший датасет графа, не базовий шар; його числа описують інші дані.")
        for name, keys in files.items()
    ]
    sources = [
        "## Джерела знахідок SIG",
        "",
        f"Базовий шар цього аналізу: {base}. Граф SIG містить і інші датасети. Знахідка з іншого файлу не є фактом про базовий шар, і її числа не можна поєднувати з числами ADS.",
        "",
        *by_file,
        "",
    ]
    return "\n".join(
        [
            "# Зона інтересу (AOI): методика",
            "",
            f"- Зона задана користувачем як полігон на мапі: {vertices} вершин, довгота {lons}, широта {lats}.",
            *(
                f"- Файл {t['file']}: {'базовий шар подій' if n == 0 else 'контекстний шар'}, файл {t['filename']}, {t['total']} рядків, з них {t['inside']} у зоні."
                for n, t in enumerate(tables)
            ),
            "- Колонка in_aoi: 1 означає, що точка лежить усередині полігона, 0 поза ним. Обчислено детерміновано (point-in-polygon), без моделі. "
            "Колонка aoi_zone: inside у полігоні, adjacent поза ним у комірках зони чи їх шести сусідах, far решта.",
            f"- Зону передано в SIG як {zone_cells} комірок H3 відкритої мапи, чиї центри лежать у полігоні, разом із питанням користувача (або без "
            "нього, якщо відповідь із питанням не мала знахідок у зоні). Від цих комірок ретривер SIG увійшов у граф, обійшов його і зібрав знахідки "
            "в зоні та пов'язані з ними; ADS отримав їх як встановлені факти.",
            "",
            *(sources if by_file else []),
            SIG_GLOSSARY,
        ]
    )
