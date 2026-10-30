"""The shipped datasets the console offers in its list (``data/demo/``, kept in git; docs/current_docs/10_verification.md
§10.2): each a cleaned, deduplicated table of git size with the upload options it is meant to be read with, written with
``data/demo/datasets.json`` (the list ``GET /api/demo/datasets`` serves) and ``data/demo/SOURCES.md`` (sources, licences).

    python scripts/demo_datasets.py            # download what is missing (no account needed), check, prepare, write
    python scripts/demo_datasets.py --offline  # from the copies already in data/raw/ and data/benchmarks/sf_fire/
    python scripts/demo_datasets.py --check    # only check data/demo/ against the pins: the files, rows, hashes

The retail table is generated (``insight_graph_service/core/demo.py``), the California housing table is kept as shipped,
the SF Fire blocks are those of ``scripts/sf_fire_benchmark.py`` — the tables the time layer was calibrated and held out
on, byte for byte —, and each other public dataset is a module of ``scripts/demo_data/``: its Kaggle files pinned by
SHA-256 and a deterministic ``prepare``, sampled to at most 50,000 rows by a time window, a region and a stable hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from insight_graph_service.core.demo import DEMO_DIR, generate_retail_dataset  # noqa: E402

RAW = ROOT / "data" / "raw"
MODULES = (
    "sf_fire_calls",
    "chicago_crashes",
    "chicago_schools",
    "chicago_crimes",
    "chicago_crimes_2024",
    "chicago_bars",
    "chicago_liquor_stores",
    "chicago_l_stations",
    "nyc_collisions",
    "us_accidents",
    "uk_road_safety",
    "us_wildfires",
)
SF_FIRE = {"calibration": "2017-01 … 2019-06", "holdout_a": "2019-07 … 2020-02", "holdout_b": "2020-03 … 2020-08"}
PDDL = "Open Data Commons Public Domain Dedication and License (PDDL) 1.0"
SHIPPED: list[dict[str, Any]] = [  # the list's head, in order; then the modules' (each module's META)
    {
        "key": "retail_synthetic",
        "title": "Роздрібні продажі (синтетичні)",
        "description": "5 000 замовлень із закладеними явищами: де знижка з'їдає маржу, де затримки ведуть до повернень.",
        "license": "згенеровано (insight_graph_service/core/demo.py)",
        "source": "SIG, синтетичні дані з відомою істиною",
        "options": {"bins": "", "categories": ""},
    },
    {
        "key": "california_housing",
        "title": "Житло Каліфорнії (мапа)",
        "description": "20 640 кварталів перепису 1990 року з координатами, проаналізовані як комірки H3: ціна житла, дохід, близькість до океану.",
        "license": "StatLib (Pace & Barry, 1997), вільне використання з посиланням",
        "source": "Pace R. K., Barry R. Sparse spatial autoregressions, 1997 (StatLib)",
        "options": {"bins": "median_income_median:4", "categories": "", "geo": "auto@5"},
    },
    *(
        {
            "key": f"sf_fire_{block}",
            "title": f"Пожежі Сан-Франциско: {label}",
            "description": f"Інциденти пожежної служби (без медичних викликів), {label}: точки й час тривоги, час прибуття і "
            "тривалість, задіяні сили, типи ситуацій і об'єктів. Блок, на якому калібрували й перевіряли часовий шар.",
            "license": PDDL,
            "source": "DataSF «Fire Incidents» (wr8u-xric) через Kaggle san-francisco/sf-fire-data-incidents-violations-and-more",
            "prepared": "scripts/sf_fire_benchmark.py: дублікати прибрано, 2017-01 … 2020-08, точки в межах міста, "
            "тривалості з часових міток, коди NFIRS згруповано",
            "options": {"time": "alarm_dttm", "geo": "lat,lon@auto", "table": True, "source": "DataSF, Fire Incidents"},
        }
        for block, label in SF_FIRE.items()
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, folder: Path, names: list[str]) -> None:
    """One file of a Kaggle dataset from its public endpoint (a zip of it, or the file), into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "sig-demo-datasets"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = response.read()
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            for name in z.namelist():
                if Path(name).name in names:
                    (folder / Path(name).name).write_bytes(z.read(name))
    else:
        (folder / names[0]).write_bytes(payload)


def write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def build(offline: bool) -> list[dict[str, Any]]:
    """Every shipped table written to ``data/demo/``; returns the list's entries, rows and hashes measured."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    write(generate_retail_dataset(), DEMO_DIR / "retail_synthetic.csv")
    california = DEMO_DIR / "california_housing.csv"
    california.write_bytes(california.read_bytes().replace(b"\r\n", b"\n"))  # kept as shipped, its line ends LF
    subprocess.run([sys.executable, str(Path(__file__).with_name("sf_fire_benchmark.py")), *(["--offline"] if offline else [])], check=True)
    for block in SF_FIRE:
        shutil.copyfile(ROOT / "data" / "benchmarks" / "sf_fire" / f"sf_fire_{block}.csv", DEMO_DIR / f"sf_fire_{block}.csv")
    entries = [dict(e) for e in SHIPPED]
    for key in MODULES:
        module = importlib.import_module(f"demo_data.{key}")
        folder = RAW / key
        for name, digest in module.FILES.items():
            if not offline and not (folder / name).is_file():
                fetch(module.URLS[name], folder, [name])
            if (got := sha256(folder / name)) != digest:
                raise SystemExit(f"{key}: {name} sha256 {got}, pinned {digest}: the source copy changed")
        write(module.prepare(folder), DEMO_DIR / f"{key}.csv")
        entries.append({"key": key, **module.META})
    return entries


def measure(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each entry with its file, rows, bytes, SHA-256 and, for a timed one, the span of its time column."""
    out = []
    for e in entries:
        path = DEMO_DIR / f"{e['key']}.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        time = (e.get("options") or {}).get("time")
        span = [frame[time].min()[:10], frame[time].max()[:10]] if time else None  # ISO stamps sort as times
        out.append(e | {"file": path.name, "rows": len(frame), "bytes": path.stat().st_size, "sha256": sha256(path), "span": span})
    return out


def sources(entries: list[dict[str, Any]]) -> str:
    """``SOURCES.md``: where each shipped table comes from, its licence and what was done to it."""
    lines = ["# Shipped datasets: sources and licences", "", "Built by `scripts/demo_datasets.py`; each file keeps its source's licence.", ""]
    for e in entries:
        rows = f"- Rows: {e['rows']:,}" + (f"; {e['span'][0]} … {e['span'][1]}" if e["span"] else "")
        lines += [f"## `{e['file']}` — {e['title']}", "", f"- Source: {e['source']}", f"- Licence: {e['license']}", rows]
        lines += [f"- {k.capitalize()}: {e[k]}" for k in ("attribution", "prepared") if e.get(k)] + [""]
    return "\n".join(lines)


def check(entries: list[dict[str, Any]]) -> int:
    """The files against the list's pins; non-zero when one differs or is missing."""
    bad = [e["key"] for e in entries if not (DEMO_DIR / e["file"]).is_file() or sha256(DEMO_DIR / e["file"]) != e["sha256"]]
    print("data/demo/ matches its pins" if not bad else f"differ from their pins: {bad}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--offline", action="store_true", help="use the copies already downloaded")
    parser.add_argument("--check", action="store_true", help="only check data/demo/ against data/demo/datasets.json")
    args = parser.parse_args()
    manifest = DEMO_DIR / "datasets.json"
    if args.check:
        return check(json.loads(manifest.read_text(encoding="utf-8")))
    entries = measure(build(args.offline))
    manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    (DEMO_DIR / "SOURCES.md").write_text(sources(entries), encoding="utf-8", newline="\n")
    for e in entries:
        print(f"{e['file']}: {e['rows']} rows, {e['bytes'] / 2**20:.1f} MB, sha256 {e['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
