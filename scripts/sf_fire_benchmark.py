"""The SF Fire Incidents benchmark: a public table of point events with times, for the time layer and the map
(docs/current_docs/10_verification.md §10.4). The file stays out of git: this script downloads it, checks it and
builds the three blocks the benchmark reads.

Source: Kaggle ``san-francisco/sf-fire-data-incidents-violations-and-more``, file ``fire-incidents.csv`` — the City of
San Francisco's mirror of DataSF "Fire Incidents" (``wr8u-xric``), Public Domain Dedication and License (PDDL), frozen
since 2020-09. One row per non-medical incident, its point, its alarm, arrival and close times, the units sent and
the NFIRS codes of what was found and done.

    python scripts/sf_fire_benchmark.py            # download (no account needed), check, prepare, split
    python scripts/sf_fire_benchmark.py --offline  # prepare and split an already downloaded copy

Blocks (by alarm time; the calibration block alone tunes, the others hold out):
    calibration  2017-01-01 … 2019-06-30
    holdout_a    2019-07-01 … 2020-02-29   the same regime
    holdout_b    2020-03-01 … 2020-08-31   a regime shift (shelter-in-place)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmarks" / "sf_fire"
URL = "https://www.kaggle.com/api/v1/datasets/download/san-francisco/sf-fire-data-incidents-violations-and-more/fire-incidents.csv"
RAW_SHA256 = "f59d2d0e703808aa2ac39e7e03de52c7a19d4f302943fcc66fbbaf4f0f61ca1b"
PREPARED_SHA256 = "a7bbc547e93b487231ecad2d21a47f7929a74421699d36690aed379c4f76add5"
START, END = "2017-01-01", "2020-09-01"
BLOCKS = {"calibration": (START, "2019-07-01"), "holdout_a": ("2019-07-01", "2020-03-01"), "holdout_b": ("2020-03-01", END)}
BOX = (37.60, 37.84, -122.53, -122.35)  # the city and its airport: drops no valid point of the window
MIN_CODE_ROWS = 600  # a detailed code rarer than this in the calibration block reads as "<series> other"
TS = "%Y-%m-%dT%H:%M:%S.%f"
SITUATION = {"1": "1xx Fire", "2": "2xx Overpressure/explosion (no fire)", "3": "3xx Rescue & EMS", "4": "4xx Hazardous condition (no fire)",
             "5": "5xx Service call", "6": "6xx Good intent call", "7": "7xx False alarm & false call",
             "8": "8xx Severe weather & natural disaster", "9": "9xx Special incident type"}  # fmt: skip
PROPERTY = {"0": "0xx Other", "1": "1xx Assembly", "2": "2xx Educational", "3": "3xx Health care & detention", "4": "4xx Residential",
            "5": "5xx Mercantile & business", "6": "6xx Industrial & utility", "7": "7xx Manufacturing", "8": "8xx Storage",
            "9": "9xx Outside & special property", "N": "NNN None", "U": "UUU Undetermined"}  # fmt: skip
ACTION = {"0": "0x Other", "1": "1x Fire control", "2": "2x Search & rescue", "3": "3x EMS & transport", "4": "4x Hazardous condition",
          "5": "5x Support services", "6": "6x Systems & services", "7": "7x Assistance", "8": "8x Information & investigation",
          "9": "9x Fill-in & standby"}  # fmt: skip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw: Path) -> None:
    """The file from Kaggle's public endpoint (a zip of the one CSV), unpacked beside the blocks."""
    with urllib.request.urlopen(URL, timeout=300) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        raw.write_bytes(z.read("fire-incidents.csv"))


def coded(raw: pd.Series, series: dict[str, str], calibration: pd.Series, detailed: bool = True) -> tuple[pd.Series, pd.Series | None]:
    """An NFIRS column ('745 - Alarm system …' before 2014, '745 Alarm system …' after) as its series and, when
    ``detailed``, its code with the code's most frequent wording; a code rare in the calibration block reads as its
    series' "other", so no hold-out row shapes the levels."""
    parts = raw.str.extract(r"^\s*([0-9A-Za-z]+)\s*-?\s*(.*?)\s*$")
    code, text = parts[0].str.upper(), parts[1]
    group = code.str[0].map(series).fillna("Unknown")
    if not detailed:
        return group, None
    words = (
        pd.DataFrame({"code": code, "text": text.str.strip()}).dropna().groupby(["code", "text"]).size().reset_index(name="n")
        .sort_values(["code", "n", "text"], ascending=[True, False, True]).drop_duplicates("code").set_index("code")["text"]
    )  # fmt: skip
    keep = set(code[calibration].value_counts().loc[lambda v: v >= MIN_CODE_ROWS].index)
    detail = (code + " " + code.map(words).fillna("")).where(code.isin(keep), group.str.split(" ", n=1).str[0] + " other")
    return group, detail.where(code.notna(), "Unknown")


def prepare(raw: Path) -> pd.DataFrame:
    """The benchmark table: the window's incidents with a point, two durations and six unit counts as metrics, ten
    categorical columns; record numbers (monotone in time) and the portal's computed regions (restating the point) left out."""
    df = pd.read_csv(raw, dtype=str, keep_default_na=False, na_values=[""]).drop_duplicates()
    alarm = pd.to_datetime(df["Alarm DtTm"], format=TS, errors="coerce")
    df = df[(alarm >= START) & (alarm < END)].assign(alarm=lambda d: pd.to_datetime(d["Alarm DtTm"], format=TS))
    xy = df["point"].fillna("").str.extract(r"\[\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\]").astype(float)
    df = df.assign(lon=xy[0].round(6), lat=xy[1].round(6))
    df = (
        df[df["lat"].between(BOX[0], BOX[1]) & df["lon"].between(BOX[2], BOX[3])]
        .sort_values(["alarm", "ID"], kind="mergesort")
        .reset_index(drop=True)
    )
    minutes = lambda col: (pd.to_datetime(df[col], format=TS, errors="coerce") - df["alarm"]).dt.total_seconds() / 60  # noqa: E731
    out = pd.DataFrame({"alarm_dttm": df["alarm"].dt.strftime("%Y-%m-%d %H:%M:%S"), "lat": df["lat"], "lon": df["lon"]})
    response, duration = minutes("Arrival DtTm"), minutes("Close DtTm")
    out["response_min"] = response.where(response.between(0, 120)).round(2)  # outside: a typing error, not a time
    out["duration_min"] = duration.where(duration.between(0, 1440)).round(2)
    for src in ("Suppression Units", "Suppression Personnel", "EMS Units", "EMS Personnel", "Other Units", "Other Personnel"):
        out[src.lower().replace(" ", "_")] = pd.to_numeric(df[src], errors="coerce").astype("Int64")
    calibration = df["alarm"] < BLOCKS["calibration"][1]
    out["battalion"] = df["Battalion"].fillna("Unknown")
    out["supervisor_district"] = ("D" + df["Supervisor District"]).fillna("Unknown")
    out["neighborhood"] = df["Analysis Neighborhood"].fillna("Unknown")
    out["situation_group"], out["situation"] = coded(df["Primary Situation"], SITUATION, calibration)
    out["property_use_group"], out["property_use"] = coded(df["Property Use"], PROPERTY, calibration)
    out["action_group"], _ = coded(df["Action Taken Primary"], ACTION, calibration, detailed=False)
    out["day_of_week"] = df["alarm"].dt.day_name()
    out["hour_band"] = pd.cut(df["alarm"].dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--offline", action="store_true", help="use the copy already in data/benchmarks/sf_fire/")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "fire-incidents.csv"
    if not args.offline:
        download(raw)
    if (got := sha256(raw)) != RAW_SHA256:
        print(f"the download is not the pinned file (sha256 {got}): the Kaggle copy changed", file=sys.stderr)
        return 1
    prepared = OUT / "prepared.csv"
    prepare(raw).to_csv(prepared, index=False, lineterminator="\n", encoding="utf-8")
    if (got := sha256(prepared)) != PREPARED_SHA256:
        print(f"prepared.csv sha256 {got}, expected {PREPARED_SHA256}: pandas read the file differently", file=sys.stderr)
        return 1
    table = pd.read_csv(prepared, dtype=str, keep_default_na=False)
    when = pd.to_datetime(table["alarm_dttm"], format="%Y-%m-%d %H:%M:%S")
    for name, (a, b) in BLOCKS.items():
        block = OUT / f"sf_fire_{name}.csv"
        table[(when >= a) & (when < b)].to_csv(block, index=False, lineterminator="\n")
        print(f"{name}: {int(((when >= a) & (when < b)).sum())} rows, sha256 {sha256(block)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
