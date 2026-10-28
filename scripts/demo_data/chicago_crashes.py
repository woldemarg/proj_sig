"""Chicago traffic crashes: one row per police-reported crash on the city's streets, its point, its local time, five
per-crash metrics and ten categorical conditions, October 2023 … September 2024 (the copy's last 12 complete months),
a third of them by a stable hash of the record id.

Source: Kaggle ``hientrandai/chicago-traffic-crashes-dataset`` (``crashes.csv`` and ``people.csv``, 2022-01-01 …
2024-10-06), a trimmed copy of the City of Chicago portal's "Traffic Crashes - Crashes" (``85ca-t3if``) and "Traffic
Crashes - People" (``u6pd-qa9d``), CPD E-Crash reports.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SLUG = "hientrandai/chicago-traffic-crashes-dataset"
FILES = {
    "crashes.csv": "3a690aea79337c34a14af699facd607514c250eda4b33b09a127af8cfd902110",
    "people.csv": "0851d0e318ab987ddfbbc528c9a9b4dfdbc13a028c6f115b348cdcf3ae720883",
}
URLS = {name: f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}/{name}" for name in FILES}
META = {
    "title": "ДТП у Чикаго, 2023–2024",
    "description": "Дорожньо-транспортні пригоди в Чикаго за 12 місяців: які умови (погода, освітлення, тип зіткнення, причина) "
    "і місця зсувають кількість травмованих і учасників, обмеження швидкості та вік водіїв.",
    "license": "City of Chicago Data Portal Terms of Use; копія Kaggle позначена CC0",
    "source": "City of Chicago Data Portal, Traffic Crashes (85ca-t3if, u6pd-qa9d), через Kaggle hientrandai/chicago-traffic-crashes-dataset",
    "attribution": "City of Chicago / Chicago Police Department (E-Crash), data.cityofchicago.org; Kaggle copy by hientrandai.",
    "prepared": "2023-10 … 2024-09; третина аварій за хешем ідентифікатора; учасники й вік водіїв з таблиці людей; рідкісні рівні — Other",
    "options": {"time": "crash_date", "geo": "lat,lon@auto", "table": True, "source": "City of Chicago, Traffic Crashes"},
}
START, END = "2023-10-01", "2024-10-01"
BOX = (41.64, 42.03, -87.95, -87.52)  # the city with O'Hare: drops only missing and (0, 0) points
THIN = 3  # a crash is kept when int(sha1(CRASH_RECORD_ID)[:8], 16) % THIN == 0: about a third, under 50,000 rows
MIN_SHARE = 0.005  # a level rarer than this in the sample reads as "Other"
CATEGORICAL = {
    "WEATHER_CONDITION": "weather",
    "LIGHTING_CONDITION": "lighting",
    "FIRST_CRASH_TYPE": "first_crash_type",
    "TRAFFICWAY_TYPE": "trafficway_type",
    "TRAFFIC_CONTROL_DEVICE": "traffic_control",
    "ROADWAY_SURFACE_COND": "road_surface",
    "PRIM_CONTRIBUTORY_CAUSE": "prim_cause",
    "DAMAGE": "damage",
}


def label(s: pd.Series) -> pd.Series:
    """'DARKNESS, LIGHTED ROAD' → 'Darkness, lighted road'; a level rarer than MIN_SHARE → 'Other'."""
    s = s.fillna("UNKNOWN").str.capitalize()
    return s.where(s.map(s.value_counts(normalize=True)) >= MIN_SHARE, "Other")


def prepare(raw_dir: Path) -> pd.DataFrame:
    """The crashes of the window with a point, thinned by a stable hash of the record id; the people file adds persons
    per crash and the drivers' mean age. Record ids, street fields, the injury labels restating injuries_total
    (MOST_SEVERE_INJURY, CRASH_TYPE), all but zero INJURIES_FATAL and the near-constant or mostly unknown ALIGNMENT,
    ROAD_DEFECT, DEVICE_CONDITION and SEC_CONTRIBUTORY_CAUSE are left out."""
    read = lambda name: pd.read_csv(raw_dir / name, dtype=str, keep_default_na=False, na_values=[""]).drop_duplicates()  # noqa: E731
    df = read("crashes.csv").drop_duplicates("CRASH_RECORD_ID")
    # CRASH_DATE ends in 'Z' but is the portal's local wall time: its hour equals CRASH_HOUR on every row
    df["when"] = pd.to_datetime(df["CRASH_DATE"].str[:19], format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    df["lat"], df["lon"] = pd.to_numeric(df["LATITUDE"]).round(6), pd.to_numeric(df["LONGITUDE"]).round(6)
    keep = (df["when"] >= START) & (df["when"] < END) & df["lat"].between(BOX[0], BOX[1]) & df["lon"].between(BOX[2], BOX[3])
    keep &= df["CRASH_RECORD_ID"].map(lambda i: int(hashlib.sha1(i.encode()).hexdigest()[:8], 16) % THIN == 0)
    df = df[keep].sort_values(["when", "CRASH_RECORD_ID"], kind="mergesort").reset_index(drop=True)
    people = read("people.csv").drop_duplicates("PERSON_ID")
    people = people[people["CRASH_RECORD_ID"].isin(df["CRASH_RECORD_ID"])]
    age = pd.to_numeric(people["AGE"], errors="coerce")
    driver_age = age.where((people["PERSON_TYPE"] == "DRIVER") & age.between(15, 99))  # outside: a typing error
    out = pd.DataFrame({"crash_date": df["when"].dt.strftime("%Y-%m-%d %H:%M:%S"), "lat": df["lat"], "lon": df["lon"]})
    speed = pd.to_numeric(df["POSTED_SPEED_LIMIT"], errors="coerce")
    out["posted_speed_limit"] = speed.where(speed.between(5, 70) & (speed % 5 == 0)).astype("Int64")  # 0, 3, 39 …: typos
    out["num_units"] = pd.to_numeric(df["NUM_UNITS"], errors="coerce").astype("Int64")
    out["persons_involved"] = df["CRASH_RECORD_ID"].map(people.groupby("CRASH_RECORD_ID").size()).astype("Int64")  # none: unknown
    out["injuries_total"] = pd.to_numeric(df["INJURIES_TOTAL"], errors="coerce").astype("Int64")
    out["driver_age"] = df["CRASH_RECORD_ID"].map(driver_age.groupby(people["CRASH_RECORD_ID"]).mean()).round(1)
    for src, name in CATEGORICAL.items():
        out[name] = label(df[src])
    out["day_of_week"] = df["when"].dt.day_name()
    out["hour_band"] = pd.cut(df["when"].dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)
    return out
