"""New York City motor vehicle collisions: one row per police-reported crash (form MV-104AN: someone injured or killed,
or $1,000 of damage) — its point, time, injury counts, vehicles, the kind of site and vehicle 1's contributing factor and
type —, March 2023 … February 2024 (the snapshot's last 12 complete months), every second crash by a stable hash of
COLLISION_ID.

Source: Kaggle ``sahirmaharajj/motor-vehicle-collisions-crashes`` (2012-07 … 2024-03-25), a copy of NYC Open Data's
"Motor Vehicle Collisions - Crashes" (``h9gi-nx95``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SLUG = "sahirmaharajj/motor-vehicle-collisions-crashes"
FILE = "Motor_Vehicle_Collisions_-_Crashes.csv"
FILES = {FILE: "0c895e5aade0f10fc18d1e1298705fd34181114aa5612e00ccfc4a483354df49"}  # 440,424,025 bytes, unzipped
URLS = {FILE: f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}/{FILE}"}
META = {
    "title": "ДТП у Нью-Йорку, 2023–2024",
    "description": "Зареєстровані поліцією Нью-Йорка аварії: де, коли, через що і з якими транспортними засобами стаються "
    "пригоди з травмованими людьми й пішоходами.",
    "license": "NYC Open Data Terms of Use; копія Kaggle позначена Apache 2.0",
    "source": "NYC Open Data (NYPD), Motor Vehicle Collisions - Crashes (h9gi-nx95), через Kaggle sahirmaharajj/motor-vehicle-collisions-crashes",
    "attribution": "NYPD Motor Vehicle Collisions - Crashes, NYC Open Data (h9gi-nx95); Kaggle mirror by sahirmaharajj.",
    "prepared": "2023-03 … 2024-02; аварії з точкою в межах міста; половина за хешем COLLISION_ID; тип транспорту — 8 класів; "
    "тип місця з полів вулиць (самі вулиці не включено)",
    "options": {"time": "crash_dttm", "geo": "lat,lon@auto", "table": True, "source": "NYC Open Data, NYPD collisions"},
}
START, END = "2023-03-01", "2024-03-01"
BOX = (40.47, 40.93, -74.27, -73.68)  # the five boroughs; drops only the missing and (0, 0) points
THIN = 2  # a crash is kept when int(sha1(COLLISION_ID)[:8], 16) % THIN == 0
MIN_FACTOR_ROWS = 60  # a factor rarer than this reads as its series' "(other)"
HIGHWAY = r"EXPRESSWAY|EXPWY|EXPY|PARKWAY|PKWY|HIGHWAY|HWY|BRIDGE|TUNNEL|THRUWAY|TURNPIKE|FDR DR|HARLEM RIVER DR|\(BQE\)"
BOULEVARD = r"^(?:EASTERN|OCEAN|PELHAM|BAY) PARKWAY"  # city streets named parkway
# the first match wins, on the lower-cased free-text type (about 250 spellings); none or blank: "Other / unknown". At
# most eight classes, so the map's cells read them as shares (a finer list leaves the geo level nothing to compare)
VEHICLE = [
    (r"^(?:sedan|4 dr sedan|2 dr sedan|3-door|convertible|pas|passenger|passanger)$|sport utility|^suv", "Car / SUV"),
    (r"^(?:taxi|limo)", "Taxi / limo"),
    (r"^amb|^ems|fdny|fire", "Emergency vehicle"),
    (r"bus", "Bus"),
    (r"^e-? ?bik|^e-? ?sco|^(?:bike|bicycle|pedicab)", "Bicycle / e-bike / e-scooter"),
    (r"moped|scoo|minibike|minicycle|motorcycle|motorbike|^mc$", "Motorcycle / moped"),
    (
        r"^(?:pick-?up|pick up|pk$)|^box|^tractor tr|van|truck|dump|garbage|refuse|flat|tow|wrecker|carry all|chassis|tank"
        r"|mixer|stake|sanitation|boom|lift|swe|plow|delivery|open body",
        "Truck / van",
    ),
]
FACTOR_SERIES = [
    (r"pavement|lane marking|shoulders|traffic control device|obstruction|glare|animals", "Road / environment (other)"),
    (r"defect|inadequate|tinted|^(?:tire|steering) failure", "Vehicle defect (other)"),
    (r"phone|texting|electronic|navigation|eating|distraction", "Distraction (other)"),
    (r"drug|medication|disability|fatigue|asleep|consciousness|illness", "Driver condition (other)"),
]


def first_match(values: pd.Series, rules: list[tuple[str, str]], default: str) -> pd.Series:
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    for pattern, label in rules:
        out = out.where(out.notna() | ~values.str.contains(pattern, regex=True, na=False), label)
    return out.fillna(default)


def prepare(raw_dir: Path) -> pd.DataFrame:
    """Three metrics (persons and pedestrians injured, vehicles involved), six categoricals; the record number, ZIP,
    street names and LOCATION (restating the point), the killed counts (all but zero), the cyclist and motorist counts
    (parts of persons_injured) and a vehicles band (it would restate vehicles_involved) left out."""
    used = lambda c: not c.endswith(("ZIP CODE", "LOCATION", "KILLED")) and "CYCLIST" not in c and "MOTORIST" not in c  # noqa: E731
    df = pd.read_csv(raw_dir / FILE, dtype=str, keep_default_na=False, na_values=[""], usecols=used).drop_duplicates()
    df["when"] = pd.to_datetime(df["CRASH DATE"] + " " + df["CRASH TIME"], format="%m/%d/%Y %H:%M")
    df["id"] = df["COLLISION_ID"].astype("int64")
    df = df.sort_values(["id", "when"], kind="mergesort").drop_duplicates("id")  # one crash per id: its earliest record
    lat, lon = pd.to_numeric(df["LATITUDE"], errors="coerce"), pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df.assign(lat=lat.round(6), lon=lon.round(6))
    df = df[(df["when"] >= START) & (df["when"] < END) & lat.between(BOX[0], BOX[1]) & lon.between(BOX[2], BOX[3])]
    df = df[df["COLLISION_ID"].map(lambda s: int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) % THIN == 0)]
    df = df.sort_values(["when", "id"], kind="mergesort").reset_index(drop=True)
    out = pd.DataFrame({"crash_dttm": df["when"].dt.strftime("%Y-%m-%d %H:%M:%S"), "lat": df["lat"], "lon": df["lon"]})
    out["persons_injured"] = df["NUMBER OF PERSONS INJURED"].astype("int64")  # at most 34 in the window: no cap
    out["pedestrians_injured"] = df["NUMBER OF PEDESTRIANS INJURED"].astype("int64")
    # vehicle i fills type slot i, factor slot i or both (an unidentified second car has a factor but no type)
    slots = lambda prefix: df[[f"{prefix}{i}" for i in range(1, 6)]].notna().sum(axis=1)  # noqa: E731
    vehicles = pd.concat([slots("VEHICLE TYPE CODE "), slots("CONTRIBUTING FACTOR VEHICLE ")], axis=1).max(axis=1)
    out["vehicles_involved"] = vehicles.where(vehicles > 0).astype("Int64")  # no slot filled: unknown, not zero
    out["borough"] = df["BOROUGH"].str.title().fillna("Not recorded")
    factor = df["CONTRIBUTING FACTOR VEHICLE 1"].str.strip().replace({"Illnes": "Illness"}).fillna("Unspecified")
    factor = factor.str.replace("Cell Phone (hand-Held)", "Cell Phone (hand-held)", regex=False)
    rare = factor.map(factor.value_counts()) < MIN_FACTOR_ROWS
    out["factor_1"] = factor.where(~rare, first_match(factor.str.lower(), FACTOR_SERIES, "Other"))
    out["vehicle_1"] = first_match(df["VEHICLE TYPE CODE 1"].str.strip().str.lower(), VEHICLE, "Other / unknown")
    # where it happened, from which street fields the report fills (on and cross, on alone, off) — not the streets
    street = df["ON STREET NAME"].str.strip().str.upper()
    highway = street.str.contains(HIGHWAY, na=False) & ~street.str.contains(BOULEVARD, na=False)
    site = pd.Series("Street segment", index=df.index).mask(highway, "Expressway / parkway")
    site = site.mask(df["CROSS STREET NAME"].notna(), "Intersection").mask(df["OFF STREET NAME"].notna(), "Off-street address")
    out["site_type"] = site.where(street.notna() | df["OFF STREET NAME"].notna(), "Not recorded")
    out["day_of_week"] = df["when"].dt.day_name()
    out["hour_band"] = pd.cut(df["when"].dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)
    return out
