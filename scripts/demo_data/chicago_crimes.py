"""Chicago crimes: one row per reported crime of 2025 — its point, its time, its class (seven), street or not, arrest,
domestic, the hour band — a sixth of them by a stable hash of the record id. The target of the cross-type validation
(docs/extra_funcs/spatial_analytics_multi_type/cross_type_context_architecture.md §8.2a), read with the bars, the liquor
stores or the 'L' stations as its context layer; ``chicago_crimes_2024`` is its held-out year.

Source: City of Chicago Data Portal, "Crimes - 2001 to Present" (``ijzp-q8t2``), a SoQL export of one year.
"""

from __future__ import annotations

import hashlib
import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd

YEAR = 2025
THIN = 6  # a crime is kept when int(sha1(id)[:8], 16) % THIN == 0: about a sixth, under 50,000 rows a year
FIELDS = "id,date,primary_type,arrest,domestic,location_description,latitude,longitude"
CLASSES = {  # primary type -> class; the rest "other" (at most eight classes: each becomes a share of a cell's crimes)
    **dict.fromkeys(["ROBBERY"], "robbery"),
    **dict.fromkeys(["BATTERY", "ASSAULT", "HOMICIDE", "CRIMINAL SEXUAL ASSAULT"], "assault"),
    **dict.fromkeys(["THEFT"], "theft"),
    **dict.fromkeys(["MOTOR VEHICLE THEFT", "BURGLARY", "CRIMINAL DAMAGE", "ARSON"], "property"),
    **dict.fromkeys(["DECEPTIVE PRACTICE"], "deceptive"),
    **dict.fromkeys(["NARCOTICS", "WEAPONS VIOLATION", "CONCEALED CARRY LICENSE VIOLATION"], "drugs_weapons"),
}
STREET = {"STREET", "SIDEWALK", "ALLEY"}


def source(year: int, sha256: str) -> tuple[dict[str, str], dict[str, str]]:
    """(FILES, URLS) of one year's export: the file pinned by its SHA-256 (the portal re-publishes: a new pin is a
    deliberate update), the SoQL query that wrote it."""
    name = f"crimes_{year}.csv"
    query = {"$select": FIELDS, "$where": f"year={year} AND latitude IS NOT NULL", "$order": "id", "$limit": 500000}
    return {name: sha256}, {name: f"https://data.cityofchicago.org/resource/ijzp-q8t2.csv?{urllib.parse.urlencode(query)}"}


def meta(year: int, role: str) -> dict[str, Any]:
    return {
        "title": f"Злочини в Чикаго, {year}",
        "description": f"Зареєстровані поліцією злочини {year} року: клас (грабіж, напад, крадіжка, майнові, шахрайство, наркотики "
        f"чи зброя, інше), вулиця чи ні, арешт, домашнє насильство, година. {role}",
        "license": "City of Chicago Data Portal Terms of Use",
        "source": "City of Chicago Data Portal, Crimes - 2001 to Present (ijzp-q8t2)",
        "attribution": "City of Chicago / Chicago Police Department (CLEAR), data.cityofchicago.org.",
        "prepared": f"{year}; шоста частина за хешем ідентифікатора; типи злочинів — у сім класів; місце — вулиця чи ні",
        "options": {
            "time": "crime_date",
            "geo": "lat,lon@auto",
            "source": "City of Chicago, Crimes",
        },  # no table level: no numeric column without cells
    }


FILES, URLS = source(YEAR, "eac8c61ee549a5f9e78b4148c949091ad858cf2577bd3e0694dd9632bf88a3f7")
META = meta(YEAR, "Ціль крос-типової перевірки: з контекстним шаром барів, магазинів алкоголю чи станцій 'L'.")


def prepare(raw_dir: Path, year: int = YEAR, thin: int = THIN) -> pd.DataFrame:
    """A year of crimes with a point, thinned by a stable hash of the record id (``thin`` 1: every one), in time order."""
    df = pd.read_csv(raw_dir / f"crimes_{year}.csv", dtype=str, keep_default_na=False, na_values=[""])
    df = df[df["id"].map(lambda i: int(hashlib.sha1(i.encode()).hexdigest()[:8], 16) % thin == 0)]
    when = pd.to_datetime(df["date"], errors="coerce")
    df = df.assign(when=when).sort_values(["when", "id"], kind="mergesort")
    out = pd.DataFrame({"crime_date": df["when"].dt.strftime("%Y-%m-%d %H:%M:%S")})
    out["lat"], out["lon"] = pd.to_numeric(df["latitude"]).round(6), pd.to_numeric(df["longitude"]).round(6)
    out["crime"] = df["primary_type"].map(CLASSES).fillna("other")
    out["street"] = df["location_description"].isin(STREET).map({True: "yes", False: "no"})
    out["arrest"], out["domestic"] = df["arrest"].str.lower(), df["domestic"].str.lower()
    out["hour_band"] = pd.cut(df["when"].dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)
    return out.reset_index(drop=True)
