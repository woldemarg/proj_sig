"""Chicago places licensed to serve alcohol on the premises (tavern, consumption on premises, late hour): one row per
place, its point. A context layer for the Chicago crimes (docs/02_discovery.md §2.9); ``chicago_liquor_stores`` is the
same licences' package-goods stores.

Source: City of Chicago Data Portal, "Liquor and Public Places of Amusement Licenses" current (``nrmj-3kcf``).
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd

FILE = "liquor_licenses.csv"
QUERY = {"$select": "license_id,license_description,latitude,longitude", "$where": "latitude IS NOT NULL", "$limit": 50000}
FILES = {FILE: "40c0a5533bfebdd347d5625a2d8f78c5a9e83c2a15f65ee8ec43de51ebf4e73a"}  # the portal re-publishes: a new pin is deliberate
URLS = {FILE: f"https://data.cityofchicago.org/resource/nrmj-3kcf.csv?{urllib.parse.urlencode(QUERY)}"}
ON_PREMISE = ("Tavern", "Consumption on Premises - Incidental Activity", "Late Hour")


def meta(title: str, description: str, kinds: tuple[str, ...]) -> dict[str, Any]:
    return {
        "title": title,
        "description": description,
        "license": "City of Chicago Data Portal Terms of Use",
        "source": "City of Chicago Data Portal, Liquor and Public Places of Amusement Licenses (nrmj-3kcf)",
        "attribution": "City of Chicago / Department of Business Affairs and Consumer Protection, data.cityofchicago.org.",
        "prepared": f"чинні ліцензії {', '.join(kinds)}; одне місце — один рядок",
        "context_only": True,
        "options": {"geo": "lat,lon"},
    }


META = meta(
    "Бари й заклади з алкоголем у Чикаго (контекстний шар)",
    "Місця з ліцензією на продаж алкоголю для споживання на місці. Контекстний шар до злочинів Чикаго.",
    ON_PREMISE,
)


def places(raw_dir: Path, kinds: tuple[str, ...]) -> pd.DataFrame:
    """The distinct points of the licences of ``kinds`` (a place's several licences: once), in coordinate order."""
    df = pd.read_csv(raw_dir / FILE, dtype=str)
    df = df[df["license_description"].isin(kinds)]
    out = pd.DataFrame({"lat": pd.to_numeric(df["latitude"]).round(6), "lon": pd.to_numeric(df["longitude"]).round(6)})
    return out.drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)


def prepare(raw_dir: Path) -> pd.DataFrame:
    return places(raw_dir, ON_PREMISE)
