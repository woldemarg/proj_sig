"""Chicago CTA 'L' stations: one row per station (its platforms once), its point and name. A context layer for the
Chicago crimes (docs/02_discovery.md §2.9).

Source: City of Chicago Data Portal, "CTA - System Information - List of 'L' Stops" (``8pix-ypme``).
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import pandas as pd

FILE = "l_stops.json"
QUERY = {"$select": "map_id,station_name,location", "$order": "map_id", "$limit": 5000}
FILES = {FILE: "7691083451e06aee30500c6a6aa59638eb1437f9e0f21c6be18c3435bfe4d377"}  # the portal re-publishes: a new pin is deliberate
URLS = {FILE: f"https://data.cityofchicago.org/resource/8pix-ypme.json?{urllib.parse.urlencode(QUERY)}"}
META = {
    "title": "Станції 'L' у Чикаго (контекстний шар)",
    "description": "Станції міської залізниці CTA. Контекстний шар до злочинів Чикаго.",
    "license": "City of Chicago Data Portal Terms of Use",
    "source": "City of Chicago Data Portal, CTA - System Information - List of 'L' Stops (8pix-ypme)",
    "attribution": "Chicago Transit Authority, data.cityofchicago.org.",
    "prepared": "одна станція — один рядок (платформи разом); точка з поля location",
    "context_only": True,
    "options": {"geo": "lat,lon"},
}


def prepare(raw_dir: Path) -> pd.DataFrame:
    """Every station once (by ``map_id``) with its point — the portal's ``location`` is ``{latitude, longitude}`` — and name."""
    stops = pd.DataFrame(json.loads((raw_dir / FILE).read_text(encoding="utf-8"))).drop_duplicates("map_id")
    out = pd.DataFrame({"lat": stops["location"].map(lambda p: round(float(p["latitude"]), 6))})
    out["lon"] = stops["location"].map(lambda p: round(float(p["longitude"]), 6))
    out["station"] = stops["station_name"]
    return out.sort_values(["lat", "lon"]).reset_index(drop=True)
