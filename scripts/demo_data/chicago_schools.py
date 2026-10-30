"""Chicago public schools: one row per school of the 2023–24 school year — its point, its name and whether it is an
elementary or a high school. A context layer for the city's crashes (``chicago_crashes``): another type of points the
crashes' cells are read against (docs/02_discovery.md §2.9), not a table to mine on its own.

Source: City of Chicago Data Portal, "Chicago Public Schools - School Locations SY2324" (``mv87-m4mi``), its CSV export.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILE = "rows.csv"
FILES = {FILE: "230aadd75860bd2ef96ad2ef4fbc4e14c72f791ac2fa6ddcf383212d323d3def"}  # 70,757 bytes, 649 schools
URLS = {FILE: "https://data.cityofchicago.org/api/views/mv87-m4mi/rows.csv?accessType=DOWNLOAD"}
META = {
    "title": "Школи Чикаго, 2023–24 (контекстний шар)",
    "description": "Розташування державних шкіл Чикаго: початкові й старші. Контекстний шар до ДТП Чикаго — комірки аварій "
    "читаються за кількістю шкіл поруч.",
    "license": "City of Chicago Data Portal Terms of Use",
    "source": "City of Chicago Data Portal, Chicago Public Schools - School Locations SY2324 (mv87-m4mi)",
    "attribution": "City of Chicago / Chicago Public Schools, data.cityofchicago.org.",
    "prepared": "усі 649 шкіл; точка з Lat/Long; тип — ES/HS",
    "context_only": True,
    "options": {"geo": "lat,lon"},
}


def prepare(raw_dir: Path) -> pd.DataFrame:
    """Every school with its point, its short name and its grade category, ordered by the school id; the WKT copy of the
    point and the address are left out."""
    df = pd.read_csv(raw_dir / FILE, dtype=str, keep_default_na=False, na_values=[""]).sort_values("School_ID", kind="mergesort")
    out = pd.DataFrame({"lat": pd.to_numeric(df["Lat"]).round(6), "lon": pd.to_numeric(df["Long"]).round(6)})
    out["school_name"] = df["Short_Name"].str.strip()
    out["grade_cat"] = df["Grade_Cat"].str.strip()
    return out.reset_index(drop=True)
