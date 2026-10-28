"""US wildfire records (FPA FOD, 6th edition): one row per wildfire a federal, state or local fire organisation
reported in California in 2016–2018, its point of origin, discovery time, final size, containment time, cause and land
owner — the latest three complete years whose discovery times are all recorded (from 2019 CAL FIRE stops reporting
most containment times, and 2020 loses a fifth of the discovery times too).

Short, K.C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020 [FPA_FOD_20221014]. 6th Edition.
Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.6
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SLUG = "behroozsohrabi/us-wildfire-records-6th-edition"
FILES = {"data.csv": "b214c36df583ebbfa1af10d94976f0e127f5285c9ec205fb64259b7b084b5894"}  # 1,066,385,139 bytes, unzipped
URLS = {name: f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}/{name}" for name in FILES}
META = {
    "title": "Лісові пожежі Каліфорнії, 2016–2018",
    "description": "Понад 26 тисяч лісових пожеж у Каліфорнії з точкою, часом виявлення, площею, тривалістю локалізації та "
    "причиною: де й чому пожежі гасять довше і як змінюється навантаження за сезонами.",
    "license": "CC0 (Kaggle); USDA Forest Service Research Data Archive, суспільне надбання з посиланням на DOI",
    "source": "USDA Forest Service, FPA FOD 6th edition (RDS-2013-0009.6), через Kaggle behroozsohrabi/us-wildfire-records-6th-edition",
    "attribution": "Short, K.C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020 [FPA_FOD_20221014]. "
    "6th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.6",
    "prepared": "Каліфорнія, 2016–2018; повтори однієї пожежі злито до найменшого FOD_ID; клас розміру не включено (він "
    "ділить площу на смуги, тож лише повторював би її); рідкісні рівні — other",
    "options": {"time": "discovery_dttm", "geo": "lat,lon@auto", "table": True, "source": "USDA Forest Service, FPA FOD"},
}
STATE, START, END = "CA", "2016-01-01", "2019-01-01"
BOX = (32.5, 42.1, -124.5, -114.1)  # California; drops no point of the window
MAX_HOURS = 180 * 24  # discovery → containment beyond 180 days reads as a typing error (the window's longest: 168 days)
MIN_ROWS = 100  # a level rarer than this reads as its column's "other"
TS = "%m/%d/%Y %H%M"
COLS = [
    *("FOD_ID", "NWCG_REPORTING_AGENCY", "FIRE_YEAR", "DISCOVERY_DATE", "DISCOVERY_TIME", "NWCG_CAUSE_CLASSIFICATION"),
    *("NWCG_GENERAL_CAUSE", "CONT_DATE", "CONT_TIME", "FIRE_SIZE", "LATITUDE", "LONGITUDE", "OWNER_DESCR", "STATE", "FIPS_NAME"),
]


def fold(values: pd.Series, other: str) -> pd.Series:
    return values.where(values.map(values.value_counts()) >= MIN_ROWS, other)


def prepare(raw_dir: Path) -> pd.DataFrame:
    """Fire size, containment hours and the county's fires that day as metrics, eight categoricals; record ids, fire
    names, the year and day of year (restating the time), state, FIPS codes, source system and the size class left out."""
    years = {str(y) for y in range(int(START[:4]), int(END[:4]))}
    chunks = pd.read_csv(raw_dir / "data.csv", dtype=str, keep_default_na=False, na_values=[""], usecols=COLS, chunksize=250_000)
    df = pd.concat(c[(c["STATE"] == STATE) & c["FIRE_YEAR"].isin(years)] for c in chunks).drop_duplicates()
    df["id"] = df["FOD_ID"].astype("int64")
    # one fire reported several times (multi-start reports at one point): the same discovery, point, size, cause and
    # containment keep the lowest FOD_ID
    same = ["DISCOVERY_DATE", "DISCOVERY_TIME", "LATITUDE", "LONGITUDE", "FIRE_SIZE", "NWCG_GENERAL_CAUSE", "CONT_DATE", "CONT_TIME"]
    df = df.sort_values("id", kind="mergesort").drop_duplicates(same)
    df["when"] = pd.to_datetime(df["DISCOVERY_DATE"] + " " + df["DISCOVERY_TIME"], format=TS, errors="coerce")  # no time: dropped
    lat, lon = pd.to_numeric(df["LATITUDE"], errors="coerce"), pd.to_numeric(df["LONGITUDE"], errors="coerce")
    keep = (df["when"] >= START) & (df["when"] < END) & lat.between(BOX[0], BOX[1]) & lon.between(BOX[2], BOX[3])
    df = df[keep].assign(lat=lat.round(6), lon=lon.round(6)).sort_values(["when", "id"], kind="mergesort").reset_index(drop=True)
    out = pd.DataFrame({"discovery_dttm": df["when"].dt.strftime("%Y-%m-%d %H:%M:%S"), "lat": df["lat"], "lon": df["lon"]})
    size = pd.to_numeric(df["FIRE_SIZE"], errors="coerce")
    out["fire_size_acres"] = size.where(size > 0)
    hours = (pd.to_datetime(df["CONT_DATE"] + " " + df["CONT_TIME"], format=TS, errors="coerce") - df["when"]).dt.total_seconds() / 3600
    out["containment_hours"] = hours.where(hours.between(0, MAX_HOURS)).round(2)  # about 37 % not reported (mostly CAL FIRE)
    county = df["FIPS_NAME"].str.removesuffix(" County")
    day_load = df.groupby([county, df["when"].dt.date], dropna=False)["id"].transform("size")
    out["county_fires_same_day"] = day_load.where(county.notna()).astype("Int64")  # this fire included
    out["general_cause"] = fold(df["NWCG_GENERAL_CAUSE"].fillna("Missing data/not specified/undetermined"), "Other causes")
    out["cause_class"] = df["NWCG_CAUSE_CLASSIFICATION"].fillna("Missing data/not specified/undetermined")
    out["owner"] = fold(df["OWNER_DESCR"].fillna("MISSING/NOT SPECIFIED"), "OTHER FEDERAL")  # the rare owners are all federal
    out["reporting_agency"] = fold(df["NWCG_REPORTING_AGENCY"].fillna("Unknown"), "Other federal")  # FWS, DOD, BOR
    out["county"] = fold(county, "Other county").where(county.notna(), "Unknown")
    out["month"] = df["when"].dt.month_name()
    out["day_of_week"] = df["when"].dt.day_name()
    out["hour_band"] = pd.cut(df["when"].dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)
    return out
