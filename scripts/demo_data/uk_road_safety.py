"""UK road accidents: STATS19 personal-injury accidents the Metropolitan Police recorded in Greater London in 2017 (the
City of London force aside), one row per accident with its point, its time and the conditions at the scene.

Source: Kaggle ``tsiaras/uk-road-safety-accidents-and-vehicles``, file ``Accident_Information.csv`` (2005–2017, codes
already in words), built from the Department for Transport's "Road Safety Data" (STATS19). One complete year: two would
pass 50,000 rows, and the Met recorded serious injuries differently from late 2016 (CRASH).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SLUG = "tsiaras/uk-road-safety-accidents-and-vehicles"
FILES = {"Accident_Information.csv": "a651405f6778c4c2618d484f226a04f4a9dc98b20572d7be7dbb11fc251a8305"}
URLS = {name: f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}/{name}" for name in FILES}
META = {
    "title": "ДТП у Лондоні, 2017",
    "description": "Дорожньо-транспортні пригоди з потерпілими, які зареєструвала Столична поліція Лондона у 2017 році: які "
    "райони, типи доріг і перехресть, години, освітлення й погода пов'язані з більшою кількістю потерпілих і транспорту.",
    "license": "Open Government Licence v3.0 (DfT STATS19); копія Kaggle — ODbL 1.0 / DbCL 1.0",
    "source": "Department for Transport, Road Safety Data (STATS19), через Kaggle tsiaras/uk-road-safety-accidents-and-vehicles",
    "attribution": "Contains public sector information licensed under the Open Government Licence v3.0 (Department for Transport, STATS19).",
    "prepared": "Metropolitan Police, 2017; точки в межах Великого Лондона; один рядок на пригоду; рідкісні рівні — Other",
    "options": {"time": "accident_datetime", "geo": "lat,lon@auto", "table": True, "source": "DfT STATS19, Metropolitan Police"},
}
FORCE, START, END = "Metropolitan Police", "2017-01-01", "2018-01-01"
BOX = (51.28, 51.70, -0.52, 0.34)  # Greater London: lat_min, lat_max, lon_min, lon_max
SPEED_LIMITS = (20, 70)  # posted limits outside (0, 10, 15 mph) are coding errors
MIN_LEVEL_ROWS = 25  # a level rarer than this reads as "Other"
MISSING = {"Data missing or out of range": "Unknown", "Unallocated": "Unknown"}


def label(s: pd.Series) -> pd.Series:
    s = s.fillna("Unknown").replace(MISSING)
    return s.where(s.map(s.value_counts()) >= MIN_LEVEL_ROWS, "Other")


def prepare(raw_dir: Path) -> pd.DataFrame:
    """The force's accidents of the window with a valid point and time: four metrics, ten categorical columns; the
    accident index, OS grid eastings/northings, LSOA, highway authority and road numbers (restating the point) left out."""
    df = pd.read_csv(raw_dir / "Accident_Information.csv", dtype=str, keep_default_na=False, na_values=["", "NA"])
    df = df[df["Police_Force"] == FORCE].drop_duplicates()
    when = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M", errors="coerce")
    df = df.assign(when=when, lat=pd.to_numeric(df["Latitude"]).round(6), lon=pd.to_numeric(df["Longitude"]).round(6))
    df = df[(df["when"] >= START) & (df["when"] < END) & df["lat"].between(BOX[0], BOX[1]) & df["lon"].between(BOX[2], BOX[3])]
    df = df.sort_values(["when", "Accident_Index"], kind="mergesort").drop_duplicates("Accident_Index").reset_index(drop=True)
    num = lambda col: pd.to_numeric(df[col], errors="coerce")  # noqa: E731
    out = pd.DataFrame({"accident_datetime": df["when"].dt.strftime("%Y-%m-%d %H:%M:%S"), "lat": df["lat"], "lon": df["lon"]})
    out["number_of_vehicles"] = num("Number_of_Vehicles").where(lambda v: v >= 1).astype("Int64")
    out["number_of_casualties"] = num("Number_of_Casualties").where(lambda v: v >= 1).astype("Int64")
    out["speed_limit_mph"] = num("Speed_limit").where(lambda v: v.between(*SPEED_LIMITS)).astype("Int64")
    out["police_attended"] = df["Did_Police_Officer_Attend_Scene_of_Accident"].map({"1": 1, "2": 0, "3": 0}).astype("Int64")
    for col, src in {
        "severity": "Accident_Severity",
        "road_type": "Road_Type",
        "junction_detail": "Junction_Detail",
        "light_conditions": "Light_Conditions",
        "weather_conditions": "Weather_Conditions",
        "road_surface": "Road_Surface_Conditions",
        "urban_rural": "Urban_or_Rural_Area",
        "local_authority": "Local_Authority_(District)",
    }.items():
        out[col] = label(df[src])
    out["day_of_week"] = df["when"].dt.day_name()
    out["hour_band"] = pd.cut(df["when"].dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)
    return out
