"""San Francisco Fire Department calls for service: one row per call, November 2016 … October 2018 (the copy's latest
24 complete months), a sixth of them by a stable hash of the call number.

The Kaggle copy carries the Databricks "sf-fire-calls" schema: one row per unit dispatched, the call's DATE only (no
clock time; ``AvailableDtTm`` empty) and ``Delay``, minutes from the call received to that unit en route (as DataSF's
``received_dttm`` → ``response_dttm``). So the time column is the call date at 00:00:00; the watch date (a watch runs
08:00 → 07:59) still tells a call that came in 00–07 from one of 08–23.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SLUG = "jacopoferretti/san-francisco-fire-department-calls-dataset"
FILES = {"fire_dept.csv": "b41cee1dae42c6e77f960bdc9f074d3eb9b1215347c4a0dcaf22b7779d47bfbf"}
URLS = {name: f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}/{name}" for name in FILES}
META = {
    "title": "Виклики пожежної служби Сан-Франциско, 2016–2018",
    "description": "Виклики пожежної служби та швидкої допомоги Сан-Франциско, один рядок на виклик: типи викликів, райони "
    "й батальйони, де підрозділи виїжджають повільніше або виклик потребує більше машин. Лише дата виклику, без години.",
    "license": "Open Data Commons PDDL 1.0 (DataSF); копія Kaggle — CC0",
    "source": "DataSF, Fire Department and EMS Dispatched Calls for Service (nuek-vuh3), через Kaggle jacopoferretti/san-francisco-fire-department-calls-dataset",
    "attribution": "San Francisco Fire Department / DataSF, Fire Department and EMS Dispatched Calls for Service (PDDL); Kaggle upload by jacopoferretti.",
    "prepared": "2016-11 … 2018-10; шоста частина викликів за хешем номера; один рядок на виклик — перший направлений підрозділ, "
    "кількість підрозділів і затримки виїзду по всіх; рідкісні рівні — Other",
    "options": {"time": "call_date", "geo": "lat,lon@auto", "table": True, "source": "DataSF, Fire Department Calls for Service"},
}
START, END = "2016-11-01", "2018-11-01"  # the file ends on 2018-11-06
THIN = 6  # a call is kept when int(sha1(CallNumber)[:8], 16) % THIN == 0: about 49,000
BOX = (37.60, 37.84, -122.53, -122.35)  # the city and its airport, as in the SF Fire Incidents benchmark
MAX_MIN = 120  # an en-route delay over two hours is a held or mistyped record, not a dispatch time
MIN_LEVEL = 50  # a level rarer than this reads as "Other"
COLS = [
    *("CallNumber", "UnitID", "CallType", "CallDate", "WatchDate", "Battalion", "FinalPriority", "ALSUnit", "CallTypeGroup"),
    *("UnitType", "UnitSequenceInCallDispatch", "SupervisorDistrict", "Neighborhood", "Location", "Delay"),
]


def fold(s: pd.Series, other: str = "Other") -> pd.Series:
    return s.where(s.map(s.value_counts()) >= MIN_LEVEL, other)


def prepare(raw_dir: Path) -> pd.DataFrame:
    """One row per call: the first-dispatched unit's row (lowest UnitSequenceInCallDispatch, then lowest Delay, then
    UnitID) for the call's fields and the first unit's type; units, ALS units and the first and last en-route delay over
    all its units. Call and incident numbers, addresses, zip, box and station area left out, and the disposition
    (recorded after the fact)."""
    df = pd.read_csv(raw_dir / "fire_dept.csv", dtype=str, usecols=COLS, keep_default_na=False, na_values=["", "null", "None"]).drop_duplicates()
    df = df[(df["CallDate"] >= START) & (df["CallDate"] < END)]
    df = df[df["CallNumber"].map(lambda s: int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) % THIN == 0)]
    xy = df["Location"].str.extract(r"\(\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\)").astype(float)
    df = df.assign(
        lat=xy[0].round(6), lon=xy[1].round(6), seq=pd.to_numeric(df["UnitSequenceInCallDispatch"]), delay=pd.to_numeric(df["Delay"], errors="coerce")
    )
    df["delay"] = df["delay"].where(df["delay"].between(0, MAX_MIN))
    df = df.sort_values(["CallNumber", "seq", "delay", "UnitID"], kind="mergesort", na_position="last")
    g = df.groupby("CallNumber", sort=False)
    per_call = g.agg(units=("UnitID", "size"), als=("ALSUnit", lambda s: int((s == "true").sum())), first=("delay", "min"), last=("delay", "max"))
    calls = g.head(1).set_index("CallNumber").join(per_call)
    calls = calls[calls["lat"].between(BOX[0], BOX[1]) & calls["lon"].between(BOX[2], BOX[3])]
    calls = calls.reset_index().sort_values(["CallDate", "CallNumber"], kind="mergesort").reset_index(drop=True)
    day = pd.to_datetime(calls["CallDate"], format="%Y-%m-%d")
    night = day > pd.to_datetime(calls["WatchDate"], format="%Y-%m-%d")  # past midnight, still the previous watch
    out = pd.DataFrame({"call_date": day.dt.strftime("%Y-%m-%d %H:%M:%S"), "lat": calls["lat"], "lon": calls["lon"]})
    out["first_enroute_min"] = calls["first"].round(2)
    out["last_enroute_min"] = calls["last"].round(2)
    out["units_dispatched"] = calls["units"].astype("int64")
    out["als_units"] = calls["als"].astype("int64")
    out["call_type"] = fold(calls["CallType"])
    out["call_type_group"] = calls["CallTypeGroup"].fillna("Unknown")
    out["final_priority"] = calls["FinalPriority"].map({"2": "2 non-emergency", "3": "3 emergency"}).fillna("Unknown")
    out["battalion"] = fold(calls["Battalion"].fillna("Unknown"))
    out["first_unit_type"] = fold(calls["UnitType"].str.title())
    out["supervisor_district"] = ("D" + calls["SupervisorDistrict"]).fillna("Unknown")
    out["neighborhood"] = calls["Neighborhood"].fillna("Unknown")
    out["day_of_week"] = day.dt.day_name()
    out["time_of_day"] = night.map({True: "00-07", False: "08-23"})
    return out
