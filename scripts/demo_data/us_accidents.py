"""US Accidents: the traffic incidents of Dallas County, Texas, in 2021–2022, one feed (Source1), one row per incident —
its start point, start and end times, the road extent it blocked and the airport weather near its start.

Kaggle ``sobhanmoosavi/us-accidents`` (Moosavi et al., 2019), CC BY-NC-SA 4.0: the sample keeps the licence. The feeds
changed over the years (Source2 reports a distance of 0 for about 91 % of its rows; Source1 grows from 2020), so the
sample keeps Source1 alone, two complete calendar years without a gap month, one county.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SLUG = "sobhanmoosavi/us-accidents"
FILES = {"US_Accidents_March23.csv": "e3e9f962e3e2289db1bdce91623abbcf9ba4d7bfc1adb091c8aeee6c2639d204"}
URLS = {name: f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}/{name}" for name in FILES}
META = {
    "title": "ДТП у США: округ Даллас, 2021–2022",
    "description": "Дорожні інциденти округу Даллас (Техас) з тривалістю, довжиною затору й погодою: які типи доріг, погода, "
    "час доби чи міста подовжують інциденти.",
    "license": "CC BY-NC-SA 4.0 (некомерційне використання, на тих самих умовах)",
    "source": "Moosavi et al., US Accidents (March 2023 release), через Kaggle sobhanmoosavi/us-accidents",
    "attribution": 'Moosavi, Samavatian, Parthasarathy, Ramnath, "A Countrywide Traffic Accident Dataset", 2019 (arXiv:1906.05409); '
    'Moosavi, Samavatian, Parthasarathy, Teodorescu, Ramnath, "Accident Risk Prediction based on Heterogeneous Sparse Data: New '
    'Dataset and Insights", ACM SIGSPATIAL 2019 (arXiv:1909.09638). This sample is redistributed under CC BY-NC-SA 4.0.',
    "prepared": "округ Даллас, джерело Source1, 2021–2022; повторні трансляції одного інциденту злито (той самий початок і вулиця: "
    "найпізніше завершення); неправдоподібні значення — порожні",
    "options": {"time": "start_time", "geo": "lat,lon@auto", "table": True, "source": "US Accidents (Moosavi et al.), Source1"},
}
YEARS, STATE, COUNTY, FEED = ("2021", "2022"), "TX", "Dallas", "Source1"
BOX = (32.53, 33.00, -97.05, -96.51)  # Dallas County: drops no point of the window
MIN_CITY_ROWS = 100  # a rarer city reads as "Other city"
FLAGS = [
    *("Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop"),
    *("Traffic_Calming", "Traffic_Signal", "Turning_Loop"),
]
USE = [
    *("ID", "Source", "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng", "Distance(mi)", "Street", "City", "County"),
    *("State", "Temperature(F)", "Humidity(%)", "Pressure(in)", "Wind_Speed(mph)", "Weather_Condition", "Sunrise_Sunset", *FLAGS),
]
WEATHER = [  # the first match wins
    ("Thunderstorm", r"T-Storm|Thunder|Hail"),
    ("Snow / ice", r"Snow|Sleet|Freezing|Wintry|Ice"),
    ("Rain", r"Rain|Drizzle|Showers"),
    ("Fog / haze / dust", r"Fog|Haze|Mist|Dust|Smoke"),
    ("Cloudy", r"Cloud|Overcast"),
    ("Clear", r"Fair|Clear"),
]
FEATURE = [
    ("Signal", "Traffic_Signal"),
    ("Stop sign", "Stop"),
    ("Junction", "Junction"),
    ("Crossing", "Crossing"),
    ("Railway", "Railway"),
    ("Station", "Station"),
]
ROADS = [("Freeway / tollway", r"^I-\d|^US-\d|Fwy|Freeway|Expy|Expressway|Tollway|Tpke"), ("Highway / loop", r"^TX-|^SH|Loop|Spur|Hwy|Highway")]


def first_match(text: pd.Series, rules: list[tuple[str, str]], default: str) -> pd.Series:
    out = pd.Series(default, index=text.index)
    for label, pattern in reversed(rules):
        out = out.mask(text.str.contains(pattern, regex=True, na=False), label)
    return out


def prepare(raw_dir: Path) -> pd.DataFrame:
    """The 3 GB file read in chunks, the window's Source1 rows of Dallas County kept, then one row per incident: six
    metrics (duration, blocked distance, the weather's temperature, humidity, pressure and wind), eight categoricals."""
    reader = pd.read_csv(raw_dir / "US_Accidents_March23.csv", usecols=USE, dtype=str, chunksize=500_000)
    keep = lambda c: c[(c["State"] == STATE) & (c["County"] == COUNTY) & (c["Source"] == FEED) & c["Start_Time"].str[:4].isin(YEARS)]  # noqa: E731
    df = pd.concat([keep(c) for c in reader]).drop_duplicates(subset=[c for c in USE if c != "ID"])
    df["start"] = pd.to_datetime(df["Start_Time"].str[:19], format="%Y-%m-%d %H:%M:%S")  # some carry ".000000000"
    df["end"] = pd.to_datetime(df["End_Time"].str[:19], format="%Y-%m-%d %H:%M:%S")
    df["num"] = df["ID"].str[2:].astype(int)
    # the feed rebroadcasts an incident under new IDs with a revised end and extent: one incident is the same start
    # second on the same street (the start point when the street is missing); its last end stays, then the lowest ID
    df["event"] = df["Street"].str.strip().fillna(df["Start_Lat"] + "," + df["Start_Lng"])
    df = df.sort_values(["start", "event", "end", "num"], ascending=[True, True, False, True], kind="mergesort")
    df = df.drop_duplicates(["start", "event"]).sort_values(["start", "num"], kind="mergesort").reset_index(drop=True)
    lat, lon = pd.to_numeric(df["Start_Lat"]).round(6), pd.to_numeric(df["Start_Lng"]).round(6)
    inside = lat.between(BOX[0], BOX[1]) & lon.between(BOX[2], BOX[3])
    df, lat, lon = df[inside].reset_index(drop=True), lat[inside].reset_index(drop=True), lon[inside].reset_index(drop=True)
    num = lambda col, lo, hi: pd.to_numeric(df[col], errors="coerce").pipe(lambda x: x.where(x.between(lo, hi)))  # noqa: E731
    out = pd.DataFrame({"start_time": df["start"].dt.strftime("%Y-%m-%d %H:%M:%S"), "lat": lat, "lon": lon})
    duration = (df["end"] - df["start"]).dt.total_seconds() / 60
    out["duration_min"] = duration.where(duration.between(1, 1440)).round(1)  # beyond a day: a stale closure
    out["distance_mi"] = num("Distance(mi)", 0, 50).round(3)
    out["temperature_f"] = num("Temperature(F)", -30, 130)
    out["humidity_pct"] = num("Humidity(%)", 1, 100)
    out["pressure_in"] = num("Pressure(in)", 25, 32)
    out["wind_speed_mph"] = num("Wind_Speed(mph)", 0, 100)  # 812 mph seen
    out["severity"] = "Severity " + df["Severity"]
    out["weather"] = first_match(df["Weather_Condition"], WEATHER, "Unknown")
    out["daylight"] = df["Sunrise_Sunset"].fillna("Unknown")
    flags = df[FLAGS].eq("True")
    out["road_feature"] = pd.Series("No feature", index=df.index).mask(flags.any(axis=1), "Other feature")
    for label, flag in reversed(FEATURE):  # traffic control first
        out["road_feature"] = out["road_feature"].mask(flags[flag], label)
    street = df["Street"].str.strip()
    out["road_type"] = first_match(street, ROADS, "Street").where(street.notna(), "Unknown")
    counts = df["City"].value_counts()
    out["city"] = df["City"].where(df["City"].isin(counts[counts >= MIN_CITY_ROWS].index), "Other city").fillna("Other city")
    out["day_of_week"] = df["start"].dt.day_name()
    out["hour_band"] = pd.cut(df["start"].dt.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)
    return out
