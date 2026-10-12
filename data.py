"""Downloads F1 results (Jolpica), practice times (OpenF1) and race-time rain (Open-Meteo) as flat rows, cached on disk."""
import datetime as dt
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.jolpi.ca/ergast/f1"
OPENF1 = "https://api.openf1.org/v1"
OPENF1_FROM = 2023  # first season with OpenF1 session results
CACHE = Path(__file__).with_name(".cache")
FIRST_SEASON = 2013  # covers the 2014, 2017, 2022 and 2026 regulation resets
WET_MM = 0.5  # rain in the first three hours of the race that makes it a wet race


def get(url, cache=True):
    path = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if cache and path.exists():
        return json.loads(path.read_text())
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
            break
        except (urllib.error.URLError, ConnectionError) as e:  # retry rate limits and dropped connections
            if getattr(e, "code", 429) not in (429, 500, 502, 503) or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    time.sleep(0.3)  # Jolpica allows ~4 requests/s
    if cache:
        CACHE.mkdir(exist_ok=True)
        path.write_text(json.dumps(data))
    return data


def season_races(season, cache):
    races, offset = {}, 0
    while True:
        d = get(f"{API}/{season}/results.json?limit=100&offset={offset}", cache)["MRData"]
        for race in d["RaceTable"]["Races"]:  # a race can be split across pages
            races.setdefault(race["round"], {**race, "Results": []})["Results"] += race["Results"]
        offset += 100
        if offset >= int(d["total"]):
            return list(races.values())


def rained(race, today):
    loc = race["Circuit"]["Location"]
    day = dt.date.fromisoformat(race["date"])
    hour = int(race.get("time", "13:00:00Z")[:2])
    # archive lags a few days behind, so recent and upcoming races use the forecast API (which also serves the past ~3 months)
    recent = (today - day).days < 60
    base = "https://api.open-meteo.com/v1/forecast" if recent else "https://archive-api.open-meteo.com/v1/archive"
    d = get(f"{base}?latitude={loc['lat']}&longitude={loc['long']}&start_date={day}&end_date={day}"
            f"&hourly=precipitation&timezone=UTC", cache=not recent)
    return sum(x or 0 for x in d["hourly"]["precipitation"][hour:hour + 3]) > WET_MM


def practice(season, cache):
    """{race date: (practice sessions run, {car number: best gap to the session's fastest lap, %})} from OpenF1."""
    if season < OPENF1_FROM:
        return {}
    sessions = get(f"{OPENF1}/sessions?year={season}", cache)
    race_date = {s["meeting_key"]: s["date_start"][:10] for s in sessions if s["session_name"] == "Race"}
    fp = {s["session_key"]: s["meeting_key"] for s in sessions
          if s["session_name"].startswith("Practice") and s["meeting_key"] in race_date}  # skips pre-season testing
    if not fp:
        return {}
    results = get(f"{OPENF1}/session_result?meeting_key>={min(race_date)}"
                  f"&meeting_key<={max(race_date)}", cache)
    laps = {}  # session -> {car number: best lap in seconds}
    for r in results:
        if r["session_key"] in fp and isinstance(r.get("duration"), (int, float)):
            laps.setdefault(r["session_key"], {})[r["driver_number"]] = r["duration"]
    out = {}
    for session, times in laps.items():
        n, gaps = out.setdefault(race_date[fp[session]], (0, {}))
        fastest = min(times.values())
        for number, t in times.items():
            gaps[number] = min(gaps.get(number, math.inf), (t / fastest - 1) * 100)
        out[race_date[fp[session]]] = (n + 1, gaps)
    return out


def to_rows(race, wet, gaps):
    return [dict(season=int(race["season"]), round=int(race["round"]), race=race["raceName"],
                 circuit=race["Circuit"]["circuitId"], wet=wet, number=int(r["number"]),
                 driver=r["Driver"]["driverId"], name=f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                 team=r["Constructor"]["constructorId"], grid=int(r["grid"]), pos=int(r["position"]),
                 practice=gaps.get(int(r["number"]), math.nan),
                 finished=r["status"] in ("Finished", "Lapped") or r["status"].startswith("+"))
            for r in race["Results"]]


def load_history(today, first=FIRST_SEASON):
    rows = []
    for season in range(first, today.year + 1):
        cache = season != today.year
        fp = practice(season, cache)
        for race in season_races(season, cache):
            rows += to_rows(race, rained(race, today), fp.get(race["date"], (0, {}))[1])
    return rows


def next_race(today):
    """(race, entries, wet, practice): entries has grid positions once qualifying has run, else None;
    wet is the forecast, or None when the race is beyond the 16-day forecast range;
    practice is (sessions run so far, {car number: gap %})."""
    races = get(f"{API}/current/next.json", cache=False)["MRData"]["RaceTable"]["Races"]
    if not races:
        return None, None, None, (0, {})
    race = races[0]
    quali = get(f"{API}/{race['season']}/{race['round']}/qualifying.json", cache=False)["MRData"]["RaceTable"]["Races"]
    entries = [dict(driver=r["Driver"]["driverId"], name=f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                    team=r["Constructor"]["constructorId"], number=int(r["number"]), grid=int(r["position"]))
               for r in quali[0]["QualifyingResults"]] if quali else None
    wet = rained(race, today) if (dt.date.fromisoformat(race["date"]) - today).days <= 15 else None
    return race, entries, wet, practice(int(race["season"]), cache=False).get(race["date"], (0, {}))
