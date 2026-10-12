"""Downloads F1 results (Jolpica) and race-time rain (Open-Meteo) as flat rows, cached on disk."""
import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.jolpi.ca/ergast/f1"
CACHE = Path(__file__).with_name(".cache")
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


def to_rows(race, wet):
    return [dict(season=int(race["season"]), round=int(race["round"]), race=race["raceName"],
                 circuit=race["Circuit"]["circuitId"], wet=wet,
                 driver=r["Driver"]["driverId"], name=f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                 team=r["Constructor"]["constructorId"], grid=int(r["grid"]), pos=int(r["position"]),
                 finished=r["status"] in ("Finished", "Lapped") or r["status"].startswith("+"))
            for r in race["Results"]]


def load_history(today, seasons):
    rows = []
    for season in range(today.year - seasons + 1, today.year + 1):
        for race in season_races(season, cache=season != today.year):
            rows += to_rows(race, rained(race, today))
    return rows


def next_race(today):
    """(race, entries, wet): entries has grid positions once qualifying has run, else None;
    wet is the forecast, or None when the race is beyond the 16-day forecast range."""
    races = get(f"{API}/current/next.json", cache=False)["MRData"]["RaceTable"]["Races"]
    if not races:
        return None, None, None
    race = races[0]
    quali = get(f"{API}/{race['season']}/{race['round']}/qualifying.json", cache=False)["MRData"]["RaceTable"]["Races"]
    entries = [dict(driver=r["Driver"]["driverId"], name=f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                    team=r["Constructor"]["constructorId"], grid=int(r["position"]))
               for r in quali[0]["QualifyingResults"]] if quali else None
    wet = rained(race, today) if (dt.date.fromisoformat(race["date"]) - today).days <= 15 else None
    return race, entries, wet
