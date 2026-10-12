"""Predict the next F1 race finishing order.

Score per driver (in finishing-position units, lower is better):
  car    - team's recency-weighted average finish this season, plus its recent improvement trend (upgrades)
  driver - recency-weighted gap to teammate (separates driver skill from the car)
  track  - driver's past results at this circuit vs their season average, conditions-matched
  rain   - driver's wet-race over/under-performance, applied only when rain is expected
Once qualifying has run, the score is blended with the grid position.

Data: Jolpica (Ergast successor) for results, Open-Meteo for rain during the race.
Usage: python f1predict.py [--rain auto|wet|dry] [--backtest]
"""
import argparse
import datetime as dt
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

API = "https://api.jolpi.ca/ergast/f1"
CACHE = Path(__file__).with_name(".cache")
SEASONS = 4             # seasons of history to load, including the current one
CAR_HALF_LIFE = 3       # races; short so upgrades show up quickly
DRIVER_HALF_LIFE = 8
TREND_WINDOW = 6        # races used to measure the car's development trend
TREND_WEIGHT = 1.0      # races of trend projected forward
TRACK_WEIGHT = 0.5
GRID_WEIGHT = 0.5       # blend with grid once qualifying is known
SHRINK = 3              # pseudo-races pulling sparse driver/track/rain stats toward 0
WET_MM = 0.5            # rain during the race window that makes it a wet race


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
    return [dict(season=int(race["season"]), round=int(race["round"]), circuit=race["Circuit"]["circuitId"], wet=wet,
                 driver=r["Driver"]["driverId"], name=f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                 team=r["Constructor"]["constructorId"], grid=int(r["grid"]), pos=int(r["position"]),
                 finished=r["status"] in ("Finished", "Lapped") or r["status"].startswith("+"))
            for r in race["Results"]]


def wmean(pairs, prior=0):
    """Weighted mean of (weight, value) pairs; prior adds zero-valued pseudo-weight (shrinkage)."""
    total = sum(w for w, _ in pairs) + prior
    return sum(w * v for w, v in pairs) / total if total else 0.0


def predict(history, field, circuit, wet, grid=None):
    """history: result rows before the race; field: {driver: team}; grid: {driver: grid pos} if known.
    Returns [(score, driver, parts)] best first."""
    races = sorted({(r["season"], r["round"]) for r in history})
    age = {k: len(races) - 1 - i for i, k in enumerate(races)}
    fin = [r for r in history if r["finished"]]  # DNFs say little about pace

    team_pos, season_pos = defaultdict(list), defaultdict(list)
    for r in fin:
        team_pos[r["team"], r["season"], r["round"]].append(r["pos"])
        season_pos[r["driver"], r["season"]].append(r["pos"])
    team_mean = {k: statistics.fmean(v) for k, v in team_pos.items()}
    season_avg = {k: statistics.fmean(v) for k, v in season_pos.items()}

    # car: a new season resets the pecking order, so use only it once it has 3 races
    latest = races[-1][0]
    car_seasons = {latest} if sum(s == latest for s, _ in races) >= 3 else {s for s, _ in races}
    car = {}
    for team in set(field.values()):
        pts = sorted((age[s, rd], m) for (t, s, rd), m in team_mean.items() if t == team and s in car_seasons)
        if not pts:
            continue
        level = wmean([(0.5 ** (a / CAR_HALF_LIFE), m) for a, m in pts])
        recent = pts[:TREND_WINDOW]
        slope = statistics.linear_regression([-a for a, _ in recent], [m for _, m in recent]).slope if len(recent) >= 3 else 0
        car[team] = level + TREND_WEIGHT * slope
    default_car = statistics.fmean(car.values()) if car else 10.0

    out = []
    for d, team in field.items():
        mine = [r for r in fin if r["driver"] == d]
        delta = lambda r: r["pos"] - season_avg[d, r["season"]]
        parts = {
            "car": car.get(team, default_car),
            "driver": wmean([(0.5 ** (age[r["season"], r["round"]] / DRIVER_HALF_LIFE),
                              r["pos"] - team_mean[r["team"], r["season"], r["round"]]) for r in mine], SHRINK),
            "track": TRACK_WEIGHT * wmean([(1 if r["wet"] == wet else 0.5, delta(r)) for r in mine if r["circuit"] == circuit], SHRINK),
            "rain": wmean([(1, delta(r)) for r in mine if r["wet"]], SHRINK) if wet else 0.0,
        }
        score = sum(parts.values())
        if grid and d in grid:
            score = (1 - GRID_WEIGHT) * score + GRID_WEIGHT * grid[d]
        out.append((score, d, parts))
    return sorted(out, key=lambda x: x[0])


def load(today):
    rows = []
    for season in range(today.year - SEASONS + 1, today.year + 1):
        for race in season_races(season, cache=season != today.year):
            rows += to_rows(race, rained(race, today))
    return rows


def backtest(rows):
    season = max(r["season"] for r in rows)
    scores = defaultdict(list)
    print(f"Backtest {season}: Spearman rank correlation with actual finishers (1 = perfect)")
    print(f"{'round':>5} {'model':>6} {'+grid':>6} {'grid':>6}")
    for rnd in sorted({r["round"] for r in rows if r["season"] == season}):
        race = [r for r in rows if (r["season"], r["round"]) == (season, rnd)]
        hist = [r for r in rows if (r["season"], r["round"]) < (season, rnd)]
        if rnd < 4 or not race:
            continue
        field = {r["driver"]: r["team"] for r in race}
        grid = {r["driver"]: r["grid"] or len(race) for r in race}  # grid 0 = pit-lane start
        actual = {r["driver"]: r["pos"] for r in race if r["finished"]}
        corr = lambda ranked: statistics.correlation([dict((d, s) for s, d, _ in ranked)[d] for d in actual],
                                                     list(actual.values()), method="ranked")
        row = {"model": corr(predict(hist, field, race[0]["circuit"], race[0]["wet"])),
               "+grid": corr(predict(hist, field, race[0]["circuit"], race[0]["wet"], grid)),
               "grid": statistics.correlation([grid[d] for d in actual], list(actual.values()), method="ranked")}
        for k, v in row.items():
            scores[k].append(v)
        print(f"{rnd:>5} " + " ".join(f"{v:>6.2f}" for v in row.values()))
    print(f"{'mean':>5} " + " ".join(f"{statistics.fmean(v):>6.2f}" for v in scores.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rain", choices=["auto", "wet", "dry"], default="auto")
    ap.add_argument("--backtest", action="store_true", help="score the model on this season's past races")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # driver names like Hülkenberg on Windows consoles
    today = dt.date.today()
    rows = load(today)
    if args.backtest:
        return backtest(rows)

    races = get(f"{API}/current/next.json", cache=False)["MRData"]["RaceTable"]["Races"]
    if not races:
        raise SystemExit("No upcoming race in the current season.")
    race = races[0]
    days_out = (dt.date.fromisoformat(race["date"]) - today).days
    if args.rain != "auto":
        wet, why = args.rain == "wet", "set by --rain"
    elif days_out <= 15:
        wet, why = rained(race, today), "forecast"
    else:
        wet, why = False, "too far out for a forecast, assuming dry (override with --rain)"

    names = {r["driver"]: r["name"] for r in rows}
    last = max((r["season"], r["round"]) for r in rows)
    field = {r["driver"]: r["team"] for r in rows if (r["season"], r["round"]) == last}
    grid = None
    quali = get(f"{API}/{race['season']}/{race['round']}/qualifying.json", cache=False)["MRData"]["RaceTable"]["Races"]
    if quali:
        q = quali[0]["QualifyingResults"]
        field = {r["Driver"]["driverId"]: r["Constructor"]["constructorId"] for r in q}
        grid = {r["Driver"]["driverId"]: int(r["position"]) for r in q}
        names |= {r["Driver"]["driverId"]: f"{r['Driver']['givenName']} {r['Driver']['familyName']}" for r in q}

    print(f"{race['season']} {race['raceName']} ({race['date']}) - {'WET' if wet else 'dry'} ({why})"
          f"{', grid known' if grid else ', qualifying not run yet'}\n")
    print(f"{'P':>2}  {'driver':<22}{'team':<14}{'score':>6}{'car':>6}{'drv':>6}{'track':>6}{'rain':>6}")
    for i, (score, d, p) in enumerate(predict(rows, field, race["Circuit"]["circuitId"], wet, grid), 1):
        print(f"{i:>2}  {names.get(d, d):<22}{field[d]:<14}{score:>6.1f}"
              f"{p['car']:>6.1f}{p['driver']:>+6.1f}{p['track']:>+6.1f}{p['rain']:>+6.1f}")
    print("\nscore ~ expected finishing position; drv/track/rain are +/- positions vs the car's level")


if __name__ == "__main__":
    main()
