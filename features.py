"""Turns result rows into one feature row per driver per race, built only from earlier races (no leakage)."""
import math
import statistics
from collections import defaultdict

import pandas as pd

REG_CHANGES = {2009, 2014, 2017, 2022, 2026}  # seasons that reset the technical regulations
ADAPT_RACES = 8  # races either side of a reset used to measure how well a team adapted
# constructorIds change with rebrands; a lineage keeps one team's car history together
LINEAGE = {"racing_point": "force_india", "aston_martin": "force_india", "alphatauri": "toro_rosso", "rb": "toro_rosso",
           "alfa": "sauber", "audi": "sauber", "renault": "lotus_f1", "alpine": "lotus_f1", "manor": "marussia"}

GROUPS = {
    "Car": ["car_level", "car_trend", "car_prev_season", "car_dnf_rate", "car_track",
            "reg_change", "rules_age", "season_races", "car_reg_adapt"],
    "Driver": ["driver_form", "teammate_delta", "driver_dnf_rate", "starts"],
    "Track & conditions": ["track_delta", "track_visits", "wet", "wet_delta"],
    "This weekend": ["practice", "grid"],
}
FEATURES = [f for fs in GROUPS.values() for f in fs]
LABELS = {
    "car_level": "Car pace this season",
    "car_trend": "Car development trend this season",
    "car_prev_season": "Car pace at end of last season",
    "reg_change": "Regulation reset season",
    "rules_age": "Races run under current rules",
    "season_races": "Races run this season",
    "car_reg_adapt": "Team's record at past rule resets",
    "car_dnf_rate": "Car retirement rate",
    "car_track": "Car fit for this circuit",
    "driver_form": "Driver recent finishes",
    "teammate_delta": "Gap to teammate",
    "driver_dnf_rate": "Driver retirement rate",
    "starts": "Driver experience (starts)",
    "track_delta": "Driver history at this circuit",
    "track_visits": "Driver visits to this circuit",
    "wet": "Wet race",
    "wet_delta": "Driver wet-weather record",
    "practice": "Practice pace this weekend",
    "grid": "Grid position",
}
NAN = math.nan


def ewma(values, half_life):
    """Average of values (oldest first) where weight halves every half_life entries back."""
    if not values:
        return NAN
    w = [0.5 ** ((len(values) - 1 - i) / half_life) for i in range(len(values))]
    return sum(a * b for a, b in zip(w, values)) / sum(w)


def mean(values):
    return statistics.fmean(values) if values else NAN


class History:
    """Everything known before the next race."""

    def __init__(self):
        self.team = defaultdict(list)    # team lineage -> one dict per race
        self.driver = defaultdict(list)  # driver -> one dict per race
        self.races = defaultdict(int)    # season -> races run
        self.adapt = defaultdict(list)   # team lineage -> places gained (-) or lost (+) at each past rule reset

    def features(self, driver, team, season, circuit, wet):
        lineage = LINEAGE.get(team, team)
        tr = self.team[lineage]
        # current and previous season kept apart so the model learns how much form carries over (little after a reset)
        car = [x["mean"] for x in tr if x["season"] == season and x["mean"] is not None]
        prev = [x["mean"] for x in tr if x["season"] == season - 1 and x["mean"] is not None]
        era = max(y for y in REG_CHANGES if y <= season)
        team_avg = {s: mean([x["mean"] for x in tr if x["season"] == s and x["mean"] is not None]) for s in {x["season"] for x in tr}}
        recent_team = tr[-10:]

        dh = self.driver[driver]
        fin = [x for x in dh if x["finished"]]
        season_avg = {s: mean([x["pos"] for x in fin if x["season"] == s]) for s in {x["season"] for x in fin}}
        here = [x for x in fin if x["circuit"] == circuit]
        weights = [1 if x["wet"] == wet else 0.5 for x in here]  # same conditions count double
        recent_car = car[-6:]
        return {
            "car_level": ewma(car, 3),
            "car_trend": statistics.linear_regression(range(len(recent_car)), recent_car).slope if len(recent_car) >= 3 else NAN,
            "car_prev_season": mean(prev[-10:]),
            "reg_change": int(season in REG_CHANGES),
            # shortcut: undercounts for the first loaded season's era (history starts mid-era), fine for training rows only
            "rules_age": sum(n for s, n in self.races.items() if s >= era),
            "season_races": self.races[season],
            "car_reg_adapt": mean(self.adapt[lineage]),
            "car_dnf_rate": sum(x["dnf"] for x in recent_team) / sum(x["cars"] for x in recent_team) if recent_team else NAN,
            "car_track": mean([x["mean"] - team_avg[x["season"]] for x in tr if x["circuit"] == circuit and x["mean"] is not None]),
            "driver_form": ewma([x["pos"] for x in fin], 5),
            "teammate_delta": ewma([x["tm_delta"] for x in fin], 8),
            "driver_dnf_rate": mean([0 if x["finished"] else 1 for x in dh[-10:]]),
            "starts": len(dh),
            "track_delta": sum(w * (x["pos"] - season_avg[x["season"]]) for w, x in zip(weights, here)) / sum(weights) if here else NAN,
            "track_visits": len(here),
            "wet": int(wet),
            "wet_delta": mean([x["pos"] - season_avg[x["season"]] for x in fin if x["wet"]]),
        }

    def add(self, race_rows):
        teams = defaultdict(list)
        for r in race_rows:
            teams[r["team"]].append(r)
        season = race_rows[0]["season"]
        self.races[season] += 1
        for team, rs in teams.items():
            fin = [r["pos"] for r in rs if r["finished"]]  # retirements say little about pace
            team_mean = statistics.fmean(fin) if fin else None
            self.team[LINEAGE.get(team, team)].append(dict(season=rs[0]["season"], circuit=rs[0]["circuit"], mean=team_mean,
                                        dnf=len(rs) - len(fin), cars=len(rs)))
            for r in rs:
                self.driver[r["driver"]].append(dict(season=r["season"], circuit=r["circuit"], wet=r["wet"], pos=r["pos"],
                                                     finished=r["finished"], tm_delta=r["pos"] - team_mean if r["finished"] else None))
        if season in REG_CHANGES and self.races[season] == ADAPT_RACES:  # first stretch of a new era is complete
            for lineage, tr in self.team.items():
                before = [x["mean"] for x in tr if x["season"] == season - 1 and x["mean"] is not None][-ADAPT_RACES:]
                after = [x["mean"] for x in tr if x["season"] == season and x["mean"] is not None]
                if before and after:
                    self.adapt[lineage].append(statistics.fmean(after) - statistics.fmean(before))


def build(rows):
    """Training frame (features + finishing position for every past entry) and the History after the last race."""
    races = defaultdict(list)
    for r in rows:
        races[r["season"], r["round"]].append(r)
    history, out = History(), []
    for (season, rnd), rs in sorted(races.items()):
        for r in rs:
            out.append(dict(season=season, round=rnd, race_name=r["race"], driver=r["driver"], name=r["name"], team=r["team"],
                            **history.features(r["driver"], r["team"], season, r["circuit"], r["wet"]),
                            practice=r.get("practice", NAN), number=r.get("number"),
                            grid=r["grid"] or len(rs), pos=r["pos"], finished=r["finished"]))  # grid 0 = pit-lane start
        history.add(rs)
    return pd.DataFrame(out), history


def upcoming(history, entries, season, circuit, wet):
    """Feature frame for a race that hasn't run; entries: [{driver, name, team, practice?, grid?}]."""
    return pd.DataFrame([dict(driver=e["driver"], name=e["name"], team=e["team"],
                              **history.features(e["driver"], e["team"], season, circuit, wet),
                              practice=e.get("practice", NAN), grid=e.get("grid", NAN))
                         for e in entries])
