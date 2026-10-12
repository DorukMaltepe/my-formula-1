"""LightGBM learning-to-rank model that orders the drivers within a race, plus walk-forward evaluation."""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from features import FEATURES, GROUPS, REG_CHANGES

TUNE_SEASONS = range(2017, 2024)  # tune.py picks settings on these (includes the 2017 and 2022 resets)
TEST_FROM = 2024                  # seasons from here on are held out: only ever used to report accuracy
REG_FEATURES = ["car_prev_season", "reg_change", "rules_age", "season_races", "car_reg_adapt"]
# defaults; tune.py writes better ones to tuned_params.json. "rounds" = boosting rounds, "exclude" = features left out
CONFIG = dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=7, min_data_in_leaf=40,
              feature_fraction=0.8, seed=0, deterministic=True, num_threads=1, verbose=-1, rounds=100)
TUNED = Path(__file__).with_name("tuned_params.json")
if TUNED.exists():
    CONFIG |= json.loads(TUNED.read_text())
MONOTONE = {"car_level": -1, "driver_form": -1, "grid": -1}  # a worse past position can only lower the score
METHODS = ["Model + grid", "Model, before qualifying", "Grid order only"]


def train(df, with_grid, config=None):
    params = {**CONFIG, **(config or {})}
    rounds, exclude = params.pop("rounds"), set(params.pop("exclude", ()))
    cols = [f for f in FEATURES if f not in exclude and (with_grid or f != "grid")]
    # retirements are mostly bad luck, so learn the order of finishers; lambdarank needs each race's rows together
    df = df[df["finished"]].sort_values(["season", "round"], kind="stable")
    groups = df.groupby(["season", "round"], sort=False).size().to_numpy()
    label = (21 - df["pos"]).clip(lower=0)  # winner gets the highest relevance; default gains weight the top places most
    params["monotone_constraints"] = [MONOTONE.get(c, 0) for c in cols]
    return lgb.train(params, lgb.Dataset(df[cols], label=label, group=groups), rounds)


def rank(model, X):
    """X sorted into predicted finishing order, with SHAP contributions per feature group (positive = finishes higher)."""
    cols = model.feature_name()
    contrib = model.predict(X[cols], pred_contrib=True)[:, :-1]  # last column is the bias term
    out = X.assign(score=contrib.sum(1))
    for group, feats in GROUPS.items():
        out[group] = contrib[:, [cols.index(f) for f in feats if f in cols]].sum(1)
    out = out.sort_values("score", ascending=False)
    return out.assign(predicted=range(1, len(out) + 1))


def importance(model):
    gain = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    group = {f: g for g, fs in GROUPS.items() for f in fs}
    return pd.DataFrame({"feature": gain.index, "group": gain.index.map(group), "share": (gain / gain.sum()).to_numpy()})


def score(order, race):
    """Compare a predicted driver order with the actual result."""
    actual = race.sort_values("pos")["driver"].tolist()
    predicted = {d: i for i, d in enumerate(order, 1)}
    return dict(spearman=pd.Series([predicted[d] for d in race["driver"]]).rank().corr(race["pos"].reset_index(drop=True).rank()),
                winner=order[0] == actual[0], podium=len(set(order[:3]) & set(actual[:3])))


def phase(season, rnd):
    if season not in REG_CHANGES:
        return "Stable rules"
    return "Rule reset, rounds 1–5" if rnd <= 5 else "Rule reset, round 6+"


def backtest(df, seasons, config=None):
    """Walk-forward: predict every race of the given seasons with models trained only on earlier races.
    Scored on the finishers. Returns (scores, predictions)."""
    scores, preds = [], []
    for (season, rnd), race in df[df["season"].isin(seasons)].groupby(["season", "round"]):
        before = (df["season"] < season) | ((df["season"] == season) & (df["round"] < rnd))
        fin = race[race["finished"]]
        pre, post = (rank(train(df[before], g, config), race) for g in (False, True))
        for method, order in zip(METHODS, (post, pre, race.sort_values("grid"))):
            order = [d for d in order["driver"] if d in set(fin["driver"])]
            scores.append(dict(season=season, round=rnd, race=race["race_name"].iloc[0], phase=phase(season, rnd),
                               method=method, **score(order, fin)))
        preds.append(race[["season", "round", "race_name", "name", "team", "grid", "pos", "finished"]]
                     .join(pre["predicted"].rename("pred_pre")).join(post["predicted"].rename("pred_grid")))
    return pd.DataFrame(scores), pd.concat(preds)


def summary(scores, by=()):
    out = scores.groupby([*by, "method"])[["spearman", "winner", "podium"]].mean().reset_index()
    out = out.sort_values([*by, "method"], key=lambda c: c.map(METHODS.index) if c.name == "method" else c)
    return out.set_index([*by, "method"])
