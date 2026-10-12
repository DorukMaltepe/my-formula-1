"""LightGBM learning-to-rank model that orders the drivers within a race, plus walk-forward evaluation."""
import lightgbm as lgb
import pandas as pd

from features import FEATURES, GROUPS

# small trees and few rounds: ~3,400 rows overfit quickly (picked by walk-forward backtest)
PARAMS = dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=7, min_data_in_leaf=40,
              feature_fraction=0.8, seed=0, deterministic=True, verbose=-1)
ROUNDS = 100
MONOTONE = {"car_level": -1, "driver_form": -1, "grid": -1}  # a worse past position can only lower the score


def columns(with_grid):
    return [f for f in FEATURES if with_grid or f != "grid"]


def train(df, with_grid):
    # retirements are mostly bad luck, so learn the order of finishers; lambdarank needs each race's rows together
    df = df[df["finished"]].sort_values(["season", "round"], kind="stable")
    cols = columns(with_grid)
    groups = df.groupby(["season", "round"], sort=False).size().to_numpy()
    label = (21 - df["pos"]).clip(lower=0)  # winner gets the highest relevance; default gains weight the top places most
    params = {**PARAMS, "monotone_constraints": [MONOTONE.get(c, 0) for c in cols]}
    return lgb.train(params, lgb.Dataset(df[cols], label=label, group=groups), ROUNDS)


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


def backtest(df, season, first_round=4):
    """Predict each race of the season from a model trained only on the races before it, scored on the finishers."""
    scores, preds = [], []
    for rnd in sorted(df.loc[df["season"] == season, "round"].unique()):
        if rnd < first_round:
            continue
        before = (df["season"] < season) | ((df["season"] == season) & (df["round"] < rnd))
        race = df[(df["season"] == season) & (df["round"] == rnd)]
        fin = race[race["finished"]]
        pre, post = rank(train(df[before], False), race), rank(train(df[before], True), race)
        for method, order in (("Model, before qualifying", pre), ("Model + grid", post), ("Grid order only", race.sort_values("grid"))):
            order = [d for d in order["driver"] if d in set(fin["driver"])]
            scores.append(dict(round=rnd, race=race["race_name"].iloc[0], method=method, **score(order, fin)))
        preds.append(race[["round", "race_name", "name", "team", "grid", "pos", "finished"]]
                     .join(pre["predicted"].rename("pred_pre")).join(post["predicted"].rename("pred_grid")))
    return pd.DataFrame(scores), pd.concat(preds)
