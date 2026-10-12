"""Pick model settings by walk-forward validation on model.TUNE_SEASONS, then report once on the held-out seasons.

Run: python tune.py   (writes tuned_params.json, which model.py loads; takes several minutes)
"""
import datetime as dt
import itertools
import json
import os
from multiprocessing import Pool

import pandas as pd

import data
import features
import model

GRID = {"num_leaves": [4, 7, 15], "min_data_in_leaf": [20, 40, 80], "learning_rate": [0.03, 0.1], "rounds": [100, 300]}
DF = None


def _init(df):
    global DF
    DF = df


def _tune_score(config):
    scores = model.backtest(DF, model.TUNE_SEASONS, config)[0]
    # objective: rank correlation averaged over the before- and after-qualifying models
    return config, model.summary(scores).loc[["Model + grid", "Model, before qualifying"], "spearman"].mean(), scores


def report(title, scores):
    print(f"\n{title}")
    print(model.summary(scores).round(3).to_string())
    print(model.summary(scores, ["phase"])["spearman"].unstack()[model.METHODS].round(3).to_string())


def main():
    pd.set_option("display.width", 160)
    df, _ = features.build(data.load_history(dt.date.today()))
    configs = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    print(f"Tuning {len(configs)} configs on {model.TUNE_SEASONS.start}–{model.TUNE_SEASONS.stop - 1}, walk-forward race by race…")
    with Pool(os.cpu_count(), initializer=_init, initargs=(df,)) as pool:
        results = sorted(pool.imap_unordered(_tune_score, configs), key=lambda r: -r[1])
    print(pd.DataFrame([{**c, "spearman": s} for c, s, _ in results]).head(10).round(4).to_string(index=False))
    best = results[0][0]
    model.TUNED.write_text(json.dumps(best, indent=2) + "\n")
    print(f"\nBest: {best} -> {model.TUNED.name}")
    # held-out seasons: never looked at during tuning, so this is the honest accuracy
    test = range(model.TEST_FROM, int(df["season"].max()) + 1)
    no_reg = {**best, "exclude": model.REG_FEATURES}
    with Pool(3) as pool:
        tune_ablated, test_full, test_ablated = pool.starmap(
            model.backtest, [(df, model.TUNE_SEASONS, no_reg), (df, test, best), (df, test, no_reg)])
    report("Tuning seasons, best config", results[0][2])
    report("Tuning seasons, without regulation features", tune_ablated[0])
    report(f"HELD-OUT {test.start}–{test.stop - 1}, best config", test_full[0])
    report(f"HELD-OUT {test.start}–{test.stop - 1}, without regulation features", test_ablated[0])


if __name__ == "__main__":
    main()
