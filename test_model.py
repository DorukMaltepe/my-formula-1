"""Offline checks on synthetic races. Run: python test_model.py"""
import features
import model

TEAMS = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C", "c2": "C"}


def race(rnd, order, circuit="x", wet=False, season=2026):
    return [dict(season=season, round=rnd, race=f"R{rnd}", circuit=circuit, wet=wet, driver=d, name=d, team=TEAMS[d],
                 grid=i, pos=i, finished=True) for i, d in enumerate(order, 1)]


def test_no_leakage():
    rows = sum((race(r, ["a1", "a2", "b1", "b2", "c1", "c2"]) for r in range(1, 7)), [])
    changed = rows[:-6] + race(6, ["c2", "c1", "b2", "b1", "a2", "a1"])  # rewrite the last race's result
    before, after = (features.build(r)[0].sort_values(["round", "driver"], ignore_index=True) for r in (rows, changed))
    cols = [c for c in features.FEATURES if c != "grid"]
    assert before[cols].equals(after[cols]),"a race's features must not depend on its own or later results"


def test_learns_car_and_driver_order():
    # A > B > C, and the "1" driver beats their teammate, with some shuffling so features vary
    orders = [["a1", "a2", "b1", "b2", "c1", "c2"], ["a1", "b1", "a2", "c1", "b2", "c2"], ["a1", "a2", "b1", "c1", "b2", "c2"]]
    rows = sum((race(r, orders[r % 3], season=s) for s in (2024, 2025, 2026) for r in range(1, 21)), [])
    df, history = features.build(rows)
    entries = [dict(driver=d, name=d, team=t) for d, t in TEAMS.items()]
    ranked = model.rank(model.train(df, False), features.upcoming(history, entries, 2026, "x", False))
    order = ranked["driver"].tolist()
    assert order[0] == "a1" and order[-1] == "c2", order


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
