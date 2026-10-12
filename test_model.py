"""Offline checks on synthetic races. Run: python test_model.py"""
import data
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


def test_practice_gaps():
    sessions = [dict(session_key=1, session_name="Day 1", meeting_key=9, date_start="2026-02-20T09:00"),  # testing: ignored
                dict(session_key=2, session_name="Practice 1", meeting_key=10, date_start="2026-03-06T01:30"),
                dict(session_key=3, session_name="Practice 2", meeting_key=10, date_start="2026-03-06T05:00"),
                dict(session_key=4, session_name="Race", meeting_key=10, date_start="2026-03-08T04:00")]
    results = [dict(session_key=2, driver_number=1, duration=100.0), dict(session_key=2, driver_number=44, duration=101.0),
               dict(session_key=3, driver_number=1, duration=90.0), dict(session_key=3, driver_number=44, duration=90.45),
               dict(session_key=3, driver_number=16, duration=None), dict(session_key=1, driver_number=16, duration=50.0)]
    real_get, data.get = data.get, lambda url, cache: sessions if "/sessions" in url else results
    try:
        n, gaps = data.practice(2026, False)["2026-03-08"]
    finally:
        data.get = real_get
    assert n == 2 and gaps[1] == 0 and abs(gaps[44] - 0.5) < 1e-9 and 16 not in gaps, (n, gaps)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
