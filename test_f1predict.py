"""Offline checks of the model on synthetic races. Run: python test_f1predict.py"""
from f1predict import predict


def race(rnd, results, circuit="x", wet=False, season=2026):
    """results: [(driver, team)] in finishing order."""
    return [dict(season=season, round=rnd, circuit=circuit, wet=wet, driver=d, name=d, team=t, grid=i, pos=i, finished=True)
            for i, (d, t) in enumerate(results, 1)]


def order(history, field, circuit="x", wet=False):
    return [d for _, d, _ in predict(history, field, circuit, wet)]


FIELD = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}


def test_faster_car_and_teammate_gap():
    hist = sum((race(r, [("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B")]) for r in range(1, 6)), [])
    assert order(hist, FIELD) == ["a1", "a2", "b1", "b2"]


def test_upgrade_trend_overtakes_static_car():
    # B starts behind and gains every race; by the last race it is level with A
    seq = [[("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B")],
           [("a1", "A"), ("b1", "B"), ("a2", "A"), ("b2", "B")],
           [("b1", "B"), ("a1", "A"), ("b2", "B"), ("a2", "A")],
           [("b1", "B"), ("a1", "A"), ("b2", "B"), ("a2", "A")]]
    hist = sum((race(r, s) for r, s in enumerate(seq, 1)), [])
    assert order(hist, FIELD)[0] == "b1"


def test_track_and_rain_specialist():
    base = [("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B")]
    hist = sum((race(r, base) for r in range(1, 6)), [])
    # b2 won here last year in the wet, and only in the wet
    hist = race(1, [("b2", "B"), ("a1", "A"), ("a2", "A"), ("b1", "B")], circuit="spa", wet=True, season=2025) + \
        sum((race(r, base, season=2025) for r in range(2, 6)), []) + hist
    dry, wet = order(hist, FIELD, "spa"), order(hist, FIELD, "spa", wet=True)
    assert dry.index("b2") >= wet.index("b2") and wet.index("b2") < wet.index("b1")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
