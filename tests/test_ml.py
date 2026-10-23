"""The ML correction: feature rows logged by the shadow planner, the pure-Python ridge, the daily trainer.
Run: python3 tests/test_ml.py"""
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
sys.path.insert(0, os.path.dirname(__file__))
import coupled  # noqa: E402
import mlmodel  # noqa: E402
import test_mpc  # noqa: E402


def test_ridge_recovers_a_linear_relation():
    rng = random.Random(1)
    X = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(400)]
    y = [2.0 * r[0] - 1.0 * r[2] + 0.3 + rng.gauss(0, 0.05) for r in X]
    m = mlmodel.Ridge(1.0).fit(X, y)
    for r, t in zip(X[:20], y[:20]):
        assert abs(m.predict(r) - t) < 0.2
    assert abs(mlmodel.Ridge.from_dict(m.to_dict()).predict(X[0]) - m.predict(X[0])) < 1e-12


def _rows(rng, n=600):
    """Rows for two rooms: in "a" the physics model's miss depends on a feature (the correction can learn it); in "b" it is noise."""
    rows = []
    for k in range(n):
        for room, learnable in (("a", True), ("b", False)):
            f = [rng.gauss(0, 1) for _ in range(len(mlmodel.FEATURES))]
            ode = f[0]
            miss = 0.8 * f[3] if learnable else rng.gauss(0, 0.5)
            rows.append([f"2026-10-20T{(k // 60) % 24:02d}:{k % 60:02d}:00", room, 60, ode + miss, f])
    return rows


def test_trainer_enables_a_correction_only_where_it_helps():
    c = mlmodel.train(_rows(random.Random(2)))
    assert c is not None and c.enabled["a|60"] and not c.enabled["b|60"], (c.enabled, c.report)
    f = [0.0] * len(mlmodel.FEATURES)
    f[3] = 1.0
    # one slope is shared by the pooled rooms (half of them have none), so room a gets about half its true 0.8
    assert 0.2 < c.correct("a", 60, f) < 0.9 and c.correct("b", 60, f) == 0.0
    d = mlmodel.MLCorrector.from_dict(c.to_dict())
    assert abs(d.correct("a", 60, f) - c.correct("a", 60, f)) < 1e-9


def test_shadow_planner_logs_rows_and_scores_the_correction():
    shadow, sim = test_mpc._shadow_run(coupled.the_house(), 16)
    rows = shadow.drain_ml_rows()
    assert len(rows) > 300 and shadow.ml_rows == [], len(rows)
    assert all(len(r[4]) == len(mlmodel.FEATURES) for r in rows)
    assert {r[2] for r in rows} == set(mlmodel.HORIZONS)
    c = mlmodel.train(rows, min_rows=60)
    assert c is not None
    shadow2, _ = test_mpc._shadow_run(coupled.the_house(), 6, setup=lambda sh: sh.set_corrector(c))
    acc = shadow2.accuracy()
    scored = [v[h] for v in acc.values() for h in v if "ml_rmse" in v[h]]
    assert scored, "the corrected predictions are scored next to the physics model's"
    st = shadow2.to_state()
    assert st["ml_errors"]
    shadow3 = test_mpc.mpc.ShadowPlanner()
    shadow3.load_state(st, shadow2.errors[-1][0] if shadow2.errors else shadow2.ml_errors[-1][0])
    assert len(shadow3.ml_errors) > 0


if __name__ == "__main__":
    import traceback
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for f in fns:
        try:
            f()
        except Exception:
            bad += 1
            print(f.__name__)
            traceback.print_exc()
    print(f"{len(fns) - bad} passed, {bad} failed")
    sys.exit(1 if bad else 0)
