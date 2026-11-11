"""The moisture model: vapour pressure maths and the fit. Run: python3 tests/test_moisture.py"""
import math
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))
import moisture as mo  # noqa: E402


def test_vapour_pressure_round_trips_and_a_warmer_room_has_a_lower_rh_for_the_same_air():
    for t, rh in ((50, 90), (67, 60), (75, 35)):
        assert abs(mo.rh_from(t, mo.pv_from(t, rh)) - rh) < 1e-9
    pv = mo.pv_from(67, 60)
    assert mo.rh_from(72, pv) < 60 < mo.rh_from(62, pv)  # the same moisture: warmer is drier in RH terms
    assert abs(mo.dew_point_f(68, 100) - 68) < 0.1 and 50 < mo.dew_point_f(67, 60) < 58


def test_nnls_respects_the_sign_constraints():
    G = [[2.0, 0.0], [0.0, 2.0]]
    b = mo.nnls_gram(G, [-1.0, 2.0], [0.0, 0.0], ridge=0.0)  # the first wants to be negative: held at 0
    assert b[0] == 0.0 and abs(b[1] - 1.0) < 1e-6
    free = mo.nnls_gram(G, [-1.0, 2.0], [-1e12, 0.0], ridge=0.0)
    assert abs(free[0] + 0.5) < 1e-6


def _house(days=8, seed=3):
    """Two rooms of a synthetic house with known moisture dynamics, as minute-log rows."""
    rng = random.Random(seed)
    t0 = datetime.fromisoformat("2026-11-01T00:00:00-08:00")
    a = {"x": 0.30, "y": 0.10}  # ventilation per hour
    c = 0.5  # exchange between them
    src = {"x": 0.004, "y": 0.0}  # kPa per hour
    pvx, pvy = 1.2, 1.1
    rows = []
    for k in range(days * 24 * 60):
        t = t0 + timedelta(minutes=k)
        out_t = 50 + 5 * math.sin(2 * math.pi * k / (24 * 60))
        out_rh = 85 + 8 * math.sin(2 * math.pi * k / (7 * 24 * 60) + 1)
        pvo = mo.pv_from(out_t, out_rh)
        sn = 0.003 * math.sin(2 * math.pi * (t.hour + t.minute / 60) / 24)
        dt = 1 / 60
        dx = a["x"] * (pvo - pvx) + c * (pvy - pvx) + src["x"] + sn
        dy = a["y"] * (pvo - pvy) + c * (pvx - pvy) + src["y"]
        pvx += dt * dx + rng.gauss(0, 0.0003)
        pvy += dt * dy + rng.gauss(0, 0.0003)
        rows.append({"t": t.isoformat(), "outdoor": out_t, "out_rh": out_rh, "units": {"u": {"action": "idle"}},
                     "rooms": {"x": {"temp": 67.0, "rh": mo.rh_from(67.0, pvx)}, "y": {"temp": 66.0, "rh": mo.rh_from(66.0, pvy)}}})
    return rows


def test_the_fit_recovers_ventilation_and_beats_nothing_changes_on_held_out_data():
    rows = _house()
    S = mo.series(rows, ["x", "y"], ["u"])
    assert len(S["t"]) > 1000
    cut = int(len(S["t"]) * 0.75)
    head = {k: ({r: v[:cut] for r, v in x.items()} if isinstance(x, dict) else x[:cut]) for k, x in S.items()}
    m = mo.fit(head, ["x", "y"], ["u"], ridge=1e-4)
    assert abs(m.rooms["x"]["a"] - 0.30) < 0.1 and abs(m.rooms["y"]["a"] - 0.10) < 0.1
    assert 0.3 < m.rooms["x"]["c"]["y"] < 0.8
    ev = mo.evaluate(m, S, cut)
    for h in (60, 120):
        for r in ("x", "y"):
            model_rmse, base_rmse, n = ev[h][r]
            assert n > 20 and model_rmse < base_rmse  # the structure is right, so the model beats persistence


def test_not_enough_data_returns_none_and_model_round_trips():
    rows = _house(days=1)
    S = mo.series(rows, ["x", "y"], ["u"])
    assert mo.fit(S, ["x", "y"], ["u"], min_steps=500) is None
    m = mo.fit(S, ["x", "y"], ["u"], min_steps=50)
    m2 = mo.MoistureModel.from_dict(m.to_dict())
    assert m2.rooms.keys() == m.rooms.keys() and abs(m2.rooms["x"]["a"] - m.rooms["x"]["a"]) < 1e-12


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
