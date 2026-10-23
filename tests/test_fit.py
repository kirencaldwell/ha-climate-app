"""The fitter recovers the simulated house from its own logs. Run: python3 tests/test_fit.py"""
import os
import sys
from datetime import datetime, timedelta

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import coupled  # noqa: E402
import fit  # noqa: E402
import model as hm  # noqa: E402


def test_fit_recovers_the_simulated_house():
    truth = coupled.the_house()
    cfg = {"rooms": [{"id": r, "unit": u, "priority": 3, "default_temp": 67} for r, u in (
        ("kitchen", "air_handler"), ("living_room", "air_handler"), ("dining_area", "air_handler"),
        ("bedroom_1", "air_handler"), ("bedroom_2", "air_handler"),
        ("primary_bedroom", "minisplit_1"), ("bedroom_3", "minisplit_2"))],
        "units": [{"id": u} for u in truth.units], "settings": {"floor": 63.0}}
    rooms, units = coupled.rooms_units_from_config(cfg, truth)
    s = coupled.settings_from_config(cfg)
    start = datetime(2026, 10, 16)
    log = []
    coupled.run(truth, rooms, units, coupled.HeuristicController(rooms, units, s), start, 7,
                coupled.Weather(start, 9, mean=55, seed=11), s, log=log, seed=11)
    neighbors = {rid: sorted(set(r.coupling) | {j for j, o in truth.rooms.items() if rid in o.coupling})
                 for rid, r in truth.rooms.items()}
    layout = {"rooms": {rid: {"units": list(truth.units), "neighbors": neighbors[rid]} for rid in truth.rooms},
              "units": {u: {"zone": [r.id for r in rooms if r.unit == u]} for u in truth.units}}  # sensor rooms: detected
    # The truth has one temperature per room, so its parameters are what the one-state fit must recover; the full fit
    # (every room with a mass state, shrunk toward none) must predict as well (a mass state can stand in for some of
    # a unit's lag, which shifts its raw coefficients without changing what it predicts).
    full, full_rep = fit.fit(log, layout)
    assert full.meta["usable"]
    m, rep = fit.fit(log, layout, mass=False)
    assert m.meta["usable"]
    for rid, v in rep["holdout"].items():
        assert full_rep["holdout"][rid]["model_rmse_F"] <= v["model_rmse_F"] + 0.05, (rid, full_rep["holdout"][rid], v)
    refs = {u: (rep.get("fan_scale", {}).get(u) or {}).get("relative_to") or "auto" for u in truth.units}
    for rid, tr in truth.rooms.items():
        fr = m.rooms[rid]
        assert abs(fr.loss - tr.loss) < 0.25 * tr.loss, (rid, fr.loss, tr.loss)
        own = next(iter(tr.heat))
        at_ref = hm.fan_factor(truth.units[own], refs[own])  # effects are learned at the unit's reference fan speed
        # Bedroom 3's own heating overlaps with its idle-unit leak (both come from its unit while the house heats).
        htol = 0.35 if rid == "bedroom_3" else 0.25
        assert abs(fr.heat[own] - tr.heat[own] * at_ref) < htol * tr.heat[own] * at_ref, (rid, fr.heat, tr.heat, refs[own])
        # Looser for cooling: with the fan under control the minisplits mostly heat on quiet and cool on low, so a
        # faster fan and a stronger cooling effect are hard to tell apart; predictions (checked below) don't suffer.
        # (and the controller cools little in a mild simulated week, so cooling effects rest on few minutes)
        tol = 0.4 if refs[own] == "auto" else 0.5
        assert abs(fr.cool[own] - tr.cool[own] * at_ref) < tol * tr.cool[own] * at_ref, (rid, fr.cool, tr.cool, refs[own])
        others = [v for u, v in fr.heat.items() if u != own]
        assert max(others) < 0.1 * tr.heat[own], (rid, fr.heat)  # no effect where there is none
    for u in ("minisplit_1", "minisplit_2"):  # the fan speeds the outer loop used, relative to its reference speed
        got, true = m.units[u].fan_scale, truth.units[u].fan_scale
        ref = rep["fan_scale"][u]["relative_to"]
        assert got and all(abs(v - true.get(k, 1) / true.get(ref, 1)) <= 0.3 for k, v in got.items()), (u, ref, got)
    leak = m.rooms["bedroom_3"].leak.get("minisplit_2", 0.0)
    assert 1.0 < leak < 4.5, m.rooms["bedroom_3"]  # the idle-in-heat leak (true 2.5)...
    assert m.rooms["bedroom_3"].heat.get("air_handler", 0) < 1.0, m.rooms["bedroom_3"].heat  # ...not an air handler effect
    b3, pb = m.rooms["bedroom_3"].sun_dir, m.rooms["primary_bedroom"].sun_dir
    # It learns which way each room faces. (How strong: midday south sun and overhead sun rise and fall together, so
    # their split varies with the data; the direction doesn't.)
    assert b3["s"] > 0.8 and b3["s"] > 2 * max(b3["e"], b3["w"]), b3
    assert pb["w"] > 0.6 and pb["w"] > 2 * max(pb["e"], pb["s"]), pb
    up = m.rooms["primary_bedroom"].coupling.get("living_room", 0)
    assert 0.05 < up < 0.2, up  # the stairs (true 0.12)
    assert {u: v.sensor_rooms for u, v in m.units.items()} == {u: v.sensor_rooms for u, v in truth.units.items()}
    for rid, v in rep["holdout"].items():
        assert v["model_rmse_F"] < 0.5 * v["no_change_rmse_F"], (rid, v)


if __name__ == "__main__":
    test_fit_recovers_the_simulated_house()
    print("ok")


def test_rooms_without_data_are_left_out_not_poisonous():
    truth = coupled.the_house()
    cfg = {"rooms": [{"id": "kitchen", "unit": "air_handler", "default_temp": 67},
                     {"id": "primary_bedroom", "unit": "minisplit_1", "default_temp": 67}],
           "units": [{"id": u} for u in truth.units], "settings": {"floor": 63.0}}
    rooms, units = coupled.rooms_units_from_config(cfg, truth)
    s = coupled.settings_from_config(cfg)
    start = datetime(2026, 10, 16)
    log = []
    coupled.run(truth, rooms, units, coupled.HeuristicController(rooms, units, s), start, 3,
                coupled.Weather(start, 5, mean=55, seed=3), s, log=log, seed=3)
    layout = {"rooms": {r: {"units": list(truth.units), "neighbors": [o for o in ("kitchen", "primary_bedroom", "ghost") if o != r]}
                        for r in ("kitchen", "primary_bedroom", "ghost")},  # "ghost": configured, never reported
              "units": {u: {"zone": {"air_handler": ["kitchen"], "minisplit_1": ["primary_bedroom"]}.get(u, [])}
                        for u in truth.units}}
    m, rep = fit.fit(log, layout)
    assert set(m.rooms) == {"kitchen", "primary_bedroom"} and "not enough data" in rep["ghost"]["skipped"]
    assert "ghost" not in m.rooms["kitchen"].coupling
    # A moved sensor: its room's readings before `since` are left out. With a day left it's fitted (but nobody's
    # neighbour yet: it covers a third of the log); with half a day it waits.
    moved = start + timedelta(days=2)
    layout["rooms"]["primary_bedroom"]["since"] = moved.isoformat()
    m2, rep2 = fit.fit(log, layout)
    assert "primary_bedroom" in m2.rooms and "primary_bedroom" not in m2.rooms["kitchen"].coupling
    layout["rooms"]["primary_bedroom"]["since"] = (moved + timedelta(hours=12)).isoformat()
    m3, rep3 = fit.fit(log, layout)
    assert "primary_bedroom" not in m3.rooms and "not enough data" in rep3["primary_bedroom"]["skipped"]
    assert rep3["holdout"]["kitchen"]["n"] > 0  # the other rooms are still checked
    assert m3.units["minisplit_1"].lag_h == 0  # its own room isn't fitted, so no lag is learned for it


def test_backtest_runs_the_model_over_logged_history():
    truth = coupled.the_house()
    cfg = {"rooms": [{"id": "kitchen", "unit": "air_handler", "default_temp": 67},
                     {"id": "primary_bedroom", "unit": "minisplit_1", "default_temp": 67}],
           "units": [{"id": u} for u in truth.units], "settings": {"floor": 63.0}}
    rooms, units = coupled.rooms_units_from_config(cfg, truth)
    s = coupled.settings_from_config(cfg)
    start = datetime(2026, 10, 16)
    log = []
    coupled.run(truth, rooms, units, coupled.HeuristicController(rooms, units, s), start, 1,
                coupled.Weather(start, 3, mean=55, seed=4), s, log=log, seed=4)
    sub = coupled.the_house()
    sub.rooms = {k: v for k, v in sub.rooms.items() if k in ("kitchen", "primary_bedroom")}
    for r in sub.rooms.values():
        r.coupling = {j: c for j, c in r.coupling.items() if j in sub.rooms}
    bt = fit.backtest(sub, log)
    actual = {row["t"]: row["rooms"]["kitchen"]["temp"] for row in log}
    pts = bt["kitchen"]["60"]
    assert len(pts) >= 40 and len(bt["kitchen"]["120"]) >= 40
    errs = [p[1] - actual[p[0]] for p in pts]
    assert (sum(e * e for e in errs) / len(errs)) ** 0.5 < 1.0  # close to the truth (rooms it doesn't model add error)


if __name__ == "__main__":
    test_backtest_runs_the_model_over_logged_history()
    test_rooms_without_data_are_left_out_not_poisonous()
    print("ok")


def test_compressor_drive_ignores_zero_frequency_and_uses_power():
    assert fit.compressor_drive({"sensor.a_compressor_frequency": 0.0}) is None
    assert fit.compressor_drive({"sensor.a_compressor_frequency": 50.0}) == 0.5
    d = fit.compressor_drive({"sensor.a_compressor_frequency": 0.0, "sensor.a_input_power": 1500.0,
                              "sensor.b_input_power": 8.0})
    assert abs(d - 1500.0 / fit.POWER_REF_W) < 1e-9
    assert fit.compressor_drive({"sensor.a_input_power": "unavailable"}) is None


def test_outputs_learned_from_power_when_one_unit_runs():
    t0 = datetime(2026, 10, 19, 0, 0)
    rows = []
    for k in range(200):
        on = 20 <= k < 180
        gap = 1.0 + (k % 3)  # the setpoint gap varies, the draw doesn't: output doesn't depend on the gap
        rows.append({"t": (t0 + timedelta(minutes=k)).isoformat(),
                     "units": {"a": {"action": "heating" if on else "idle", "sp": 66 + gap, "temp": 66.0},
                               "b": {"action": "idle", "sp": 60, "temp": 66.0}},
                     "extra": {"sensor.x_input_power": 600.0 if on else 10.0}})
    out = fit.fit_outputs(rows, ["a", "b"])
    assert "b" not in out
    assert abs(out["a"]["k0"] - 0.2) < 0.01 and out["a"]["k1"] < 0.01
    # series uses the measured draw for the running unit
    _, _, _, _, Y, _, _, _ = fit.series(rows, [], ["a", "b"], {"a": hm.UnitModel(k0=0.2, k1=0.0, ymin=0.1)})
    assert abs(Y["a"][100] - 0.2) < 1e-9 and Y["b"][100] == 0.0


def test_rule_effect_keeps_the_default_scale():
    u = hm.UnitModel()
    assert abs(hm.rule_effect(4.0, u) - 4.0) < 1e-9
    weak = hm.UnitModel(k0=0.2, k1=0.0, ymin=0.15)  # runs at 0.2: an effect of 20 per unit output is 5 F/h
    assert abs(hm.rule_effect(20.0, weak) - 20.0 * 0.2 / 0.8) < 1e-9


def test_mass_state_fitted_by_paths():
    # A room whose air swings fast around a slower mass: heater on 20 min of every 90, cool outside.
    n, tau, km, loss, heat = 3000, 4.0, 3.0, 0.2, 12.0
    t, M = 66.0, 66.0
    T, out, sun, Yh = [], [], [], []
    for k in range(n):
        y = 1.0 if (k % 90) < 20 else 0.0
        T.append(round(t / 0.18) * 0.18)
        out.append(45.0)
        sun.append({})
        Yh.append(y)
        d = loss * (45 - t) + km * (M - t) + heat * y
        M += (t - M) * (1 - pow(2.718281828459045, -1 / (tau * 60)))
        t += d / 60
    rm = hm.RoomModel(loss=0.3, heat={"u": 5.0}, cool={"u": 0.0})
    Ts, Ye, Lk = {"r": T}, {"u": Yh}, {"u": [0.0] * n}
    one, rep1 = fit.refine_paths("r", rm, Ts, out, sun, Ye, Lk, 2400)
    two, rep2 = fit.refine_mass("r", one, Ts, out, sun, Ye, Lk, 2400)
    assert rep2 is not None and two.mass_k > 0.5
    assert rep2["path_rmse_F"] < 0.6 * rep1["path_rmse_F"]
    assert abs(hm.mass_step(two, 60.0, 70.0, two.mass_tau_h) - (60 + 10 * (1 - pow(2.718281828459045, -1)))) < 1e-6


def test_start_up_pulse_is_recovered_and_absent_when_there_is_none():
    truth = coupled.the_house()
    b3 = truth.rooms["bedroom_3"]
    b3.leak = {"minisplit_2": 1.0}
    b3.leak_pulse, b3.leak_pulse_tau_h = 8.0, 25 / 60  # a small leaky room: a fast pulse of heat after every compressor start
    cfg = {"rooms": [{"id": r, "unit": u, "priority": 3, "default_temp": 67} for r, u in (
        ("kitchen", "air_handler"), ("living_room", "air_handler"), ("dining_area", "air_handler"),
        ("bedroom_1", "air_handler"), ("bedroom_2", "air_handler"),
        ("primary_bedroom", "minisplit_1"), ("bedroom_3", "minisplit_2"))],
        "units": [{"id": u} for u in truth.units], "settings": {"floor": 63.0}}
    rooms, units = coupled.rooms_units_from_config(cfg, truth)
    s = coupled.settings_from_config(cfg)
    start = datetime(2026, 10, 16)
    log = []
    coupled.run(truth, rooms, units, coupled.HeuristicController(rooms, units, s), start, 7,
                coupled.Weather(start, 9, mean=50, seed=13), s, log=log, seed=13)
    neighbors = {rid: sorted(set(r.coupling) | {j for j, o in truth.rooms.items() if rid in o.coupling})
                 for rid, r in truth.rooms.items()}
    layout = {"rooms": {rid: {"units": list(truth.units), "neighbors": neighbors[rid]} for rid in truth.rooms},
              "units": {u: {"zone": [r.id for r in rooms if r.unit == u]} for u in truth.units}}
    m, rep = fit.fit(log, layout, mass=False)
    got = m.rooms["bedroom_3"]
    assert 3.0 < got.leak_pulse < 14.0 and any(abs(got.leak_pulse_tau_h * 60 - t) < 0.1 for t in fit.PULSE_TAUS_MIN), (got.leak_pulse, got.leak_pulse_tau_h)
    assert rep["bedroom_3"]["pulse"]["holdout_rmse_F"][1] <= rep["bedroom_3"]["pulse"]["holdout_rmse_F"][0] * 1.02, rep["bedroom_3"]["pulse"]
    assert all(not r.leak_pulse for rid, r in m.rooms.items() if rid != "bedroom_3"), {rid: r.leak_pulse for rid, r in m.rooms.items()}
    plain, _ = fit.fit(log, layout, mass=False, pulse=False)
    # with the pulse the held-out predictions of Bedroom 3 must not be worse than without it
    assert rep["holdout"]["bedroom_3"]["model_rmse_F"] <= _["holdout"]["bedroom_3"]["model_rmse_F"] * 1.02
