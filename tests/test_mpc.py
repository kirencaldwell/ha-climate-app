"""The "would do" planner and live accuracy tracking, run against the simulated house. Run: python3 tests/test_mpc.py"""
import dataclasses
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import core  # noqa: E402
import coupled  # noqa: E402
import model as hm  # noqa: E402
import mpc  # noqa: E402


def _house():
    truth = coupled.the_house()
    cfg = {"rooms": [{"id": r, "unit": u, "priority": p, "default_temp": 67} for r, u, p in (
        ("kitchen", "air_handler", 2), ("living_room", "air_handler", 1), ("bedroom_1", "air_handler", 1),
        ("primary_bedroom", "minisplit_1", 5), ("bedroom_3", "minisplit_2", 4))],
        "units": [{"id": u} for u in truth.units], "settings": {"floor": 63.0}}
    rooms, units = coupled.rooms_units_from_config(cfg, truth)
    return truth, rooms, units, coupled.settings_from_config(cfg)


def _shadow_run(model, hours, ignore=(), setup=None):
    truth, rooms, units, s = _house()
    start = datetime.fromisoformat("2026-10-16T00:00:00-07:00")
    w = coupled.Weather(start, 2, mean=52, seed=5)
    sim = coupled.CoupledSim(truth, start, w, seed=5)
    ctl = coupled.HeuristicController(rooms, units, s)
    shadow = mpc.ShadowPlanner()
    shadow.set_model(model)
    if setup:
        setup(shadow)
    for _ in range(int(hours * 60)):
        now = sim.now
        ctl.tick(sim, now)
        temps = {r.id: sim.reading(r.id) for r in rooms}
        actual = {u.id: {"mode": sim.mode[u.id], "setpoint": sim.sp[u.id], "running": sim.running[u.id],
                         "fan": sim.fan[u.id]} for u in units}
        shadow.observe(now, temps, actual, w.outdoor(now), w.sun(now))
        if shadow.due(now):
            shadow.run(now, rooms, units, temps, actual, w.outdoor(now), w.sun, s, ignore)
        sim.advance(1)
    return shadow, sim


def test_tracking_scores_a_good_model_well_and_the_planner_decides():
    shadow, sim = _shadow_run(coupled.the_house(), 12)
    assert shadow.error is None, shadow.error
    acc = shadow.accuracy()
    assert set(acc) >= {"kitchen", "primary_bedroom"}
    for rid, by in acc.items():
        for h in ("30", "60", "120"):
            v = by[h]
            assert v["n"] >= 10, (rid, h, v)
            assert v["rmse"] < 0.3, (rid, h, v)  # true model: only sensor rounding and the inner loop's timing
        assert by["120"]["rmse"] < by["120"]["base_rmse"] or by["120"]["base_rmse"] < 0.3, (rid, by["120"])
    d = shadow.debug(sim.now)
    assert d["plan"] and d["plan"]["mode"] in ("heat", "cool", "off") and d["plan"]["ms"] is not None
    assert set(d["decision"]) == {"air_handler", "minisplit_1", "minisplit_2"}
    h = d["history"]["kitchen"]
    assert len(h["actual"]) > 60 and h["pred60"] and h["pred120"] and h["plan"]
    assert shadow.log_fields()["house"] == shadow.house.mode


def test_tracking_exposes_a_bad_model():
    bad = coupled.the_house()
    for r in bad.rooms.values():  # pretend the units do nothing and the house doesn't leak
        r.heat = {u: 0.0 for u in r.heat}
        r.cool = {u: 0.0 for u in r.cool}
        r.loss = 0.0
    good, _ = _shadow_run(coupled.the_house(), 12)
    poor, _ = _shadow_run(bad, 12)
    g, p = good.accuracy()["kitchen"]["120"]["rmse"], poor.accuracy()["kitchen"]["120"]["rmse"]
    assert p > 2 * g, (g, p)


def test_ignored_rooms_are_not_scored_and_errors_are_contained():
    shadow, sim = _shadow_run(coupled.the_house(), 2, ignore={"bedroom_3"})
    assert "bedroom_3" not in shadow.plan.predicted
    shadow.set_model(hm.HouseModel({"kitchen": hm.RoomModel(coupling={"kitchen": "oops"})}, {}))  # broken model
    temps = {"kitchen": 67.0, "nowhere": 60.0}
    shadow.observe(sim.now, temps, {}, 50.0, 0.0)
    from datetime import timedelta
    shadow.observe(sim.now + timedelta(minutes=1), temps, {}, 50.0, 0.0)
    assert shadow.error  # reported, not raised
    shadow.run(sim.now, [], [], temps, {}, 50.0, lambda t: 0.0, core.Settings())


def test_readiness_gate():
    good, sim = _shadow_run(coupled.the_house(), 12)
    rooms = {"kitchen", "primary_bedroom", "bedroom_3"}
    ok, why = good.readiness(sim.now, rooms, strict=False)
    assert not ok and any("fit" in w for w in why)  # the truth model wasn't fitted, so it isn't marked usable
    good.model.meta["usable"] = True
    ok, why = good.readiness(sim.now, rooms)
    assert ok, why
    assert not good.readiness(sim.now + timedelta(minutes=20), rooms)[0]  # plan gone stale
    fresh = mpc.ShadowPlanner()
    assert fresh.readiness(sim.now, rooms) == (False, ["no model yet"])
    bad = coupled.the_house()
    for r in bad.rooms.values():
        r.heat = {u: 0.0 for u in r.heat}
        r.cool = {u: 0.0 for u in r.cool}
        r.loss = 0.0
    poor, sim2 = _shadow_run(bad, 12)
    poor.model.meta["usable"] = True
    ok, why = poor.readiness(sim2.now, rooms)
    assert not ok and any("no better" in w for w in why), why


def test_readiness_tolerates_noise_but_not_a_real_miss():
    good, sim = _shadow_run(coupled.the_house(), 12)
    good.model.meta["usable"] = True
    rooms = {"kitchen"}
    score = lambda m, b: {"kitchen": {h: {"n": 100, "rmse": m, "base_rmse": b} for h in ("60", "120")}}
    good.accuracy = lambda: score(0.63, 0.49)  # 0.14 F behind a room that hardly moves (the kitchen today): noise
    assert good.readiness(sim.now, rooms)[0]
    good.accuracy = lambda: score(0.70, 0.49)  # 0.21 F behind: past the tolerance, a real miss
    assert not good.readiness(sim.now, rooms)[0]
    good.accuracy = lambda: score(2.7, 2.5)  # a room that moves, 0.2 F behind: still a miss to hand over on...
    assert not good.readiness(sim.now, rooms)[0]
    assert good.readiness(sim.now, rooms, strict=False)[0]  # ...but not enough to hand it back (10% slack on top)


def test_thermostat_units_are_planned_around():
    truth, rooms, units, s = _house()
    start = datetime.fromisoformat("2026-10-16T03:00:00-07:00")
    w = coupled.Weather(start, 2, mean=50, seed=5)
    sh = mpc.ShadowPlanner()
    sh.set_model(truth)
    temps = {r.id: 64.0 for r in rooms}
    actual = {u.id: {"mode": "heat", "setpoint": 66.0, "running": False} for u in units}
    sh.run(start, rooms, units, temps, actual, w.outdoor(start), w.sun, s, fixed={"minisplit_2": ("cool", 66.0)})
    assert sh.error is None, sh.error
    assert sh.decision["minisplit_2"]["mode"] == "thermostat" and sh.house.mode == "cool"


def test_scores_survive_a_restart():
    import json
    sh, sim = _shadow_run(coupled.the_house(), 6)
    saved = json.loads(json.dumps(sh.to_state()))  # through JSON, as it's stored
    back = mpc.ShadowPlanner()
    back.load_state(saved, sim.now + timedelta(minutes=5))
    assert back.accuracy() == sh.accuracy() and back.accuracy()
    assert len(back.points) == len(sh.points) and set(back.readings) == set(sh.readings)
    assert not back.records  # in-progress predictions aren't restored
    later = mpc.ShadowPlanner()
    later.load_state(saved, sim.now + timedelta(hours=25))  # a day later: all of it has aged out
    assert not later.errors and not later.points and all(not d for d in later.readings.values())
    later.load_state({"version": 99}, sim.now)  # an unknown format is ignored, not an error


def test_chart_history_comfort_and_accuracy_series():
    truth, rooms, units, s = _house()
    start = datetime.fromisoformat("2026-10-16T00:00:00-07:00")
    log = []
    coupled.run(truth, rooms, units, coupled.HeuristicController(rooms, units, s), start, 0.5,
                coupled.Weather(start, 2, seed=9), s, log=log, seed=9)
    for k, row in enumerate(log):  # as the integration logs it: what the model would do
        row["house"] = "heat"
        row["mpc"] = {"house": "heat", "comfort": 0.5, "units": {"air_handler": {"mode": "heat", "setpoint": 66.0}}}
    samples = [mpc.compact(r) for r in log[::5]]
    assert samples[0]["u"]["air_handler"][4:] == ["heat", 66.0] and samples[0]["mc"] == 0.5
    cs = mpc.comfort_series(samples, ["kitchen", "primary_bedroom"], s)
    assert len(cs) == len(samples) and cs[0][1] is not None and cs[-1][1] is None  # the last 2 h aren't over yet
    assert all(c[2] == 0.5 for c in cs)
    sh, sim = _shadow_run(coupled.the_house(), 6)
    by = mpc.accuracy_by_hour(sh.errors)
    assert set(by["kitchen"]) == {"30", "60", "120"} and all(len(v) >= 3 for v in by["kitchen"].values())
    assert sh.projection and len(sh.projection["temps"]["kitchen"]) == 12  # 2 h ahead if settings stay


def test_forecast_follows_the_hourly_points_from_the_current_reading():
    now = datetime.fromisoformat("2026-10-16T09:10:00-07:00")
    h = lambda k: now.replace(minute=0) + timedelta(hours=k)
    f = mpc.along(now, 55.0, [(h(0), 54.0), (h(1), 58.0), (h(2), 61.0), (h(3), None)])  # 9:00 already started
    assert f(now) == 55.0  # what's observed, not the forecast for this hour
    assert abs(f(h(1)) - 58.0) < 1e-9 and abs(f(h(1) + timedelta(minutes=30)) - 59.5) < 1e-9
    assert f(h(9)) == 61.0  # past the end: hold the last value
    assert abs(f(now + timedelta(minutes=25)) - (55 + 3 * 25 / 50)) < 1e-9  # straight from 55 now to 58 at 10:00
    assert mpc.along(now, 55.0, [])(h(2)) == 55.0  # no forecast: current conditions held
    assert mpc.along(now, None, [(h(1), 58.0)])(h(1)) == 58.0  # no observation: start from the forecast


def test_planner_and_projection_use_a_forecast():
    truth, rooms, units, s = _house()
    now = datetime.fromisoformat("2026-10-16T09:00:00-07:00")
    w = coupled.Weather(now, 2, mean=52, seed=5)
    temps = {r.id: 66.0 for r in rooms}
    actual = {u.id: {"mode": "heat", "setpoint": 66.0, "running": False} for u in units}
    cold, warm = mpc.ShadowPlanner(), mpc.ShadowPlanner()
    cold.set_model(truth)
    warm.set_model(truth)
    cold.run(now, rooms, units, temps, actual, 50.0, w.sun, s)
    warm.run(now, rooms, units, temps, actual, mpc.along(now, 50.0, [(now + timedelta(hours=2), 80.0)]), w.sun, s)
    assert cold.error is None and warm.error is None, (cold.error, warm.error)
    c, h = cold.projection["temps"]["kitchen"][-1][1], warm.projection["temps"]["kitchen"][-1][1]
    assert h > c  # a warming forecast makes the house expected warmer in 2 h


def test_joint_planner_runs_in_shadow_and_never_touches_the_decision():
    truth, rooms, units, s = _house()
    start = datetime.fromisoformat("2026-10-16T03:00:00-07:00")
    w = coupled.Weather(start, 2, mean=50, seed=5)
    temps = {r.id: 64.0 for r in rooms}
    actual = {u.id: {"mode": "heat", "setpoint": 66.0, "running": False} for u in units}
    off = mpc.ShadowPlanner()
    off.set_model(truth)
    off.run(start, rooms, units, temps, actual, w.outdoor(start), w.sun, dataclasses.replace(s, joint_shadow=False))
    on = mpc.ShadowPlanner()
    on.set_model(truth)
    on.run(start, rooms, units, temps, actual, w.outdoor(start), w.sun, dataclasses.replace(s, joint_shadow=True))
    assert off.jplan is None and on.jerror is None and on.jplan is not None
    assert on.decision == off.decision and on.house.mode == off.house.mode  # the joint plan changes nothing
    f = on.log_fields()["joint"]
    assert f["house"] in ("off", "heat", "cool") and set(f["run"]) <= {u.id for u in units}
    assert on.debug(start)["joint"]["flips_24h"] == 0
    print("joint plan ms", on.jplan_ms)


def test_joint_planner_can_switch_a_leaky_idle_unit_fully_off():
    import joint
    truth, rooms, units, s = _house()
    start = datetime.fromisoformat("2026-10-16T03:00:00-07:00")
    w = coupled.Weather(start, 2, mean=50, seed=5)
    temps = {r.id: 64.0 for r in rooms}
    temps["bedroom_3"] = 67.5  # warm, on a unit that leaks heat while idle in heat; the other rooms need heat
    house = core.HouseState("heat", start - timedelta(hours=3))
    running = {u.id: False for u in units}
    on = joint.JointPlanner(joint.JointSettings(sleep_option=True)).plan(truth, rooms, units, temps, start, house, running, w.outdoor, w.sun, s)
    off = joint.JointPlanner(joint.JointSettings(sleep_option=False)).plan(truth, rooms, units, temps, start, house, running, w.outdoor, w.sun, s)
    assert "sleep" in on.pattern["minisplit_2"] and not any("sleep" in p for p in off.pattern.values())
    assert on.cost < off.cost  # switching it off avoids the leak: a cheaper plan
    assert set(on.sleep) <= {u.id for u in units}


def test_joint_planner_as_the_live_decision():
    truth, rooms, units, s = _house()
    start = datetime.fromisoformat("2026-10-16T03:00:00-07:00")
    w = coupled.Weather(start, 2, mean=50, seed=5)
    temps = {r.id: 64.0 for r in rooms}
    temps["bedroom_3"] = 67.5
    actual = {u.id: {"mode": "heat", "setpoint": 66.0, "running": False} for u in units}
    live = dataclasses.replace(s, model_planner="joint")
    sh = mpc.ShadowPlanner()
    sh.set_model(truth)
    sh.house = core.HouseState("heat", start - timedelta(hours=3))
    sh.jhouse = core.HouseState("heat", start - timedelta(hours=3))
    sh.run(start, rooms, units, temps, actual, w.outdoor(start), w.sun, live)
    assert sh.error is None and sh.jerror is None, (sh.error, sh.jerror)
    assert set(sh.decision) == {u.id for u in units} and sh.house is sh.jhouse and sh.live_reason.startswith("joint")
    modes = {d["mode"] for d in sh.decision.values()}
    assert modes <= {"heat", "off"} and any(d.get("run") for d in sh.decision.values())
    assert all(d.get("setpoint") is not None for d in sh.decision.values() if d["mode"] == "heat")
    sh2 = mpc.ShadowPlanner()  # the default planner leaves the decision to the optimizer
    sh2.set_model(truth)
    sh2.run(start, rooms, units, temps, actual, w.outdoor(start), w.sun, s)
    assert sh2.live_reason is None


def test_readiness_can_skip_only_the_fit_check():
    truth, rooms, units, s = _house()
    sh = mpc.ShadowPlanner()
    sh.set_model(truth)
    truth.meta = {"usable": False}
    ok, why = sh.readiness(datetime.fromisoformat("2026-10-16T03:00:00-07:00"), [])
    assert not ok and any("doesn't beat" in x for x in why)
    ok2, why2 = sh.readiness(datetime.fromisoformat("2026-10-16T03:00:00-07:00"), [], skip_fit=True)
    assert not any("doesn't beat" in x for x in why2) and any("no fresh plan" in x for x in why2)  # every other check still counts


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
