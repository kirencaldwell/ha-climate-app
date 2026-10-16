"""The "would do" planner and live accuracy tracking, run against the simulated house. Run: python3 tests/test_mpc.py"""
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


def _shadow_run(model, hours, ignore=()):
    truth, rooms, units, s = _house()
    start = datetime.fromisoformat("2026-10-16T00:00:00-07:00")
    w = coupled.Weather(start, 2, mean=52, seed=5)
    sim = coupled.CoupledSim(truth, start, w, seed=5)
    ctl = coupled.HeuristicController(rooms, units, s)
    shadow = mpc.ShadowPlanner()
    shadow.set_model(model)
    for _ in range(int(hours * 60)):
        now = sim.now
        ctl.tick(sim, now)
        temps = {r.id: sim.reading(r.id) for r in rooms}
        actual = {u.id: {"mode": sim.mode[u.id], "setpoint": sim.sp[u.id], "running": sim.running[u.id]} for u in units}
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
