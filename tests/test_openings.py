"""Suggested register and door positions. Run: python3 tests/test_openings.py"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))
import openings  # noqa: E402

ROOMS = [{"id": "b2", "name": "Bedroom 2", "unit": "ah"}, {"id": "k", "name": "Kitchen", "unit": "ah"}, {"id": "bath", "name": "Bath", "unit": None}]
LIVE = {"b2": {"temp": 72.7, "target": 67, "priority": 0.3}, "k": {"temp": 63.3, "target": 67.4, "priority": 10}}
OPS = [{"id": "r1", "kind": "register", "room": "b2", "state": "open"}, {"id": "r2", "kind": "register", "room": "k", "state": "open"},
       {"id": "d1", "kind": "door", "room": "b2", "state": "open"}, {"id": "d2", "kind": "door", "room": "k", "state": "cracked"},
       {"id": "r3", "kind": "register", "room": "bath", "state": "open"}, {"id": "r4", "kind": "register", "room": None, "state": "open"}]


def test_heating_closes_the_overheated_room_and_keeps_the_cold_one_open():
    r = openings.recommend(OPS, ROOMS, LIVE, "heat")
    assert r["r1"]["rec"] == "closed" and not r["r1"]["match"] and "5.7° over" in r["r1"]["why"]
    assert r["r2"]["rec"] == "open" and r["r2"]["match"]
    assert r["d1"]["rec"] == "cracked" and r["d2"]["rec"] == "open" and not r["d2"]["match"]


def test_a_door_is_never_suggested_further_open_for_a_room_that_is_too_warm():
    ops = [{"id": "d", "kind": "door", "room": "b2", "state": "closed"}, {"id": "e", "kind": "door", "room": "k", "state": "wide"}]
    r = openings.recommend(ops, ROOMS, LIVE, "heat")
    assert r["d"]["rec"] == "closed" and r["d"]["match"]  # closed already: nothing to do
    assert r["e"]["rec"] == "wide" and r["e"]["match"]  # a cold room's door that is wide open already is fine


def test_nothing_for_rooms_without_a_unit_or_reading_and_nothing_when_the_house_is_off():
    r = openings.recommend(OPS, ROOMS, LIVE, "heat")
    assert "r3" not in r and "r4" not in r
    assert openings.recommend(OPS, ROOMS, LIVE, "off") == {} and openings.recommend(OPS, ROOMS, LIVE, None) == {}


def test_the_threshold_scales_with_the_rooms_weight_and_flips_for_cooling():
    near = {"k": {"temp": 68.3, "target": 67, "priority": 1}}  # 1.3 over, allowed 1.0 -> half closed
    ops = [{"id": "r", "kind": "register", "room": "k", "state": "open"}]
    assert openings.recommend(ops, ROOMS, near, "heat")["r"]["rec"] == "partial"
    heavy = {"k": {"temp": 68.3, "target": 67, "priority": 25}}  # allowed 0.2: 1.3 over is 6.5x
    assert openings.recommend(ops, ROOMS, heavy, "heat")["r"]["rec"] == "closed"
    cool = {"k": {"temp": 65.0, "target": 67, "priority": 1}}  # cooling and 2 under its target: past its goal
    assert openings.recommend(ops, ROOMS, cool, "cool")["r"]["rec"] == "closed"
    assert openings.recommend(ops, ROOMS, cool, "heat")["r"]["rec"] == "open"


def test_register_multipliers_redistribute_a_units_output_and_equal_one_when_all_open():
    class R:  # a fitted room: its heat gains per unit
        def __init__(self, g): self.heat = {"ah": g}
    mr = {"k": R(1.0), "b2": R(4.0), "lr": R(5.0)}
    ru = {"k": "ah", "b2": "ah", "lr": "ah"}
    regs = [{"id": "rk", "kind": "register", "room": "k", "state": "open"}, {"id": "rb", "kind": "register", "room": "b2", "state": "open"}]
    assert openings.register_multipliers(regs, ru, mr) == {"k": 1.0, "b2": 1.0, "lr": 1.0}
    closed = openings.register_multipliers(regs, ru, mr, {"rb": "closed"})
    assert closed["b2"] < 0.3 and closed["k"] > 1.3  # the kitchen gets the air the bedroom no longer takes
    assert closed["lr"] > 1.3  # the living room has no register entry but shares the freed air as well
    total = sum(mr[r].heat["ah"] * closed[r] for r in mr)
    assert abs(total - sum(g.heat["ah"] for g in mr.values())) < 1e-9  # the unit's total output is conserved
    assert set(openings.register_multipliers([{**regs[0], "state": "unset"}], ru, mr).values()) == {1.0}


def test_door_level_is_the_mean_of_its_doors_with_a_default():
    d = [{"id": "a", "kind": "door", "room": "r", "state": "closed"}, {"id": "b", "kind": "door", "room": "r", "state": "wide"},
         {"id": "c", "kind": "door", "room": "q", "state": "unset"}]
    assert openings.door_level(d, "r") == 0.5 and openings.door_level(d, "q") == openings.DOOR_DEFAULT == openings.door_level(d, "none")


def test_the_advisor_closes_the_overheated_rooms_register_to_feed_the_cold_one():
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    import core
    import mpc
    import test_mpc
    from datetime import datetime, timedelta
    truth, rooms, units, s = test_mpc._house()
    start = datetime.fromisoformat("2026-10-16T03:00:00-07:00")
    w = test_mpc.coupled.Weather(start, 2, mean=45, seed=5)
    temps = {r.id: 66.0 for r in rooms}
    temps["bedroom_1"], temps["kitchen"] = 72.0, 62.0
    actual = {u.id: {"mode": "heat", "setpoint": 66.0, "running": False} for u in units}
    sh = mpc.ShadowPlanner()
    sh.set_model(truth)
    sh.jhouse = core.HouseState("heat", start - timedelta(hours=3))
    items = [{"id": "r-b2", "kind": "register", "name": "Bedroom 1 register", "room": "bedroom_1", "state": "open"},
             {"id": "r-k", "kind": "register", "name": "Kitchen register", "room": "kitchen", "state": "open"}]
    sh.set_openings(items, {r.id: r.unit for r in rooms})
    sh.run(start, rooms, units, temps, actual, w.outdoor(start), w.sun, s)
    adv = sh.opening_advice
    assert adv is not None and adv["r-b2"]["rec"] in ("closed", "partial") and not adv["r-b2"]["match"], adv
    assert adv["r-k"]["rec"] == "open" and adv["r-k"]["match"] and "less" in adv["r-b2"]["why"]


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
