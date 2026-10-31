"""Registers and doors: levels, confidence, redistribution and the advisor. Run: python3 tests/test_openings.py"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))
import openings  # noqa: E402

ROOMS = [{"id": "b2", "name": "Bedroom 2", "unit": "ah"}, {"id": "k", "name": "Kitchen", "unit": "ah"}, {"id": "bath", "name": "Bath", "unit": None}]
LIVE = {"b2": {"temp": 72.7, "target": 67, "priority": 0.3}, "k": {"temp": 63.3, "target": 67.4, "priority": 10}}
OPS = [{"id": "r1", "kind": "register", "room": "b2", "state": "open"}, {"id": "r2", "kind": "register", "room": "k", "state": "open"},
       {"id": "d1", "kind": "door", "room": "b2", "state": "open"}, {"id": "d2", "kind": "door", "room": "k", "state": "cracked"},
       {"id": "r3", "kind": "register", "room": "bath", "state": "open"}, {"id": "r4", "kind": "register", "room": None, "state": "open"}]


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


def test_the_advisor_still_answers_when_nothing_is_planned_to_run():
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    import core
    import mpc
    import test_mpc
    from datetime import datetime
    truth, rooms, units, s = test_mpc._house()
    start = datetime.fromisoformat("2026-10-16T12:00:00-07:00")
    w = test_mpc.coupled.Weather(start, 2, mean=60, seed=5)
    temps = {r.id: core.target_temp(r, start) for r in rooms}  # everything on target: the plan is to run nothing
    temps["bedroom_1"] += 5.0
    temps["kitchen"] -= 3.0
    actual = {u.id: {"mode": "heat", "setpoint": 66.0, "running": False} for u in units}
    sh = mpc.ShadowPlanner()
    sh.set_model(truth)
    items = [{"id": "r-b1", "kind": "register", "name": "Bedroom 1 register", "room": "bedroom_1", "state": "open"},
             {"id": "r-k", "kind": "register", "name": "Kitchen register", "room": "kitchen", "state": "open"}]
    sh.set_openings(items, {r.id: r.unit for r in rooms})
    sh.run(start, rooms, units, temps, actual, w.outdoor(start), w.sun, s)
    assert sh.jerror is None, sh.jerror
    assert sh.opening_advice is not None and set(sh.opening_advice) == {"r-b1", "r-k"}


def test_confidence_starts_fresh_decays_to_the_owners_fifty_percent_and_is_zero_when_unset():
    from datetime import datetime, timedelta
    now = datetime.fromisoformat("2026-10-31T12:00:00-07:00")
    mk = lambda ago_h, st="partial", key="changed": {"kind": "register", "state": st, key: (now - timedelta(hours=ago_h)).isoformat()}
    assert abs(openings.confidence(mk(0), now) - 0.95) < 1e-6
    assert 0.6 < openings.confidence(mk(24), now) < 0.7  # a day later: most of the way down
    assert abs(openings.confidence(mk(24 * 14), now) - 0.5) < 0.01  # the long-run floor: the owner's own estimate
    assert openings.confidence({"kind": "register", "state": "unset"}, now) == 0.0
    assert abs(openings.confidence(mk(0, key="confirmed"), now) - 0.95) < 1e-6  # confirming resets it
    assert openings.confidence({"kind": "door", "state": "open"}, None) == 0.5  # no time known: the floor


def test_uncertain_registers_pull_the_multipliers_toward_neutral_and_a_trial_state_counts_as_just_set():
    from datetime import datetime, timedelta
    now = datetime.fromisoformat("2026-10-31T12:00:00-07:00")
    class R:
        def __init__(self, g): self.heat = {"ah": g}
    mr = {"k": R(1.0), "b2": R(4.0), "lr": R(5.0)}
    ru = {"k": "ah", "b2": "ah", "lr": "ah"}
    old = (now - timedelta(days=30)).isoformat()
    regs = [{"id": "rk", "kind": "register", "room": "k", "state": "open", "changed": old},
            {"id": "rb", "kind": "register", "room": "b2", "state": "closed", "changed": old}]
    face = openings.register_multipliers(regs, ru, mr)
    sure = openings.register_multipliers(regs, ru, mr, now=now, conf=(0.5, 0.95, 24.0))
    assert face["b2"] < sure["b2"] < 1.0  # a 50% sure "closed" is far less closed than a certain one
    assert face["k"] > sure["k"]  # and a 50% sure "open" is not as open as a certain one, so the redistribution is milder
    trial = openings.register_multipliers(regs, ru, mr, {"rb": "open"}, now=now, conf=(0.5, 0.95, 24.0))
    assert trial["b2"] > sure["b2"]  # trying "open" (as if just set): the room gets more air than under the doubtful "closed"
    just = [{**regs[1], "changed": now.isoformat()}, regs[0]]
    assert openings.register_multipliers(just, ru, mr, now=now, conf=(0.5, 0.95, 24.0))["b2"] < sure["b2"]  # freshly set: trusted more


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
