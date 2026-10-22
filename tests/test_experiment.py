"""Experiment days: the plan and the commands it gives. Run: python3 tests/test_experiment.py"""
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
import core  # noqa: E402
import experiment as ex  # noqa: E402

T0 = datetime.fromisoformat("2026-10-18T09:00:00-07:00")
UNITS = ["air_handler", "minisplit_1", "minisplit_2"]


def test_plan_fills_the_time_and_ends_on_normal_control():
    p = ex.plan(T0, 8, UNITS, fan_units=("minisplit_1", "minisplit_2"), mode="heat")
    b = p["blocks"]
    assert b[0]["kind"] == "float" and b[-1]["kind"] == "restore"
    assert b[-1]["end"] == (T0 + timedelta(hours=8)).isoformat()
    assert datetime.fromisoformat(b[-1]["start"]) == T0 + timedelta(hours=7)  # the last hour is normal control
    for x, y in zip(b, b[1:]):
        assert x["end"] == y["start"]  # no gaps
    solos = [x for x in b if x["kind"] == "solo"]
    assert {x["unit"] for x in solos} == set(UNITS)
    assert all(x["fan"] is None for x in solos if x["unit"] == "air_handler")  # the air handler keeps its own fan
    assert [x["fan"] for x in solos if x["unit"] == "minisplit_2"][:4] == list(ex.FAN_CYCLE)  # a sweep in one run
    assert ex.plan(T0, 100, UNITS)["end"] == (T0 + timedelta(hours=ex.MAX_H)).isoformat()  # clamped


def test_current_block_and_mode():
    p = ex.plan(T0, 4, UNITS)
    assert ex.current(p, T0)["kind"] == "float" and ex.current(p, T0 - timedelta(minutes=1)) is None
    assert ex.current(p, T0 + timedelta(hours=3, minutes=30))["kind"] == "restore"
    assert ex.current(p, T0 + timedelta(hours=4)) is None
    assert ex.pick_mode("cool", 40) == "cool" and ex.pick_mode("off", 50) == "heat" and ex.pick_mode("off", 75) == "cool"


def test_leaky_units_are_the_ones_with_a_real_leak():
    class R:  # the fit lists every unit for its own room, with a zero where there is no leak
        def __init__(self, leak):
            self.leak = leak
    rooms = [R({"air_handler": 0.0}), R({"minisplit_1": 0.0}), R({"minisplit_2": 6.9})]
    assert ex.leaky_units(rooms) == {"minisplit_2"}
    assert ex.order(UNITS, ex.leaky_units(rooms)) == ["minisplit_2", "air_handler", "minisplit_1"]
    assert ex.leaky_units([]) == set()


def test_leaky_units_go_first():
    assert ex.order(UNITS, leaky={"minisplit_2"}) == ["minisplit_2", "air_handler", "minisplit_1"]
    assert ex.order(UNITS, leaky=()) == UNITS  # no leak fitted: as given
    first = ex.plan(T0, 8, ex.order(UNITS, {"minisplit_2"}), ("minisplit_1", "minisplit_2"))["blocks"]
    assert [x["unit"] for x in first if x["kind"] == "solo"][0] == "minisplit_2"


def _cmds(room=66.0):
    return {u: core.Command("heat", 66.0, "", True, 66.0, 67.0, room, 0.0) for u in UNITS}


def test_solo_unit_pushes_and_the_others_idle():
    s = core.Settings()
    b = {"kind": "solo", "unit": "minisplit_2", "fan": "low"}
    rooms = {"air_handler": ["kitchen"], "minisplit_1": ["primary_bedroom"], "minisplit_2": ["bedroom_3"]}
    out = ex.commands(b, "heat", _cmds(), rooms, {"kitchen": 66.0, "primary_bedroom": 66.0, "bedroom_3": 66.0}, s)
    assert out["minisplit_2"].setpoint >= 67.5 and out["minisplit_2"].fan == "low" and out["minisplit_2"].active
    assert all(out[u].setpoint <= 64.5 and out[u].fan is None for u in ("air_handler", "minisplit_1"))
    # the house limits still hold; the ceiling has CEILING_EXTRA more room on an experiment day
    warm = ex.commands(b, "heat", _cmds(73.5), rooms, {"kitchen": 66.0, "primary_bedroom": 66.0, "bedroom_3": 73.5}, s)
    assert warm["minisplit_2"].active  # past the everyday ceiling (72), under the experiment's
    hot = ex.commands(b, "heat", _cmds(75.0), rooms, {"kitchen": 66.0, "primary_bedroom": 66.0, "bedroom_3": 75.0}, s)
    assert not hot["minisplit_2"].active
    cold = ex.commands({"kind": "float"}, "heat", _cmds(62.5), rooms, {"kitchen": 62.5, "primary_bedroom": 66.0, "bedroom_3": 66.0}, s)
    assert cold["air_handler"].active and not cold["minisplit_1"].active


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
