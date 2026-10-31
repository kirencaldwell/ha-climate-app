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
