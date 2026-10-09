import os, sys
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import core
from core import Room, ScheduleEntry, Unit, UnitState, Settings
from house import SimHouse

MON = datetime(2026, 10, 5, 12, 0)  # a Monday


def sched_room(**kw):
    return Room("r", "R", "u", schedule=[
        ScheduleEntry([0, 1, 2, 3, 4], "06:30", 69),
        ScheduleEntry(list(range(7)), "22:00", 66)], default_temp=68, **kw)


def test_schedule_picks_latest_entry():
    r = sched_room()
    assert core.scheduled_temp(r, MON.replace(hour=7)) == 69
    assert core.scheduled_temp(r, MON.replace(hour=23)) == 66


def test_schedule_wraps_past_midnight_and_weekend():
    r = sched_room()
    assert core.scheduled_temp(r, MON.replace(hour=2)) == 66  # Sunday 22:00 entry
    sat = datetime(2026, 10, 10, 9, 0)
    assert core.scheduled_temp(r, sat) == 66  # weekday entry doesn't apply


def test_no_schedule_uses_default():
    assert core.scheduled_temp(Room("r", "R", "u", default_temp=67), MON) == 67


def test_hold_overrides_until_expiry():
    r = sched_room(hold_temp=72, hold_until=MON + timedelta(hours=1))
    assert core.target_temp(r, MON) == 72
    assert core.target_temp(r, MON + timedelta(hours=2)) == core.scheduled_temp(r, MON + timedelta(hours=2))


def test_priority_weights_error():
    a = Room("a", "A", "u", priority=5, default_temp=68)
    b = Room("b", "B", "u", priority=1, default_temp=68)
    rd = {"a": (64.0, MON), "b": (72.0, MON)}
    err, _ = core.weighted_error([a, b], rd, MON, Settings())
    assert err < 0  # high-priority cold room dominates


def test_stale_readings_ignored_and_unit_held():
    r = Room("r", "R", "u")
    old = {"r": (60.0, MON - timedelta(hours=2))}
    cmd = core.decide(Unit("u", "climate.x"), [r], old, 70, UnitState("off"), MON)
    assert cmd.setpoint is None and cmd.mode == "off"


def test_heats_when_cold_and_compensates_unit_sensor():
    r = Room("r", "R", "u", default_temp=68)
    cmd = core.decide(Unit("u", "c"), [r], {"r": (66.0, MON)}, 69.0, UnitState("off"), MON)
    assert cmd.mode == "heat"
    assert cmd.setpoint == 68 + 3 + 1  # target + (unit - room) + push


def test_deadband_stays_off():
    r = Room("r", "R", "u", default_temp=68)
    assert core.decide(Unit("u", "c"), [r], {"r": (67.5, MON)}, None, UnitState("off"), MON).mode == "off"


def test_min_run_time_respected():
    r = Room("r", "R", "u", default_temp=68)
    st = UnitState("heat", MON - timedelta(seconds=60), "heat")
    assert core.decide(Unit("u", "c"), [r], {"r": (69.0, MON)}, None, st, MON).mode == "heat"
    st = UnitState("heat", MON - timedelta(seconds=600), "heat")
    assert core.decide(Unit("u", "c"), [r], {"r": (69.0, MON)}, None, st, MON).mode == "off"


def test_mode_switch_lockout():
    r = Room("r", "R", "u", default_temp=68)
    st = UnitState("off", MON - timedelta(seconds=400), "heat")
    assert core.decide(Unit("u", "c"), [r], {"r": (70.0, MON)}, None, st, MON).mode == "off"
    st = UnitState("off", MON - timedelta(seconds=1000), "heat")
    assert core.decide(Unit("u", "c"), [r], {"r": (70.0, MON)}, None, st, MON).mode == "cool"


def test_setpoint_clamped():
    r = Room("r", "R", "u", default_temp=68)
    u = Unit("u", "c", min_setpoint=61, max_setpoint=70)
    assert core.decide(u, [r], {"r": (60.0, MON)}, 75.0, UnitState("off"), MON).setpoint == 70


def _run(rooms, units, outdoor, hours=48):
    h = SimHouse(rooms, units, outdoor, MON.replace(hour=0))
    h.step(hours * 60)
    return h


def test_sim_holds_68_in_cold_weather_with_biased_unit_sensor():
    rooms = [Room("living", "Living", "ah", default_temp=68), Room("kitchen", "Kitchen", "ah", default_temp=68, priority=2),
             Room("bed1", "Bed 1", "ms1", default_temp=68), Room("bed2", "Bed 2", "ms2", default_temp=68)]
    units = [Unit("ah", "c1"), Unit("ms1", "c2"), Unit("ms2", "c3")]
    h = SimHouse(rooms, units, lambda t: 35.0, MON.replace(hour=0))
    h.sim["bed1"].temp = 62
    temps = []
    for _ in range(48 * 60):
        h.step(1)
        temps.append({k: v.temp for k, v in h.sim.items()})
    settled = temps[6 * 60:]
    for rid in h.sim:
        vals = [t[rid] for t in settled]
        assert 66.5 <= min(vals) and max(vals) <= 69.5, (rid, min(vals), max(vals))


def test_sim_follows_overnight_setback():
    r = Room("bed", "Bed", "ms", schedule=[ScheduleEntry(list(range(7)), "07:00", 69), ScheduleEntry(list(range(7)), "22:00", 65)])
    h = SimHouse([r], [Unit("ms", "c")], lambda t: 40.0, MON.replace(hour=0))
    h.step(8 * 60)
    assert abs(h.sim["bed"].temp - 69) < 1.5
    h.step(16 * 60)  # to 24:00
    assert abs(h.sim["bed"].temp - 65) < 1.5
