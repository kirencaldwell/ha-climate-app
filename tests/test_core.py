import dataclasses, os, sys
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import core
from core import Room, ScheduleEntry, Unit, UnitState, Settings
from house import SimHouse

MON = datetime(2026, 10, 5, 12, 0)  # a Monday
# Symmetric comfort, no cool aim: for tests about other mechanics. The cool-leaning defaults have their own tests.
NEUTRAL = Settings(band_below=1.0, band_above=1.5, warm_weight=1.0, aim_offset=0.0)


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
        assert 66.0 <= min(vals) and max(vals) <= 69.0, (rid, min(vals), max(vals))  # band 66.5-69, aimed at 67.5


def test_sim_follows_overnight_setback():
    r = Room("bed", "Bed", "ms", schedule=[ScheduleEntry(list(range(7)), "07:00", 69), ScheduleEntry(list(range(7)), "22:00", 65)])
    h = SimHouse([r], [Unit("ms", "c")], lambda t: 40.0, MON.replace(hour=0))
    h.step(8 * 60)
    assert abs(h.sim["bed"].temp - 69) < 1.5
    h.step(16 * 60)  # to 24:00
    assert abs(h.sim["bed"].temp - 65) < 1.5


def _diag(rooms, readings, **kw):
    u = Unit("u", "climate.u")
    return core.diagnose(u, rooms, readings, kw.get("unit_temp"), UnitState(), core.Command("heat", 68.0, "heating"), MON)


def test_diagnose_weights_and_driver():
    a = Room("a", "A", "u", priority=4, default_temp=68)
    b = Room("b", "B", "u", priority=1, default_temp=68)
    d = _diag([a, b], {"a": (64.0, MON), "b": (70.0, MON)}, unit_temp=69.0)
    assert [r["id"] for r in d["rooms"]] == ["a", "b"]  # highest priority first
    assert d["rooms"][0]["weight_pct"] == 80.0 and d["rooms"][1]["weight_pct"] == 20.0
    assert d["rooms"][0]["delta"] == -4.0
    assert d["driver"] == "a"
    assert d["error"] == round((4 * -4 + 1 * 2) / 5, 2)
    assert d["offset"] == round(69.0 - d["avg"], 1)
    assert d["mode"] == "off" and d["setpoint"] == 68.0 and d["reason"] == "heating"


def test_diagnose_ignores_stale_and_missing_sensors():
    a = Room("a", "A", "u", priority=3)
    b = Room("b", "B", "u", priority=3)
    d = _diag([a, b], {"a": (66.0, MON - timedelta(hours=3))})  # a stale, b missing
    assert d["error"] is None and d["driver"] is None
    assert all(not r["fresh"] and r["weight_pct"] == 0 for r in d["rooms"])


def test_diagnose_skips_other_units_rooms():
    a = Room("a", "A", "u"); other = Room("x", "X", "other")
    d = _diag([a, other], {"a": (68.0, MON), "x": (60.0, MON)})
    assert [r["id"] for r in d["rooms"]] == ["a"]



def test_sim_never_mixes_modes():
    # One room cold, one hot: the two units must never run heat and cool at the same time.
    rooms = [Room("cold", "Cold", "ah", default_temp=70), Room("hot", "Hot", "ms1", default_temp=66)]
    h = SimHouse(rooms, [Unit("ah", "c1"), Unit("ms1", "c2")], lambda t: 50.0, MON.replace(hour=0))
    h.sim["cold"].temp, h.sim["hot"].temp = 64, 72
    for _ in range(12 * 60):
        h.step(1)
        modes = {s.mode for s in h.state.values()} - {"off"}
        assert len(modes) <= 1, modes


# --- house-wide controller: one shared mode, comfort band, hard limits, predicted temps ---

def _house(bed=67.0, down=67.0, bed_pred=None, down_pred=None, house=None, at=MON, states=None, busy=None):
    rooms = [Room("bed", "Bedroom", "ms", priority=5, default_temp=67), Room("down", "Living", "ah", priority=2, default_temp=67)]
    units = [Unit("ah", "c1"), Unit("ms", "c2")]
    readings = {"bed": (bed, at), "down": (down, at)}
    pred = {"bed": bed_pred, "down": down_pred}
    return core.control_step(units, rooms, readings, pred, {}, states or {}, house or core.HouseState(), at, NEUTRAL, busy=busy)


def _started(rooms, units, readings, pred=None, manual=None, busy=None):
    """control_step twice, past the confirmation time: leaving "off" takes a held demand, not one reading."""
    house = core.HouseState()
    for minutes in (0, 11):
        now = MON + timedelta(minutes=minutes)
        rd = {k: (v[0], now) for k, v in readings.items()}
        cmds, house, info = core.control_step(units, rooms, rd, pred or {}, {}, {}, house, now, NEUTRAL, manual=manual, busy=busy)
    return cmds, house, info


def _house_started(bed=67.0, down=67.0, bed_pred=None, down_pred=None, busy=None):
    rooms = [Room("bed", "Bedroom", "ms", priority=5, default_temp=67), Room("down", "Living", "ah", priority=2, default_temp=67)]
    units = [Unit("ah", "c1"), Unit("ms", "c2")]
    return _started(rooms, units, {"bed": (bed, MON), "down": (down, MON)}, {"bed": bed_pred, "down": down_pred}, busy=busy)


def test_costs_zero_inside_band_and_floor_beats_priority():
    s = NEUTRAL
    hi_pri = Room("a", "A", "u", priority=5, default_temp=67)
    lo_pri = Room("b", "B", "u", priority=1, default_temp=67)
    assert core.room_costs(hi_pri, 66.2, None, MON, s) == (0.0, 0.0)  # inside 66-68.5
    assert core.room_costs(hi_pri, 68.4, None, MON, s) == (0.0, 0.0)
    slightly_cold = core.room_costs(hi_pri, 65.5, None, MON, s)[0]
    freezing = core.room_costs(lo_pri, 60.0, None, MON, s)[0]
    assert freezing > 10 * slightly_cold  # below the 65 floor wins regardless of priority
    assert core.room_costs(lo_pri, 73.0, None, MON, s)[1] > core.room_costs(hi_pri, 69.5, None, MON, s)[1]


def test_prediction_counts_before_it_happens():
    r = Room("a", "A", "u", priority=5, default_temp=67)
    assert core.room_costs(r, 68.0, None, MON)[1] == 0
    assert core.room_costs(r, 68.0, 70.0, MON)[1] > 0  # heading over the band -> already hot demand


def test_predict_trend():
    hist = [(MON - timedelta(minutes=m), 66.0 + (20 - m) * 0.05) for m in range(20, -1, -2)]  # +0.05 F/min
    p = core.predict(hist, MON)
    assert abs(p - (67.0 + 0.05 * 45)) < 0.05
    assert core.predict(hist[-2:], MON) is None  # too little history
    flat = [(MON - timedelta(minutes=m), 66.0) for m in range(20, -1, -2)]
    assert core.predict(flat, MON) == 66.0


def test_cold_morning_heats_with_every_unit_steady_at_target():
    # Downstairs below the floor -> heat. Every unit holds the 67 target; the inverters slow down on their own
    # as rooms get there, instead of flipping between run/idle setpoints.
    cmds, house, info = _house_started(bed=66.3, down=59.5)
    assert house.mode == "heat" and info["cold"] > 0
    assert cmds["ah"].mode == "heat" and cmds["ah"].setpoint == 67 and cmds["ah"].active
    assert cmds["ms"].mode == "heat" and cmds["ms"].setpoint == 67


def test_all_units_always_share_the_house_mode():
    for bed, down in [(66.3, 59.5), (70.0, 66.5), (64.0, 71.0), (67.0, 67.0)]:
        cmds, house, _ = _house(bed=bed, down=down, house=core.HouseState("heat", MON - timedelta(hours=1)))
        assert {c.mode for c in cmds.values()} == {house.mode}


def test_afternoon_switches_to_cool_after_confirming():
    heating = core.HouseState("heat", MON - timedelta(hours=3))
    _, house, info = _house(bed=68.3, bed_pred=70.5, down=66.5, house=heating)
    assert house.mode == "heat" and house.pending == "cool" and "confirming" in info["reason"]
    later = MON + timedelta(minutes=11)
    cmds, house, info = _house(bed=68.3, bed_pred=70.5, down=66.5, house=house, at=later)
    assert house.mode == "cool" and "switching" in info["reason"]
    assert cmds["ms"].mode == "cool" and cmds["ms"].active
    assert cmds["ah"].mode == "cool" and cmds["ah"].setpoint == 67 and not cmds["ah"].active  # main floor fine


def test_blip_does_not_switch():
    heating = core.HouseState("heat", MON - timedelta(hours=3))
    _, house, _ = _house(bed=68.3, bed_pred=70.5, down=66.5, house=heating)
    assert house.pending == "cool"
    _, house, _ = _house(bed=67.5, down=66.5, house=house, at=MON + timedelta(minutes=5))  # demand went away
    assert house.mode == "heat" and house.pending is None
    _, house, _ = _house(bed=68.3, bed_pred=70.5, down=66.5, house=house, at=MON + timedelta(minutes=12))
    assert house.mode == "heat" and house.pending == "cool"  # confirmation starts over


def test_own_cooling_trend_does_not_flip_back_to_heat():
    cooling = core.HouseState("cool", MON - timedelta(hours=1))
    st = {"ms": UnitState("cool", MON, "cool", active=True)}
    _, house, _ = _house(bed=66.8, bed_pred=64.0, down=67.0, house=cooling, states=st)
    assert house.mode == "cool" and house.pending is None  # falling because we're cooling it
    _, house, _ = _house(bed=66.8, bed_pred=64.0, down=67.0, house=cooling)  # not driven: a real cold trend
    assert house.pending == "heat"


def test_manual_cooling_trend_does_not_trigger_heat():
    # Unit cooled by hand: the bedroom drops fast, but that's the unit, not the room getting cold.
    _, house, _ = _house_started(bed=66.2, bed_pred=63.0, down=67.0, busy={"ms"})
    assert house.mode == "off"
    _, house, _ = _house_started(bed=66.2, bed_pred=63.0, down=67.0)
    assert house.mode == "heat"


def test_idle_house_mode_says_so():
    cmds, house, info = _house(bed=67.0, down=67.0, house=core.HouseState("heat", MON - timedelta(hours=1)))
    assert house.mode == "heat" and "idle" in info["reason"]
    assert "idling" in cmds["ah"].reason and cmds["ah"].setpoint == 67


def test_switch_waits_for_min_time_in_mode():
    recent = core.HouseState("heat", MON - timedelta(minutes=10))
    _, house, info = _house(bed=68.3, bed_pred=70.5, down=66.5, house=recent)
    assert house.mode == "heat" and "holding" in info["reason"]


def test_no_flip_for_mild_opposite_demand():
    # Downstairs a bit cold, bedroom a touch warm: heat keeps going rather than flipping.
    heating = core.HouseState("heat", MON - timedelta(hours=3))
    _, house, _ = _house(bed=68.7, down=65.5, house=heating)
    assert house.mode == "heat"


def test_never_returns_to_off_once_running():
    _, house, _ = _house(bed=67.0, down=67.0, house=core.HouseState("cool", MON - timedelta(hours=1)))
    assert house.mode == "cool"
    _, house, info = _house(bed=67.0, down=67.0)
    assert house.mode == "off" and "comfortable" in info["reason"]


def test_setpoint_stays_put_as_room_reaches_target():
    heating = core.HouseState("heat", MON - timedelta(hours=1))
    a, _, _ = _house(down=66.6, house=heating)
    b, _, _ = _house(down=67.4, house=heating)
    assert a["ah"].setpoint == b["ah"].setpoint == 67  # no run/idle setpoint flip
    assert a["ah"].active and not b["ah"].active


def test_stabilize_holds_small_and_frequent_setpoint_changes():
    s = core.Settings()
    prev = (core.Command("heat", 67.0, "x"), MON - timedelta(minutes=30))
    held, changed = core.stabilize(core.Command("heat", 67.4, "y"), prev, MON, s)
    assert held.setpoint == 67.0 and not changed  # under 1 F
    recent = (core.Command("heat", 67.0, "x"), MON - timedelta(minutes=5))
    held, changed = core.stabilize(core.Command("heat", 69.0, "y"), recent, MON, s)
    assert held.setpoint == 67.0 and not changed  # changed 5 min ago
    out, changed = core.stabilize(core.Command("heat", 69.0, "y"), prev, MON, s)
    assert out.setpoint == 69.0 and changed
    out, changed = core.stabilize(core.Command("cool", 72.0, "y"), recent, MON, s)
    assert out.mode == "cool" and changed  # mode changes always go through


def test_schedule_glides_into_changes():
    r = Room("r", "R", "u", schedule=[ScheduleEntry(list(range(7)), "07:00", 69), ScheduleEntry(list(range(7)), "22:00", 65)])
    assert core.scheduled_temp(r, MON.replace(hour=5, minute=59)) == 65
    assert core.scheduled_temp(r, MON.replace(hour=6, minute=30)) == 67  # halfway through the hour before 07:00
    assert core.scheduled_temp(r, MON.replace(hour=7)) == 69
    assert core.scheduled_temp(r, MON.replace(hour=21, minute=45)) == 66  # setback eases in too


def test_manual_dry_or_fan_holds_and_blocks_others():
    rooms = [Room("bed", "Bedroom", "ms", default_temp=67), Room("down", "Living", "ah", default_temp=67)]
    units = [Unit("ah", "c1"), Unit("ms", "c2")]
    for m in ("dry", "fan_only"):
        cmds, house, info = core.control_step(units, rooms, {"down": (60.0, MON)}, {}, {}, {}, core.HouseState(), MON,
                                              manual={"ms": m})
        assert cmds["ms"].mode == m and cmds["ah"].mode == "off" and house.mode == "off" and m in info["reason"]


def test_sim_spring_day_switches_without_flip_flopping():
    # Cold night, warm sunny afternoon upstairs: heat in the morning, cool in the afternoon, no rapid flipping.
    import math
    rooms = [Room("bed", "Bedroom", "ms", priority=5, default_temp=67), Room("down", "Living", "ah", priority=2, default_temp=67)]
    outdoor = lambda t: 50 + 22 * math.sin(math.pi * (t.hour + t.minute / 60 - 9) / 12)  # ~28 at 3am, ~72 at 3pm
    h = SimHouse(rooms, [Unit("ah", "c1"), Unit("ms", "c2")], outdoor, MON.replace(hour=0))
    h.sim["bed"].ua = h.sim["down"].ua = 0.004
    modes, temps = [], []
    for _ in range(24 * 60):
        h.step(1)
        # sun on the upstairs in the afternoon
        if 12 <= h.now.hour < 18:
            h.sim["bed"].temp += 0.02
        modes.append(h.house.mode)
        temps.append((h.sim["bed"].temp, h.sim["down"].temp))
    switches = sum(1 for a, b in zip(modes, modes[1:]) if a != b and a != "off")
    assert "heat" in modes and "cool" in modes
    assert switches <= 2, switches
    assert h.changes <= 24, h.changes  # setpoint/mode commands across both units in a day: steady, not chattering
    settled = temps[3 * 60:]
    assert min(t for pair in settled for t in pair) > 63.5
    assert max(t for pair in settled for t in pair) < 73.5



# --- a schedule change can switch the house mode (the household keeps it cold, so a setback should cool) ---

def _night(bed, bed_pred, at, house, down=66.5):
    """Last night's setup: priority-5 primary bedroom set back 68 -> 65 at 21:30, main floor on a flat 67."""
    bed_room = Room("bed", "Primary", "ms", priority=5, default_temp=67, schedule=[
        ScheduleEntry(list(range(7)), "07:00", 68), ScheduleEntry(list(range(7)), "21:30", 65)])
    rooms = [bed_room, Room("down", "Main", "ah", priority=2, default_temp=67)]
    units = [Unit("ah", "c1"), Unit("ms", "c2")]
    return core.control_step(units, rooms, {"bed": (bed, at), "down": (down, at)}, {"bed": bed_pred, "down": None},
                             {}, {}, house, at)


def test_evening_setback_switches_heat_to_cool_after_confirming():
    at = MON.replace(hour=22, minute=36)
    _, house, info = _night(67.5, None, at, core.HouseState("heat", at - timedelta(hours=2)))
    assert info["hot"] > 0 and house.mode == "heat" and house.pending == "cool"
    cmds, house, _ = _night(67.5, None, at + timedelta(minutes=11), house)
    assert house.mode == "cool" and cmds["ms"].mode == "cool" and cmds["ms"].active


# --- setpoints don't flicker between two whole degrees ---

def test_rounding_jitter_holds_setpoint_and_reason():
    s = core.Settings()
    prev = (core.Command("cool", 68.0, "at target; idling"), MON - timedelta(minutes=2))
    outs = [core.stabilize(core.Command("cool", float(round(raw)), "at target; idling", False, raw), prev, MON, s)
            for raw in (67.45, 67.55, 67.45, 67.55)]
    assert {o.setpoint for o, _ in outs} == {68.0}
    assert {o.reason for o, _ in outs} == {"at target; idling"}  # no "(holding ...)" alternating in the log
    assert not any(changed for _, changed in outs)


def test_real_move_held_by_interval_says_so():
    s = core.Settings()
    prev = (core.Command("heat", 67.0, "x"), MON - timedelta(minutes=5))
    out, changed = core.stabilize(core.Command("heat", 69.0, "heating toward 69°", True, 69.2), prev, MON, s)
    assert out.setpoint == 67.0 and not changed and "recently changed" in out.reason


def test_single_step_on_coarse_sensor_is_not_a_trend():
    # 00:10 on 2026-10-14: the bedroom unit's sensor stepped 66.2 -> 67.1 once and the trend predicted 68.9.
    hist = [(MON - timedelta(minutes=m), 66.2 if m > 10 else 67.1) for m in range(30, -1, -1)]
    assert core.predict(hist, MON) == 67.1
    flicker = [(MON - timedelta(minutes=m), 66.2 if m % 2 else 67.1) for m in range(30, -1, -1)]
    assert core.predict(flicker, MON) == 67.1
    rising = [(MON - timedelta(minutes=m), 66.2 + 0.9 * ((30 - m) // 10)) for m in range(30, -1, -1)]  # 3 steps up
    assert core.predict(rising, MON) > 68.9


# --- optional free cooling/heating from the weather (off by default) ---

def test_free_cooling_is_off_by_default():
    r = Room("bed", "Primary", "ms", priority=5, schedule=[ScheduleEntry(list(range(7)), "21:30", 65)])
    assert core.room_costs(r, 67.1, None, MON.replace(hour=23), outdoor=55.0)[1] > 0  # a cold night still cools


def test_free_cooling_when_enabled_lets_a_slightly_warm_room_coast():
    s = dataclasses.replace(NEUTRAL, free_delta=5.0)
    at = MON.replace(hour=0, minute=35)
    r = Room("bed", "Primary", "ms", priority=5, schedule=[ScheduleEntry(list(range(7)), "21:30", 65)])
    assert core.room_costs(r, 67.1, None, at, s, outdoor=55.0)[1] == 0
    assert core.room_costs(r, 67.1, None, at, s, outdoor=65.0)[1] > 0  # mild night: no free cooling
    assert core.room_costs(r, 73.0, None, at, s, outdoor=40.0)[1] > 0  # ceiling always counts


def test_free_heating_when_enabled_lets_a_slightly_cool_room_coast():
    s = dataclasses.replace(NEUTRAL, free_delta=5.0)
    r = Room("r", "R", "u", priority=5, default_temp=70)
    assert core.room_costs(r, 68.5, None, MON, s, outdoor=88.0)[0] == 0
    assert core.room_costs(r, 68.5, None, MON, s, outdoor=70.0)[0] > 0
    assert core.room_costs(r, s.floor - 1.0, None, MON, s, outdoor=88.0)[0] > 0  # floor always counts


def test_priority_schedule_steps_and_falls_back_to_fixed_priority():
    from core import PriorityEntry
    wk = [0, 1, 2, 3, 4]
    r = Room("bed", "Bed", "u", priority=4,
             priority_schedule=[PriorityEntry(wk, "08:00", 1), PriorityEntry(wk, "20:00", 5)])
    day = MON.replace(hour=0, minute=0)
    assert core.room_priority(r, day.replace(hour=12)) == 1
    assert core.room_priority(r, day.replace(hour=22)) == 5
    assert core.room_priority(r, day.replace(hour=7)) == 5  # Sunday had no entry, so Friday 20:00 carries
    assert core.room_priority(Room("x", "X", "u", priority=4), day) == 4
    assert core.room_priority(Room("x", "X", "u", priority=9), day) == 5  # clamped


def test_priority_schedule_shifts_unit_toward_the_room_in_use():
    from core import PriorityEntry
    wk = [0, 1, 2, 3, 4, 5, 6]
    living = Room("liv", "Living", "u", priority=3, default_temp=68)
    bed = Room("bed", "Bed", "u", priority=3, default_temp=68,
               priority_schedule=[PriorityEntry(wk, "08:00", 1), PriorityEntry(wk, "21:00", 5)])
    readings_at = lambda t: {"liv": (70.0, t), "bed": (66.0, t)}
    noon, night = MON.replace(hour=12, minute=0), MON.replace(hour=22, minute=0)
    e_noon = core.weighted_error([living, bed], readings_at(noon), noon, Settings())[0]
    e_night = core.weighted_error([living, bed], readings_at(night), night, Settings())[0]
    # noon: (3*2 + 1*-2)/4 = +1.0 (warm living room dominates); night: (3*2 + 5*-2)/8 = -0.5 (cool bedroom does)
    assert abs(e_noon - 1.0) < 1e-9 and abs(e_night + 0.5) < 1e-9


# --- daylight saving: timers measure real time, schedules follow the local clock ---
from datetime import timezone as _tz
from zoneinfo import ZoneInfo as _Zone

_LA = _Zone("America/Los_Angeles")
_FALL = datetime(2026, 11, 1, 8, 0, tzinfo=_tz.utc)  # 01:00 PDT; clocks go 02:00 PDT -> 01:00 PST
_SPRING = datetime(2027, 3, 14, 9, 0, tzinfo=_tz.utc)  # 01:00 PST; clocks go 02:00 PST -> 03:00 PDT


def _local(utc, minutes):
    return (utc + timedelta(minutes=minutes)).astimezone(_LA)  # what dt_util.now() returns


def test_elapsed_is_real_time_across_dst():
    for base in (_FALL, _SPRING):
        assert core.elapsed(_local(base, 90), _local(base, 30)) == 3600


def test_setpoint_interval_is_15_real_minutes_across_fall_back():
    s = Settings()
    first = _local(_FALL, 55)  # 01:55 PDT
    prev = (core.Command("cool", 68.0, raw=68.0), first)
    later = _local(_FALL, 75)  # 01:15 PST, 20 real minutes on (wall clock says 40 minutes earlier)
    _, changed = core.stabilize(core.Command("cool", 66.0, raw=66.0), prev, later, s)
    assert changed


def test_mode_switch_still_confirms_across_spring_forward():
    s = Settings()
    house = core.HouseState("heat", _local(_SPRING, -120), "cool", _local(_SPRING, 58))  # pending since 01:58 PST
    at = _local(_SPRING, 61)  # 03:01 PDT, 3 real minutes later (wall clock says 63)
    new, why = core.choose_mode(0.0, 50.0, house, at, s)
    assert new.mode == "heat" and "confirming" in why


def test_trend_is_sane_through_the_repeated_hour():
    hist = [(_local(_FALL, m), 70.0 - 0.05 * m) for m in range(30, 91, 5)]  # steady fall through 02:00 PDT -> 01:00 PST
    p = core.predict(hist, hist[-1][0], Settings())
    assert p is not None and p < hist[-1][1]


def test_schedule_follows_the_local_clock_on_dst_days():
    r = Room("r", "R", "u", schedule=[ScheduleEntry(list(range(7)), "07:00", 70), ScheduleEntry(list(range(7)), "21:00", 64)],
             priority_schedule=[core.PriorityEntry(list(range(7)), "07:00", 2), core.PriorityEntry(list(range(7)), "21:00", 5)])
    for day in (datetime(2026, 11, 1, tzinfo=_LA), datetime(2027, 3, 14, tzinfo=_LA)):
        assert core.target_temp(r, day.replace(hour=7, minute=0)) == 70
        assert core.target_temp(r, day.replace(hour=5, minute=59)) == 64
        assert core.room_priority(r, day.replace(hour=6, minute=59)) == 5
        assert core.room_priority(r, day.replace(hour=7, minute=0)) == 2


# --- thermostat control: units not driven by us still share the compressor ---
def test_unit_on_its_thermostat_in_heat_or_cool_sets_the_house_mode():
    rooms = [Room("bed", "Bedroom", "ms", default_temp=67), Room("down", "Living", "ah", default_temp=67)]
    units = [Unit("ah", "c1"), Unit("ms", "c2")]
    readings = {"down": (60.0, MON), "bed": (66.0, MON)}  # living room wants heat
    cmds, house, info = core.control_step(units, rooms, readings, {}, {}, {}, core.HouseState(), MON, manual={"ms": "cool"})
    assert house.mode == "cool" and cmds["ah"].mode == "cool" and "following cool" in info["reason"]
    cmds, house, _ = core.control_step(units, rooms, readings, {}, {}, {}, core.HouseState(), MON, manual={"ms": "heat"})
    assert house.mode == "heat" and cmds["ah"].mode == "heat"


def test_unit_on_its_thermostat_in_auto_or_off_does_not_steer():
    rooms = [Room("bed", "Bedroom", "ms", default_temp=67), Room("down", "Living", "ah", default_temp=67)]
    units = [Unit("ah", "c1"), Unit("ms", "c2")]
    readings = {"down": (60.0, MON), "bed": (66.0, MON)}
    for m in ("auto", "off", "heat_cool"):
        _, house, info = _started(rooms, units, readings, manual={"ms": m})
        assert house.mode == "heat" and "following" not in info["reason"], m


# --- sensor dropouts and missing readings ---
def test_short_dropout_keeps_the_last_reading_then_lets_go():
    s = Settings()
    got, last = core.hold_reading((70.0, MON), None, MON, s)
    assert got == (70.0, MON)
    got, last = core.hold_reading(None, last, MON + timedelta(minutes=7), s)
    assert got == (70.0, MON)  # bridged
    got, _ = core.hold_reading(None, last, MON + timedelta(minutes=16), s)
    assert got is None  # past the grace: really gone


def test_no_readings_keeps_the_setpoint_instead_of_jumping_to_a_limit():
    s = Settings()
    u, rooms = Unit("ah", "c1"), [Room("k", "Kitchen", "ah", default_temp=67)]
    cmd = core.unit_command(u, rooms, {}, {}, "cool", None, UnitState("cool"), MON, s)
    assert cmd.mode == "cool" and cmd.setpoint is None and "holding setpoint" in cmd.reason
    held, changed = core.stabilize(cmd, (core.Command("cool", 67.0, raw=67.0), MON - timedelta(hours=1)), MON, s)
    assert held.setpoint == 67.0 and not changed  # mid-run: keep what we sent
    fresh, changed = core.stabilize(cmd, None, MON, s)
    assert fresh.setpoint is None and changed  # just restarted: don't touch the unit's setpoint
    after, changed = core.stabilize(core.Command("cool", 67.0, raw=67.2), (core.Command("cool", 67.0, raw=67.0), MON), MON + timedelta(minutes=2), s)
    assert after.setpoint == 67.0  # readings back: no jump


def test_rooms_on_thermostat_units_do_not_pick_the_house_mode():
    rooms = [Room("bed3", "Bedroom 3", "ms", default_temp=67), Room("k", "Kitchen", "ah", default_temp=67)]
    units = [Unit("ah", "c1"), Unit("ms", "c2")]
    readings = {"bed3": (76.0, MON), "k": (64.5, MON)}  # bedroom 3's unit sensor reads hot; the kitchen is clearly cool
    _, house, info = _started(rooms, units, readings, manual={"ms": "off"})
    assert house.mode == "heat" and "bed3" not in info["rooms"]
    _, house, _ = _started(rooms, units, readings, manual={"ms": "unavailable"})
    assert house.mode == "heat"
    # with nothing under our control, every room still counts (what Room Climate would do)
    _, house, info = core.control_step(units, rooms, readings, {}, {}, {}, core.HouseState(), MON, manual={"ms": "off", "ah": "off"})
    assert "bed3" in info["rooms"]


# --- leaving "off", the hard floor, and units that just changed mode ---
def test_a_hair_under_the_floor_does_not_start_the_house():
    s = Settings()
    r = Room("pb", "Primary", "ms", priority=5, default_temp=65)  # night target 65, as in the house
    u = Unit("ms", "c2")
    c, h = core.room_costs(r, 64.94, None, MON, s)
    assert c == 0 and h == 0  # the floor (63) is below the target band, so a rounding step costs nothing
    _, house, info = core.control_step([u], [r], {"pb": (64.94, MON)}, {}, {}, {}, core.HouseState(), MON)
    assert house.mode == "off" and info["reason"] == "house comfortable"


def test_leaving_off_needs_real_demand_held_for_the_confirmation_time():
    s = Settings()
    h = core.HouseState()
    h, why = core.choose_mode(0.5, 0.0, h, MON, s)  # below start_cost
    assert h.mode == "off" and h.pending is None
    h, why = core.choose_mode(5.0, 0.0, h, MON, s)
    assert h.mode == "off" and h.pending == "heat" and "confirming" in why
    h, why = core.choose_mode(5.0, 0.0, h, MON + timedelta(minutes=5), s)
    assert h.mode == "off" and "5/10" in why
    h, why = core.choose_mode(5.0, 0.0, h, MON + timedelta(minutes=10), s)
    assert h.mode == "heat" and h.since == MON + timedelta(minutes=10)


def test_demand_that_goes_away_or_flips_restarts_the_confirmation():
    s = Settings()
    h, _ = core.choose_mode(5.0, 0.0, core.HouseState(), MON, s)
    h, _ = core.choose_mode(0.2, 0.0, h, MON + timedelta(minutes=6), s)  # gone
    assert h.mode == "off" and h.pending is None
    h, _ = core.choose_mode(5.0, 0.0, h, MON + timedelta(minutes=7), s)
    h, _ = core.choose_mode(0.0, 5.0, h, MON + timedelta(minutes=12), s)  # flipped to cool: new clock
    assert h.pending == "cool" and h.pending_since == MON + timedelta(minutes=12)


# --- rooms on their unit's own sensor; what a unit stores ---
def test_room_reading_its_own_units_sensor_follows_the_house_instead_of_voting():
    units = [Unit("ah", "climate.ah"), Unit("ms", "climate.ms")]
    rooms = [Room("k", "Kitchen", "ah", sensor="sensor.k", default_temp=67),
             Room("b3", "Bedroom 3", "ms", sensor="climate.ms", default_temp=67)]
    # Bedroom 3's unit sensor reads 77 right after heating: it must not flip a house that the kitchen needs heated.
    house = core.HouseState("heat", MON - timedelta(hours=2))
    cmds, h, info = core.control_step(units, rooms, {"k": (64.0, MON), "b3": (77.0, MON)}, {}, {"ms": 77.0}, {}, house, MON, NEUTRAL)
    assert h.mode == "heat" and info["hot"] == 0 and info["rooms"]["b3"][1] > 0  # its demand is shown, not counted
    assert cmds["ms"].mode == "heat" and cmds["ms"].setpoint == 67  # it still holds its own target
    # ...and its swing to 55 in cool doesn't drag the house to heat
    house = core.HouseState("cool", MON - timedelta(hours=2))
    _, h, info = core.control_step(units, rooms, {"k": (68.0, MON), "b3": (55.0, MON)}, {}, {"ms": 55.0}, {}, house, MON, NEUTRAL)
    assert h.mode == "cool" and info["cold"] == 0


def test_stored_setpoint_matches_what_the_units_read_back():
    readback_f = lambda f: round(core.stored_setpoint_c(f) * 1.8 + 32, 1)
    # seen on the units: 64 -> 64.4, 65 -> 65.3, 67 -> 67.1, 68 -> 68.0, 61 -> 60.8, 73 -> 73.4
    for want, got in ((64, 64.4), (65, 65.3), (67, 67.1), (68, 68.0), (61, 60.8), (73, 73.4)):
        assert readback_f(want) == got, (want, readback_f(want))


def test_rooms_sharing_a_unit_vote_with_their_net_demand():
    # The kitchen at 65 at night shares the air handler with a living room and bedroom at 67. Heating for them pushes
    # the kitchen over its band; that alone must not flip the house to cool.
    units = [Unit("ah", "climate.ah")]
    rooms = [Room("k", "Kitchen", "ah", sensor="s.k", priority=2, default_temp=65),
             Room("lr", "Living", "ah", sensor="s.lr", priority=1, default_temp=67),
             Room("b1", "Bedroom 1", "ah", sensor="s.b1", priority=1, default_temp=67)]
    heating = core.HouseState("heat", MON - timedelta(hours=2))
    readings = {"k": (67.0, MON), "lr": (65.3, MON), "b1": (65.2, MON)}  # kitchen 0.5 over (P2); the others 0.7-0.8 under
    _, h, info = core.control_step(units, rooms, readings, {}, {}, {}, heating, MON, NEUTRAL)
    assert h.mode == "heat" and h.pending is None and info["hot"] == 0
    # Only when the zone as a whole is too warm does it ask for cooling.
    readings = {"k": (68.5, MON), "lr": (69.5, MON), "b1": (69.6, MON)}
    _, h, info = core.control_step(units, rooms, readings, {}, {}, {}, heating, MON, NEUTRAL)
    assert info["hot"] > 0 and info["cold"] == 0 and h.pending == "cool"


def test_separate_zones_still_vote_separately():
    # A hot bedroom on its own minisplit and a cold first floor are a real conflict: both count.
    units = [Unit("ah", "climate.ah"), Unit("ms", "climate.ms")]
    rooms = [Room("k", "Kitchen", "ah", sensor="s.k", priority=1, default_temp=67),
             Room("bed", "Bedroom", "ms", sensor="s.bed", priority=5, default_temp=65)]
    _, _, info = core.control_step(units, rooms, {"k": (64.0, MON), "bed": (69.0, MON)}, {}, {}, {},
                                   core.HouseState("heat", MON - timedelta(hours=2)), MON)
    assert info["cold"] > 0 and info["hot"] > 0


# --- comfort leans cool (the household prefers a sweater to being too warm) ---
def test_comfort_band_and_costs_lean_cool():
    s = Settings()
    r = Room("r", "R", "u", priority=3, default_temp=67)
    assert core.room_costs(r, 65.6, None, MON, s) == (0.0, 0.0)  # 1.4 under: still comfortable
    assert core.room_costs(r, 68.4, None, MON, s)[1] > 0  # 1.4 over: too warm
    below = core.room_costs(r, 65.5 - 0.5, None, MON, s)[0]  # 0.5 past the cool edge
    above = core.room_costs(r, 68.0 + 0.5, None, MON, s)[1]  # 0.5 past the warm edge
    assert abs(above - 2 * below) < 1e-9


def test_heating_aims_half_a_degree_cool_and_ties_round_down():
    u = Unit("u", "climate.u")
    for target, want in ((67, 66.0), (68, 67.0)):
        r = Room("r", "R", "u", sensor="s.r", default_temp=target)
        heat = core.unit_command(u, [r], {"r": (target, MON)}, {}, "heat", None, UnitState("heat"), MON, Settings())
        cool = core.unit_command(u, [r], {"r": (target, MON)}, {}, "cool", None, UnitState("cool"), MON, Settings())
        assert heat.setpoint == want and cool.setpoint == target, (target, heat.setpoint, cool.setpoint)
    assert core.round_cool(66.5) == 66 and core.round_cool(66.51) == 67 and core.round_cool(67.49) == 67
