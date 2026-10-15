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
    return core.control_step(units, rooms, readings, pred, {}, states or {}, house or core.HouseState(), at, busy=busy)


def test_costs_zero_inside_band_and_floor_beats_priority():
    s = core.Settings()
    hi_pri = Room("a", "A", "u", priority=5, default_temp=67)
    lo_pri = Room("b", "B", "u", priority=1, default_temp=67)
    assert core.room_costs(hi_pri, 66.2, None, MON) == (0.0, 0.0)  # inside 66-68.5
    assert core.room_costs(hi_pri, 68.4, None, MON) == (0.0, 0.0)
    slightly_cold = core.room_costs(hi_pri, 65.5, None, MON)[0]
    freezing = core.room_costs(lo_pri, 60.0, None, MON)[0]
    assert freezing > 10 * slightly_cold  # below the 65 floor wins regardless of priority
    assert core.room_costs(lo_pri, 73.0, None, MON)[1] > core.room_costs(hi_pri, 69.5, None, MON)[1]


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
    cmds, house, info = _house(bed=66.3, down=59.5)
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
    _, house, _ = _house(bed=66.2, bed_pred=63.0, down=67.0, busy={"ms"})
    assert house.mode == "off"
    _, house, _ = _house(bed=66.2, bed_pred=63.0, down=67.0)
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
    s = Settings(free_delta=5.0)
    at = MON.replace(hour=0, minute=35)
    r = Room("bed", "Primary", "ms", priority=5, schedule=[ScheduleEntry(list(range(7)), "21:30", 65)])
    assert core.room_costs(r, 67.1, None, at, s, outdoor=55.0)[1] == 0
    assert core.room_costs(r, 67.1, None, at, s, outdoor=65.0)[1] > 0  # mild night: no free cooling
    assert core.room_costs(r, 73.0, None, at, s, outdoor=40.0)[1] > 0  # ceiling always counts


def test_free_heating_when_enabled_lets_a_slightly_cool_room_coast():
    s = Settings(free_delta=5.0)
    r = Room("r", "R", "u", priority=5, default_temp=70)
    assert core.room_costs(r, 68.5, None, MON, s, outdoor=88.0)[0] == 0
    assert core.room_costs(r, 68.5, None, MON, s, outdoor=70.0)[0] > 0
    assert core.room_costs(r, 63.0, None, MON, s, outdoor=88.0)[0] > 0  # floor always counts


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
        _, house, info = core.control_step(units, rooms, readings, {}, {}, {}, core.HouseState(), MON, manual={"ms": m})
        assert house.mode == "heat" and "following" not in info["reason"], m
