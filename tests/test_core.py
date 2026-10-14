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


def test_cold_morning_heats_with_bedrooms_on_maintenance():
    # Bedrooms slightly cool but inside the band; downstairs below the floor -> heat, bedroom unit idles at the floor.
    cmds, house, info = _house(bed=66.3, down=59.5)
    assert house.mode == "heat" and info["cold"] > 0
    assert cmds["ah"].mode == "heat" and cmds["ah"].setpoint == 67 and cmds["ah"].active
    assert cmds["ms"].mode == "heat" and cmds["ms"].setpoint == 65 and not cmds["ms"].active


def test_all_units_always_share_the_house_mode():
    for bed, down in [(66.3, 59.5), (70.0, 66.5), (64.0, 71.0), (67.0, 67.0)]:
        cmds, house, _ = _house(bed=bed, down=down, house=core.HouseState("heat", MON - timedelta(hours=1)))
        assert {c.mode for c in cmds.values()} == {house.mode}


def test_afternoon_switches_to_cool_when_upstairs_heads_hot():
    heating = core.HouseState("heat", MON - timedelta(hours=3))
    cmds, house, info = _house(bed=68.3, bed_pred=70.5, down=66.5, house=heating)
    assert house.mode == "cool" and "switching" in info["reason"]
    assert cmds["ms"].mode == "cool" and cmds["ms"].active
    assert cmds["ah"].mode == "cool" and cmds["ah"].setpoint == 72  # main floor is fine -> idles at the ceiling


def test_own_cooling_trend_does_not_flip_back_to_heat():
    cooling = core.HouseState("cool", MON - timedelta(hours=1))
    st = {"ms": UnitState("cool", MON, "cool", active=True)}
    _, house, _ = _house(bed=66.8, bed_pred=64.0, down=67.0, house=cooling, states=st)
    assert house.mode == "cool"  # falling because we're cooling it; at target it idles
    _, house, _ = _house(bed=66.8, bed_pred=64.0, down=67.0, house=cooling)  # not driven: a real cold trend
    assert house.mode == "heat"


def test_manual_cooling_trend_does_not_trigger_heat():
    # Unit cooled by hand: the bedroom drops fast, but that's the unit, not the room getting cold.
    _, house, _ = _house(bed=66.2, bed_pred=63.0, down=67.0, busy={"ms"})
    assert house.mode == "off"
    _, house, _ = _house(bed=66.2, bed_pred=63.0, down=67.0)
    assert house.mode == "heat"


def test_idle_house_mode_says_so():
    cmds, house, info = _house(bed=67.0, down=67.0, house=core.HouseState("heat", MON - timedelta(hours=1)))
    assert house.mode == "heat" and "idle" in info["reason"]
    assert "idle at 65" in cmds["ah"].reason


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


def test_unit_keeps_heating_to_target_then_idles():
    st = {"ah": UnitState("heat", MON, "heat", active=True)}
    heating = core.HouseState("heat", MON - timedelta(hours=1))
    cmds, _, _ = _house(down=66.6, house=heating, states=st)
    assert cmds["ah"].active and cmds["ah"].setpoint == 67  # past the band edge but not yet at target
    cmds, _, _ = _house(down=67.1, house=heating, states=st)
    assert not cmds["ah"].active and cmds["ah"].setpoint == 65


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
    settled = temps[3 * 60:]
    assert min(t for pair in settled for t in pair) > 63.5
    assert max(t for pair in settled for t in pair) < 73.5

