"""Thermal model, optimizer and coupled simulator (no numpy needed). Run: python3 tests/test_model.py"""
import dataclasses
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import core  # noqa: E402
import model as hm  # noqa: E402
import optimizer  # noqa: E402

NIGHT = datetime(2026, 10, 5, 2, 0)  # a Monday, 2am
NEUTRAL = core.Settings(band_below=1.0, band_above=1.5, warm_weight=1.0, aim_offset=0.0)  # see tests/test_core.py


def two_rooms(coupling=0.0):
    m = hm.HouseModel(
        {"down": hm.RoomModel(loss=0.05, heat={"ah": 5.0}, cool={"ah": 4.0}, coupling={"up": coupling}),
         "up": hm.RoomModel(loss=0.05, heat={"ms": 6.0}, cool={"ms": 5.0}, coupling={"down": coupling})},
        {"ah": hm.UnitModel(sensor_rooms=["down"]), "ms": hm.UnitModel(sensor_rooms=["up"])})
    rooms = [core.Room("down", "Down", "ah", sensor="s.down", priority=3, default_temp=67),
             core.Room("up", "Up", "ms", sensor="s.up", priority=3, default_temp=67)]
    units = [core.Unit("ah", "climate.ah"), core.Unit("ms", "climate.ms")]
    return m, rooms, units


def test_physics_heat_loss_and_coupling():
    m, _, _ = two_rooms(coupling=0.5)
    d = hm.derivative(m, {"down": 70.0, "up": 66.0}, 50.0, 0.0, {"ah": 0.0, "ms": 0.0})
    assert d["down"] < 0 and d["up"] < 0.5 * 4 + 1e-9  # leaks outdoors; up is pulled toward down
    assert d["up"] == 0.05 * (50 - 66) + 0.5 * (70 - 66)
    d = hm.derivative(m, {"down": 67.0, "up": 67.0}, 67.0, 0.0, {"ah": 1.0, "ms": -0.5})
    assert d["down"] == 5.0 and d["up"] == -2.5  # full heat downstairs, half cooling upstairs


def test_inverter_starts_modulates_and_stops():
    u = hm.UnitModel()
    assert hm.unit_output(u, "heat", 68, 67.8, False) == (0.0, False)  # inside the start margin: stays off
    y, run = hm.unit_output(u, "heat", 68, 67.0, False)
    assert run and abs(y - 0.8) < 1e-9  # 0.3 + 0.5 * 1.0
    y, run = hm.unit_output(u, "heat", 68, 68.5, True)
    assert run and y == u.ymin  # past setpoint but still running at minimum
    assert hm.unit_output(u, "heat", 68, 69.2, True) == (0.0, False)  # overshot by more than `stop`: stops
    y, _ = hm.unit_output(u, "cool", 66, 70, False)
    assert y == -1.0
    assert hm.unit_output(u, "off", 66, 70, True) == (0.0, False)


def test_sun_direction():
    from datetime import timezone
    tz = timezone(timedelta(hours=-7))
    noon, late = datetime(2026, 10, 17, 13, 0, tzinfo=tz), datetime(2026, 10, 17, 17, 0, tzinfo=tz)
    assert abs(hm.sun_azimuth(noon, 47.68, -122.38) - 181.2) < 2  # HA's sun.sun said 181.2 then
    assert abs(hm.sun_azimuth(datetime(2026, 10, 17, 15, 0, tzinfo=tz), 47.68, -122.38) - 214.7) < 2
    n, l = hm.sun_at(noon, 47.68, -122.38, 0), hm.sun_at(late, 47.68, -122.38, 0)
    assert n["s"] > 0.8 and n["e"] == 0 and l["w"] > l["s"] > 0  # south wall at midday, west wall late
    assert hm.sun_vector(-3, 200)["s"] == 0 and hm.sun_vector(30, None)["s"] == 0  # night; no azimuth known
    assert hm.sun_at(noon, 47.68, -122.38, 100)["s"] < 0.3 * n["s"]  # clouds
    m = hm.HouseModel({"r": hm.RoomModel(loss=0, solar=1.0, sun_dir={"s": 2.0})}, {})
    assert hm.derivative(m, {"r": 67.0}, None, {"h": 0.5, "s": 0.5}, {})["r"] == 1.5
    assert hm.derivative(m, {"r": 67.0}, None, 0.5, {})["r"] == 0.5  # a plain number is overhead only


def test_solar_index():
    assert hm.solar_index(-5, 0) == 0 and hm.solar_index(None) == 0
    assert abs(hm.solar_index(90, 0) - 1) < 1e-9 and abs(hm.solar_index(90, 100) - 0.25) < 1e-9


def test_model_round_trips_through_json_shape():
    m, _, _ = two_rooms(coupling=0.3)
    m.meta = {"note": "x"}
    back = hm.HouseModel.from_dict(m.to_dict())
    assert back.to_dict() == m.to_dict()
    d = m.to_dict()
    d["rooms"]["down"]["future_field"] = 1  # newer files must still load
    assert hm.HouseModel.from_dict(d).rooms["down"].loss == 0.05


def test_band_cost_matches_the_controllers_room_cost():
    s = core.Settings()
    r = core.Room("r", "R", "u", priority=4, default_temp=67)
    for t in (58.0, 62.9, 65.5, 66.0, 67.0, 68.6, 71.0, 74.5):
        c, h = core.room_costs(r, t, None, NIGHT, s)
        assert abs(optimizer.band_cost(t, 67, 4, s) - (c + h)) < 1e-9, t


def _plan(m, rooms, units, temps, house, outdoor=50.0, **kw):
    ps = optimizer.PlanSettings(**kw)
    n = int(ps.horizon_h * 60 / ps.step_min)
    return optimizer.plan(m, rooms, units, temps, NIGHT, house, {"ah": 67, "ms": 67}, {}, [outdoor] * n, [0.0] * n,
                          NEUTRAL, ps)


def test_plan_heats_a_cold_house_and_cools_a_hot_one():
    m, rooms, units = two_rooms()
    p = _plan(m, rooms, units, {"down": 63.0, "up": 63.5}, core.HouseState())
    assert p.mode == "heat" and p.by_mode["off"]["cost"] > p.cost
    p = _plan(m, rooms, units, {"down": 72.0, "up": 71.5}, core.HouseState(), outdoor=80.0)
    assert p.mode == "cool"
    p = _plan(m, rooms, units, {"down": 67.0, "up": 67.0}, core.HouseState(), outdoor=67.0)
    assert p.mode == "off"  # nothing to do


def test_plan_respects_the_minimum_time_in_a_mode():
    m, rooms, units = two_rooms()
    just_switched = core.HouseState("heat", NIGHT - timedelta(minutes=5))
    p = _plan(m, rooms, units, {"down": 72.0, "up": 72.0}, just_switched, outdoor=80.0)
    assert p.mode == "heat" and set(p.by_mode) == {"heat"}  # cool isn't even considered yet


def test_priorities_settle_an_upstairs_downstairs_conflict():
    m, rooms, units = two_rooms(coupling=0.2)
    house = core.HouseState("heat", NIGHT - timedelta(hours=3))
    temps = {"down": 63.5, "up": 72.0}  # downstairs cold, upstairs hot, mild night: one mode can't fix both
    rooms[0].priority, rooms[1].priority = 1, 5
    assert _plan(m, rooms, units, temps, house, outdoor=60.0).mode == "cool"  # the bedroom matters more tonight
    rooms[0].priority, rooms[1].priority = 5, 1
    assert _plan(m, rooms, units, temps, house, outdoor=60.0).mode == "heat"


def test_setpoints_follow_a_schedule_change_inside_the_horizon():
    m, rooms, units = two_rooms()
    wake = NIGHT.replace(hour=5, minute=0)
    rooms[1].schedule = [core.ScheduleEntry(list(range(7)), "07:00", 70), core.ScheduleEntry(list(range(7)), "22:00", 66)]
    house = core.HouseState("heat", wake - timedelta(hours=5))
    ps = optimizer.PlanSettings(horizon_h=3, step_min=15, substeps=3)
    n = int(ps.horizon_h * 60 / ps.step_min)
    p = optimizer.plan(m, rooms, units, {"down": 67.0, "up": 66.0}, wake, house, {"ah": 67, "ms": 66}, {},
                       [50.0] * n, [0.0] * n, core.Settings(), ps)
    assert p.later["ms"] > p.setpoints["ms"]  # the later block is planned for the 7am target, the first block isn't


def test_sim_mpc_avoids_the_heuristics_zone_conflict_flip_flop():
    import coupled
    truth = coupled.the_house()
    cfg = {"rooms": [
        {"id": "kitchen", "unit": "air_handler", "priority": 2, "default_temp": 65},  # kitchen wants 65 at night...
        {"id": "living_room", "unit": "air_handler", "priority": 1, "default_temp": 67},  # ...its zone wants 67
        {"id": "bedroom_1", "unit": "air_handler", "priority": 1, "default_temp": 67},
        {"id": "primary_bedroom", "unit": "minisplit_1", "priority": 5, "default_temp": 65},
        {"id": "bedroom_3", "unit": "minisplit_2", "priority": 4, "default_temp": 67}],
        "units": [{"id": u} for u in truth.units], "settings": {"floor": 63.0}}
    rooms, units = coupled.rooms_units_from_config(cfg, truth)
    start = datetime(2026, 10, 16)
    w = lambda: coupled.Weather(start, 2, mean=52, seed=4)
    for drive in (False, True):  # with the old fixed setpoints, and with the outer loop under both controllers
        s = dataclasses.replace(coupled.settings_from_config(cfg), drive=drive)
        heur = coupled.run(truth, rooms, units, coupled.HeuristicController(rooms, units, s), start, 1, w(), s, seed=4)
        mpc = coupled.run(truth, rooms, units, coupled.MPCController(rooms, units, truth, s), start, 1, w(), s, seed=4)
        if not drive:
            old = heur
            assert heur["mode_switches"] >= 4  # heating for the zone overheats the kitchen -> cool -> zone cold -> heat
        assert mpc["mode_switches"] <= old["mode_switches"] / 3, (drive, mpc)
        assert mpc["discomfort"] <= heur["discomfort"] * 1.25, (drive, mpc, heur)
        assert mpc["energy_unit_h"] < old["energy_unit_h"], (drive, mpc)
    assert heur["mode_switches"] <= old["mode_switches"] / 3, heur  # the outer loop alone ends the flip-flop


def test_plan_errs_cool_when_it_has_to_miss():
    m, rooms, units = two_rooms()
    house = core.HouseState("heat", NIGHT - timedelta(hours=3))
    p = _plan(m, rooms, units, {"down": 66.5, "up": 66.5}, house, outdoor=50.0)
    s = core.Settings()
    ps = optimizer.PlanSettings()
    n = int(ps.horizon_h * 60 / ps.step_min)
    q = optimizer.plan(m, rooms, units, {"down": 66.5, "up": 66.5}, NIGHT, house, {"ah": 67, "ms": 67}, {},
                       [50.0] * n, [0.0] * n, s, ps)
    assert sum(q.setpoints.values()) < sum(p.setpoints.values())  # the cool-leaning defaults plan lower setpoints
    assert optimizer.band_cost(68.5, 67, 3, s) == 2 * optimizer.band_cost(65.0, 67, 3, s)  # 0.5 past each edge


def test_units_on_their_thermostat_are_left_alone_and_set_the_mode():
    m, rooms, units = two_rooms(coupling=0.2)
    house = core.HouseState("heat", NIGHT - timedelta(hours=3))
    ps = optimizer.PlanSettings()
    n = int(ps.horizon_h * 60 / ps.step_min)
    p = optimizer.plan(m, rooms, units, {"down": 66.0, "up": 71.0}, NIGHT, house, {"ah": 67, "ms": 67}, {"ms": True},
                       [50.0] * n, [0.0] * n, NEUTRAL, ps, fixed={"ms": ("cool", 66.0)})
    assert p.mode == "cool" and set(p.by_mode) == {"cool"}  # follows the thermostat unit's cool
    assert "ms" not in p.setpoints and "ah" in p.setpoints  # only plans the unit it can command


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
