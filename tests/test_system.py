import os, sys
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import core
from core import Room, Unit, SystemState, UnitState, Settings
from house import SimHouse

T0 = datetime(2026, 7, 15, 12, 0)
UNITS = [Unit("ah", "c1"), Unit("ms1", "c2"), Unit("ms2", "c3")]


def rooms(down_prio=3, up_prio=3):
    return [Room("down", "Down", "ah", priority=down_prio, default_temp=68),
            Room("b1", "B1", "ms1", priority=up_prio, default_temp=68),
            Room("b2", "B2", "ms2", priority=up_prio, default_temp=68)]


def run(rs, temps, state=None, now=T0, settings=Settings()):
    rd = {k: (v, now) for k, v in temps.items()}
    return core.decide_system(UNITS, rs, rd, {}, state or SystemState(), now, settings)


def modes(cmds):
    return {k: c.mode for k, c in cmds.items()}


def test_hot_upstairs_cool_downstairs_high_priority_downstairs_wins_heat():
    cmds, st = run(rooms(down_prio=5, up_prio=2), {"down": 64, "b1": 71, "b2": 71})
    assert modes(cmds) == {"ah": "heat", "ms1": "off", "ms2": "off"}
    assert st.mode == "heat"
    assert cmds["ms1"].hvac_mode == "off"  # default idle mode


def test_hot_upstairs_cool_downstairs_high_priority_upstairs_wins_cool():
    cmds, st = run(rooms(down_prio=1, up_prio=5), {"down": 64, "b1": 71, "b2": 71})
    assert modes(cmds) == {"ah": "off", "ms1": "cool", "ms2": "cool"}
    assert st.mode == "cool"


def test_idle_units_can_fan_instead_of_off():
    cmds, _ = run(rooms(5, 2), {"down": 64, "b1": 71, "b2": 71}, settings=Settings(idle_mode="fan_only"))
    assert cmds["ms1"].hvac_mode == "fan_only" and cmds["ah"].hvac_mode == ""


def test_all_units_never_in_opposite_modes():
    for dp in range(1, 6):
        for up in range(1, 6):
            cmds, _ = run(rooms(dp, up), {"down": 64, "b1": 72, "b2": 66})
            active = {c.mode for c in cmds.values()} - {"off"}
            assert len(active) <= 1


def test_vote_scales_with_how_far_off_not_just_priority():
    # Equal priority; downstairs is 6F cold, one bedroom only 1.2F hot -> heat.
    cmds, _ = run(rooms(3, 3), {"down": 62, "b1": 69.2, "b2": 68})
    assert cmds["ah"].mode == "heat"


def test_within_deadband_stays_off():
    cmds, st = run(rooms(), {"down": 67.5, "b1": 68.5, "b2": 68})
    assert st.mode == "off" and set(modes(cmds).values()) == {"off"}


def test_running_mode_not_displaced_by_slightly_higher_opposing_vote():
    rs = rooms(down_prio=3, up_prio=3)
    _, st = run(rs, {"down": 64, "b1": 68, "b2": 68})  # heating started
    later = T0 + timedelta(minutes=30)
    # Downstairs still 2F short (stay=6); b1 is 1.5F hot (start_cool=4.5). 4.5 < 1.5*6 -> keep heating.
    cmds, st2 = run(rs, {"down": 66, "b1": 69.5, "b2": 68}, st, later)
    assert st2.mode == "heat" and cmds["ah"].mode == "heat" and cmds["ms1"].mode == "off"


def test_yields_when_opposing_demand_clearly_dominates_then_lockout():
    rs = rooms(down_prio=3, up_prio=5)
    _, st = run(rs, {"down": 64, "b1": 68, "b2": 68})
    assert st.mode == "heat"
    t1 = T0 + timedelta(minutes=30)
    # Downstairs almost there (stay=0.5*3=1.5); bedrooms 4F hot at prio 5 -> yield.
    cmds, st = run(rs, {"down": 67.5, "b1": 72, "b2": 72}, st, t1)
    assert st.mode == "off" and set(modes(cmds).values()) == {"off"}
    # Right after: cool wanted but heat->cool lockout (15 min) applies.
    t2 = t1 + timedelta(minutes=5)
    cmds, st = run(rs, {"down": 67.5, "b1": 72, "b2": 72}, st, t2)
    assert st.mode == "off"
    t3 = t1 + timedelta(minutes=16)
    cmds, st = run(rs, {"down": 67.5, "b1": 72, "b2": 72}, st, t3)
    assert st.mode == "cool" and cmds["ms1"].mode == "cool" and cmds["ah"].mode == "off"


def test_min_run_blocks_immediate_yield():
    rs = rooms(down_prio=1, up_prio=5)
    _, st = run(rs, {"down": 64, "b1": 68, "b2": 68})
    cmds, st = run(rs, {"down": 67.5, "b1": 72, "b2": 72}, st, T0 + timedelta(seconds=60))
    assert st.mode == "heat"


def test_stops_all_when_everything_at_target():
    rs = rooms()
    _, st = run(rs, {"down": 64, "b1": 68, "b2": 68})
    cmds, st = run(rs, {"down": 68.2, "b1": 68, "b2": 68}, st, T0 + timedelta(minutes=20))
    assert st.mode == "off" and set(modes(cmds).values()) == {"off"}


def test_stale_room_does_not_vote():
    rs = rooms()
    rd = {"down": (60.0, T0 - timedelta(hours=3)), "b1": (68.0, T0), "b2": (68.0, T0)}
    cmds, st = core.decide_system(UNITS, rs, rd, {}, SystemState(), T0)
    assert st.mode == "off"


def test_sim_mixed_zones_never_opposing_and_priority_room_held():
    rs = [Room("down", "Down", "ah", priority=5, default_temp=68), Room("b1", "B1", "ms1", priority=2, default_temp=68),
          Room("b2", "B2", "ms2", priority=2, default_temp=68)]
    h = SimHouse(rs, UNITS, lambda t: 50.0, T0.replace(hour=0), shared_mode=True)
    h.sim["b1"].temp = h.sim["b2"].temp = 74  # hot upstairs
    h.sim["down"].temp = 63                    # cold downstairs
    worst_down = 0
    for _ in range(12 * 60):
        h.step(1)
        active = {h.cmd[u].mode for u in h.cmd} - {"off"}
        assert len(active) <= 1
        worst_down = max(worst_down, abs(h.sim["down"].temp - 68)) if _ > 180 else worst_down
    assert worst_down < 2.0
