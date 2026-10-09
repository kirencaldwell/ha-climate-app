"""Pure control logic (no Home Assistant imports). All temperatures are degrees F.

A *room* has a wifi sensor, a target (from its schedule) and a priority.
A *unit* is a Mitsubishi climate entity serving one or more rooms. The
controller decides each unit's hvac mode and setpoint from the rooms' sensors,
not from the unit's own (poorly placed) sensor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

MIN_PRIORITY, MAX_PRIORITY = 1, 5


@dataclass
class ScheduleEntry:
    days: list[int]  # 0=Monday .. 6=Sunday
    start: str  # "HH:MM"
    temp: float

    def minutes(self) -> int:
        h, m = self.start.split(":")
        return int(h) * 60 + int(m)


@dataclass
class Room:
    id: str
    name: str
    unit: str
    sensor: Optional[str] = None
    priority: int = 3
    default_temp: float = 68.0
    schedule: list[ScheduleEntry] = field(default_factory=list)
    hold_temp: Optional[float] = None  # manual override ...
    hold_until: Optional[datetime] = None  # ... until this time (None = until next schedule change)
    floor: int = 1
    shape: dict = field(default_factory=dict)  # {"x","y","w","h"} on a 0-100 map grid
    windows: list = field(default_factory=list)  # [{"facing": compass deg, "area": relative size}], for solar gain


@dataclass
class Unit:
    id: str
    entity: str
    min_setpoint: float = 61.0
    max_setpoint: float = 88.0


@dataclass
class Settings:
    deadband: float = 1.0  # start heating/cooling when weighted error passes this
    stop_margin: float = 0.0  # stop when error returns to within this of target
    min_run_s: int = 300  # don't stop sooner than this after starting
    min_off_s: int = 300  # don't restart sooner than this after stopping
    mode_switch_s: int = 900  # off time required between heat <-> cool
    max_offset: float = 6.0  # clamp on sensor-vs-unit-sensor compensation
    stale_s: int = 900  # reading older than this is ignored
    allow_cool: bool = True
    allow_heat: bool = True
    switch_ratio: float = 1.5  # opposite mode must outweigh the active one by this factor to take over
    precompensate_min: int = 60  # look this far ahead (free-drift prediction) when deciding to start
    pred_weight: float = 0.6  # how much the predicted error counts against the current error when starting
    pred_guard: float = 3.0  # a room already this far off ignores prediction: act on what is happening now
    preheat_max: float = 0.5  # run up to this far past target to bank heat/cool ahead of a predicted swing
    idle_mode: str = "off"  # what a unit that isn't needed does while the system runs the other units: "off" | "fan_only"


@dataclass
class UnitState:
    mode: str = "off"  # "off" | "heat" | "cool"
    since: Optional[datetime] = None
    last_run: Optional[str] = None  # last non-off mode, for heat<->cool lockout


@dataclass
class Command:
    mode: str
    setpoint: Optional[float]
    reason: str = ""
    hvac_mode: str = ""  # what to send to the climate entity ("heat"/"cool"/"off"/"fan_only"); "" = same as mode


def _hold_active(room: Room, now: datetime) -> bool:
    if room.hold_temp is None:
        return False
    return room.hold_until is None or now < room.hold_until


def scheduled_temp(room: Room, now: datetime) -> float:
    """Latest schedule entry at or before `now`, looking back up to a week."""
    best = None
    for back in range(8):
        day = now - timedelta(days=back)
        for e in room.schedule:
            if day.weekday() not in e.days:
                continue
            at = day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=e.minutes())
            if at <= now and (best is None or at > best[0]):
                best = (at, e.temp)
        if best:
            return best[1]
    return room.default_temp


def target_temp(room: Room, now: datetime) -> float:
    return room.hold_temp if _hold_active(room, now) else scheduled_temp(room, now)


def weighted_error(rooms: list[Room], readings: dict[str, tuple[float, datetime]],
                   now: datetime, settings: Settings):
    """Priority-weighted (reading - target). Returns (error, avg_reading) or None."""
    num = den = rsum = 0.0
    for r in rooms:
        got = readings.get(r.id)
        if not got or (now - got[1]).total_seconds() > settings.stale_s:
            continue
        w = max(MIN_PRIORITY, min(MAX_PRIORITY, r.priority))
        num += w * (got[0] - target_temp(r, now))
        rsum += w * got[0]
        den += w
    if not den:
        return None
    return num / den, rsum / den


def decide(unit: Unit, rooms: list[Room], readings: dict[str, tuple[float, datetime]],
           unit_temp: Optional[float], state: UnitState, now: datetime,
           settings: Settings = Settings()) -> Command:
    """Return the command for one unit. Caller stores the new state if mode changed."""
    res = weighted_error(rooms, readings, now, settings)
    if res is None:
        # No trustworthy sensor: leave the unit alone rather than guess.
        return Command(state.mode, None, "no fresh sensor readings; holding")
    err, avg = res
    target = avg - err  # weighted-average target
    held = (now - state.since).total_seconds() if state.since else 1e9

    mode = state.mode
    reason = "in deadband"
    if mode == "heat":
        if err >= settings.stop_margin and held >= settings.min_run_s:
            mode, reason = "off", "reached target"
        else:
            reason = "heating"
    elif mode == "cool":
        if err <= -settings.stop_margin and held >= settings.min_run_s:
            mode, reason = "off", "reached target"
        else:
            reason = "cooling"
    else:
        want = None
        if err <= -settings.deadband and settings.allow_heat:
            want = "heat"
        elif err >= settings.deadband and settings.allow_cool:
            want = "cool"
        if want:
            # Respect off-time lockouts: restart delay always, longer when flipping modes.
            if held < settings.min_off_s:
                reason = "waiting out min off time"
            elif state.last_run not in (None, want) and held < settings.mode_switch_s:
                reason = "waiting out heat/cool switch lockout"
            else:
                mode, reason = want, "heating" if want == "heat" else "cooling"

    if mode == "off":
        return Command("off", None, reason)

    setpoint = _setpoint(unit, mode, target, avg, unit_temp, settings)
    return Command(mode, setpoint, reason)


def _setpoint(unit: Unit, mode: str, target: float, avg: float, unit_temp: Optional[float],
              settings: Settings) -> float:
    """Make the unit's own thermostat chase the *room* temperature: shift the setpoint by how far
    the unit's sensor disagrees with ours, then push 1F past target so it keeps running until we stop it."""
    offset = 0.0
    if unit_temp is not None:
        offset = max(-settings.max_offset, min(settings.max_offset, unit_temp - avg))
    sp = target + offset + (1.0 if mode == "heat" else -1.0)
    return float(max(unit.min_setpoint, min(unit.max_setpoint, round(sp))))


def apply(state: UnitState, cmd: Command, now: datetime) -> UnitState:
    """Return the unit's state after issuing `cmd`."""
    if cmd.mode == state.mode:
        return state
    return UnitState(cmd.mode, now, cmd.mode if cmd.mode != "off" else state.last_run)


# --- Multi-zone systems: one outdoor unit, so every indoor unit shares one mode -------------------

@dataclass
class SystemState:
    mode: str = "off"  # "heat"/"cool" while any unit is running, else "off"
    since: Optional[datetime] = None
    last_run: Optional[str] = None
    units: dict[str, UnitState] = field(default_factory=dict)


def room_errors(room: Room, temp: float, now: datetime, settings: Settings, pred_temp: Optional[float]):
    """(start, stop_heat, stop_cool) errors, each reading minus target, for one room.

    `pred_temp` is the free-drift (unit idle) temperature predicted `precompensate_min` ahead, or None
    to behave exactly as before. Prediction makes the room start early (it will leave the band) and,
    while running, carry on a little past target (`preheat_max`) so it can coast through the swing.
    Only drift is anticipated, not schedule changes, so a setback never triggers active cooling."""
    e_now = temp - target_temp(room, now)
    if pred_temp is None:
        return e_now, e_now, e_now
    e_pred = pred_temp - target_temp(room, now)
    if abs(e_now) >= settings.pred_guard:
        start = e_now
    else:
        start = (1 - settings.pred_weight) * e_now + settings.pred_weight * e_pred
    cap = settings.preheat_max
    return start, e_now - min(cap, max(0.0, -e_pred)), e_now + min(cap, max(0.0, e_pred))


def _unit_errs(rooms: list[Room], readings, now: datetime, settings: Settings, preds):
    """Priority-weighted (e_now, avg_reading, start, stop_heat, stop_cool) for a unit, or None."""
    tot = [0.0] * 5
    den = 0.0
    for r in rooms:
        got = readings.get(r.id)
        if not got or (now - got[1]).total_seconds() > settings.stale_s:
            continue
        w = max(MIN_PRIORITY, min(MAX_PRIORITY, r.priority))
        start, sh, sc = room_errors(r, got[0], now, settings, (preds or {}).get(r.id))
        for i, v in enumerate((got[0] - target_temp(r, now), got[0], start, sh, sc)):
            tot[i] += w * v
        den += w
    return None if not den else tuple(v / den for v in tot)


def _votes(rooms: list[Room], readings, now: datetime, settings: Settings, preds=None):
    """Priority-weighted need to heat / cool across all rooms.

    start_*: rooms past the deadband (what can start a mode). stay_*: any shortfall at all
    (what keeps the running mode alive), which is what gives the vote hysteresis."""
    start_h = start_c = stay_h = stay_c = 0.0
    for r in rooms:
        got = readings.get(r.id)
        if not got or (now - got[1]).total_seconds() > settings.stale_s:
            continue
        w = max(MIN_PRIORITY, min(MAX_PRIORITY, r.priority))
        start, sh, sc = room_errors(r, got[0], now, settings, (preds or {}).get(r.id))
        if start <= -settings.deadband and settings.allow_heat:
            start_h += w * -start
        if start >= settings.deadband and settings.allow_cool:
            start_c += w * start
        stay_h += w * max(0.0, -sh - settings.stop_margin)
        stay_c += w * max(0.0, sc - settings.stop_margin)
    return start_h, start_c, stay_h, stay_c


def decide_system(units: list[Unit], rooms: list[Room], readings: dict[str, tuple[float, datetime]],
                  unit_temps: dict[str, Optional[float]], state: SystemState, now: datetime,
                  settings: Settings = Settings(), preds: Optional[dict[str, float]] = None) -> tuple[dict[str, Command], SystemState]:
    """Decide every unit's command when all units must share one heat/cool mode.

    1. Rooms vote (priority x how far off target) for heat vs cool. The higher total wins.
    2. The running mode is only displaced when the opposite vote beats the running mode's remaining
       need by `switch_ratio`, after min run time, and the new mode then waits out `mode_switch_s`.
    3. If `preds` (room id -> predicted idle temperature `precompensate_min` ahead) is given, rooms vote
       and units start on a blend of current and predicted error, and run a little past target (see `room_errors`).
    4. Within the chosen mode each unit runs its own start/stop hysteresis. Units that don't need it
       coast (`idle_mode`) rather than fight the shared mode.
    """
    start_h, start_c, stay_h, stay_c = _votes(rooms, readings, now, settings, preds)
    sys_held = (now - state.since).total_seconds() if state.since else 1e9
    why = ""

    M: Optional[str] = None
    if state.mode != "off":
        M = state.mode
        own_stay, opp = (stay_h, start_c) if M == "heat" else (stay_c, start_h)
        if opp > settings.switch_ratio * own_stay and sys_held >= settings.min_run_s:
            M, why = None, f"yielding {state.mode} to higher-priority {'cool' if M == 'heat' else 'heat'} demand"
    elif start_h or start_c:
        want = "heat" if start_h > start_c else "cool"  # ties go to cool
        if sys_held < settings.min_off_s:
            why = "waiting out min off time"
        elif state.last_run not in (None, want) and sys_held < settings.mode_switch_s:
            why = "waiting out heat/cool switch lockout"
        else:
            M = want

    cmds: dict[str, Command] = {}
    new_units: dict[str, UnitState] = {}
    for u in units:
        mine = [r for r in rooms if r.unit == u.id]
        us = state.units.get(u.id, UnitState())
        res = _unit_errs(mine, readings, now, settings, preds)
        uheld = (now - us.since).total_seconds() if us.since else 1e9
        mode = us.mode if us.mode == M else "off"  # a unit running the old mode is dropped on a switch
        reason = why or "idle"
        if res is None:
            if mode != "off":
                cmds[u.id] = Command(mode, None, "no fresh sensor readings; holding")
                new_units[u.id] = us
                continue
        elif M is not None:
            e_now, avg, e_start, e_sh, e_sc = res
            start_need = -e_start if M == "heat" else e_start
            stop_need = -e_sh if M == "heat" else e_sc
            if mode == M:
                if stop_need <= -settings.stop_margin and uheld >= settings.min_run_s:
                    mode, reason = "off", "reached target"
                else:
                    reason = "heating" if M == "heat" else "cooling"
            elif start_need >= settings.deadband and uheld >= settings.min_off_s:
                mode, reason = M, "heating" if M == "heat" else "cooling"
            elif M and not why:
                reason = f"coasting; system is in {M} mode"
        if mode == "off" or res is None:
            coasting = state.mode != "off" and M is not None
            hv = settings.idle_mode if coasting else "off"
            cmds[u.id] = Command("off", None, reason, hv)
        else:
            e_now, avg, _s, e_sh, e_sc = res
            carry = (e_now - e_sh) if mode == "heat" else -(e_sc - e_now)  # extra reach for pre-heat/pre-cool
            cmds[u.id] = Command(mode, _setpoint(u, mode, avg - e_now + carry, avg, unit_temps.get(u.id), settings), reason)
        new_units[u.id] = apply(us, cmds[u.id], now)

    running = [c.mode for c in cmds.values() if c.mode != "off"]
    new_mode = running[0] if running else "off"
    if new_mode == state.mode:
        new = SystemState(state.mode, state.since, state.last_run, new_units)
    else:
        new = SystemState(new_mode, now, new_mode if new_mode != "off" else state.mode, new_units)
    # System that is still running keeps every idle unit coasting, not hard off.
    if new_mode != "off":
        for u in units:
            c = cmds[u.id]
            if c.mode == "off":
                c.hvac_mode = settings.idle_mode
    return cmds, new
