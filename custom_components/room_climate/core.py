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

    # Make the unit's own thermostat chase the *room* temperature: shift the setpoint by
    # how far the unit's sensor disagrees with ours.
    offset = 0.0
    if unit_temp is not None:
        offset = max(-settings.max_offset, min(settings.max_offset, unit_temp - avg))
    setpoint = target + offset
    # Push past target a little so the unit actually runs until we say stop.
    setpoint += 1.0 if mode == "heat" else -1.0
    setpoint = max(unit.min_setpoint, min(unit.max_setpoint, round(setpoint)))
    return Command(mode, float(setpoint), reason)


def apply(state: UnitState, cmd: Command, now: datetime) -> UnitState:
    """Return the unit's state after issuing `cmd`."""
    if cmd.mode == state.mode:
        return state
    return UnitState(cmd.mode, now, cmd.mode if cmd.mode != "off" else state.last_run)
