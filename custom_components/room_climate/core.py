"""Pure control logic (no Home Assistant imports). All temperatures are degrees F.

A *room* has a wifi sensor, a target (from its schedule) and a priority.
A *unit* is a Mitsubishi climate entity serving one or more rooms. The
controller picks one house-wide mode (the units share an outdoor compressor) from
the rooms' current and predicted temperatures, then a setpoint per unit: its
rooms' target when they need it, else a maintenance setpoint near the limits.
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
    unit: Optional[str]  # None = monitor-only (sensor shown, never drives a unit)
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
    # Comfort band around each room's target: inside it a room is "fine" and asks for nothing.
    band_below: float = 1.0
    band_above: float = 1.5
    # House-wide hard limits: a room past these outweighs any priority.
    floor: float = 65.0
    ceiling: float = 72.0
    hard_weight: float = 20.0
    # Prediction: project each room's recent trend this far ahead.
    horizon_s: int = 2700
    history_s: int = 1800
    max_predicted_change: float = 3.0
    # House mode switching (heat <-> cool): only when the other side is clearly worse, and not too often.
    min_mode_s: int = 1200
    switch_ratio: float = 1.5
    switch_margin: float = 0.5
    min_cost: float = 0.05  # demand below this counts as comfortable
    max_offset: float = 6.0  # clamp on sensor-vs-unit-sensor compensation
    # Reading older than this is ignored. Long, because Bluetooth sensors (e.g. Govee) only update when the value
    # changes; a sensor that really drops out goes unavailable instead.
    stale_s: int = 7200
    allow_cool: bool = True
    allow_heat: bool = True
    deadband: float = 1.0  # legacy (card gauge width); kept so stored settings still load
    stop_margin: float = 0.0  # legacy
    min_run_s: int = 300  # legacy
    min_off_s: int = 300  # legacy
    mode_switch_s: int = 900  # legacy


@dataclass
class UnitState:
    mode: str = "off"  # "off" | "heat" | "cool"
    since: Optional[datetime] = None
    last_run: Optional[str] = None  # last non-off mode
    active: bool = False  # driving its rooms to target (vs idling at the maintenance setpoint)


@dataclass
class Command:
    mode: str
    setpoint: Optional[float]
    reason: str = ""
    active: bool = False


@dataclass
class HouseState:
    """The one mode all units share (they have a common outdoor compressor)."""
    mode: str = "off"  # "off" until something first needs conditioning, then "heat" | "cool"
    since: Optional[datetime] = None


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


def predict(history: list[tuple[datetime, float]], now: datetime, settings: Settings = Settings()) -> Optional[float]:
    """Project a room's temperature `horizon_s` ahead from a least-squares trend over the last `history_s`."""
    pts = [(t, v) for t, v in history if (now - t).total_seconds() <= settings.history_s]
    if len(pts) < 3 or (pts[-1][0] - pts[0][0]).total_seconds() < 600:
        return None
    xs = [(t - pts[0][0]).total_seconds() for t, _ in pts]
    ys = [v for _, v in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    change = max(-settings.max_predicted_change, min(settings.max_predicted_change, slope * settings.horizon_s))
    return ys[-1] + change


def room_costs(room: Room, current: float, predicted: Optional[float], now: datetime,
               settings: Settings = Settings()) -> tuple[float, float]:
    """(cold, hot) discomfort for one room. A room counts as cold if it is cold now *or* will be, same for hot."""
    t = target_temp(room, now)
    w = max(MIN_PRIORITY, min(MAX_PRIORITY, room.priority))
    lo = current if predicted is None else min(current, predicted)
    hi = current if predicted is None else max(current, predicted)
    cold = w * max(0.0, (t - settings.band_below) - lo) ** 2 + settings.hard_weight * max(0.0, settings.floor - lo) ** 2
    hot = w * max(0.0, hi - (t + settings.band_above)) ** 2 + settings.hard_weight * max(0.0, hi - settings.ceiling) ** 2
    return cold, hot


def choose_mode(cold: float, hot: float, house: HouseState, now: datetime,
                settings: Settings = Settings()) -> tuple[str, str]:
    """Pick the house mode from total cold vs hot demand. Never goes back to off once running."""
    can = {"heat": settings.allow_heat, "cool": settings.allow_cool}
    if house.mode not in ("heat", "cool"):
        want = "heat" if cold >= hot else "cool"
        if max(cold, hot) < settings.min_cost or not can[want]:
            return "off", "house comfortable"
        return want, f"{'cold' if want == 'heat' else 'hot'} demand {max(cold, hot):.2f}"
    mine, theirs = (cold, hot) if house.mode == "heat" else (hot, cold)
    other = "cool" if house.mode == "heat" else "heat"
    if can[other] and theirs >= settings.min_cost and theirs > mine * settings.switch_ratio + settings.switch_margin:
        held = (now - house.since).total_seconds() if house.since else 1e9
        if held < settings.min_mode_s:
            return house.mode, f"{other} wanted ({theirs:.2f} vs {mine:.2f}); holding {house.mode} min time"
        return other, f"switching: {'hot' if other == 'cool' else 'cold'} demand {theirs:.2f} vs {mine:.2f}"
    if max(cold, hot) < settings.min_cost:
        return house.mode, f"{house.mode} mode, idle: nothing needed"
    return house.mode, f"{house.mode} (cold {cold:.2f}, hot {hot:.2f})"


def unit_command(unit: Unit, rooms: list[Room], readings: dict[str, tuple[float, datetime]],
                 predicted: dict[str, Optional[float]], mode: str, unit_temp: Optional[float],
                 state: UnitState, now: datetime, settings: Settings = Settings()) -> Command:
    """Setpoint for one unit in the house mode: target when its rooms need it, else a maintenance setpoint
    (floor in heat, ceiling in cool) so it stays in the shared mode but only works near the limits."""
    if mode not in ("heat", "cool"):
        return Command("off", None, "house comfortable")
    res = weighted_error(rooms, readings, now, settings)
    maint = settings.floor if mode == "heat" else settings.ceiling
    if res is None:
        sp = max(unit.min_setpoint, min(unit.max_setpoint, round(maint)))
        return Command(mode, float(sp), "no fresh sensors; maintenance")
    err, avg = res
    target = avg - err
    # Priority-weighted predicted temperature (falls back to the reading where there is no trend yet)
    num = den = 0.0
    for r in rooms:
        got = readings.get(r.id)
        if not got or (now - got[1]).total_seconds() > settings.stale_s:
            continue
        w = max(MIN_PRIORITY, min(MAX_PRIORITY, r.priority))
        p = predicted.get(r.id)
        num += w * (got[0] if p is None else p)
        den += w
    pavg = num / den
    was = state.active and state.mode == mode
    if mode == "heat":
        # Start below the band (now or predicted), keep going until the rooms reach target.
        active = avg < target if was else min(avg, pavg) < target - settings.band_below
    else:
        active = avg > target if was else max(avg, pavg) > target + settings.band_above
    base = target if active else maint
    offset = 0.0
    if unit_temp is not None:  # make the unit's own thermostat chase the *room* temperature
        offset = max(-settings.max_offset, min(settings.max_offset, unit_temp - avg))
    sp = max(unit.min_setpoint, min(unit.max_setpoint, round(base + offset)))
    why = ("heating to target" if mode == "heat" else "cooling to target") if active else f"idle at {sp}° (rooms fine)"
    return Command(mode, float(sp), why, active)


MANUAL_MODES = ("dry", "fan_only")


def control_step(units: list[Unit], rooms: list[Room], readings: dict[str, tuple[float, datetime]],
                 predicted: dict[str, Optional[float]], unit_temps: dict[str, Optional[float]],
                 states: dict[str, UnitState], house: HouseState, now: datetime,
                 settings: Settings = Settings(), manual: Optional[dict[str, str]] = None,
                 busy: Optional[set[str]] = None):
    """One control tick for the whole house -> (commands by unit id, new HouseState, info dict).

    All units always share one mode. A unit put into dry or fan by hand takes over: the controller holds
    and keeps every other unit off so the compressor never gets conflicting modes. `busy` lists units that are
    actually heating/cooling right now (by us or by hand): their rooms' trends come from the unit, not the room.
    """
    manual = {u: m for u, m in (manual or {}).items() if m in MANUAL_MODES}
    if manual:
        m = next(iter(manual.values()))
        cmds = {u.id: (Command(manual[u.id], None, f"manual {manual[u.id]}; holding") if u.id in manual
                       else Command("off", None, f"blocked: manual {m} on another unit")) for u in units}
        return cmds, house, {"mode": m, "reason": f"manual {m} in use", "cold": None, "hot": None}

    cold = hot = 0.0
    for r in rooms:
        if r.unit is None:
            continue  # monitor-only rooms don't steer the house
        got = readings.get(r.id)
        if not got or (now - got[1]).total_seconds() > settings.stale_s:
            continue
        # A room whose unit is heating/cooling it (by us or by hand) trends because of the unit, and that stops at
        # its setpoint, so the trend says nothing about the opposite need (it would flip cool->heat as cooling
        # works). Use the reading alone.
        driven = states.get(r.unit, UnitState()).active or r.unit in (busy or ())
        c, h = room_costs(r, got[0], None if driven else predicted.get(r.id), now, settings)
        cold, hot = cold + c, hot + h
    mode, reason = choose_mode(cold, hot, house, now, settings)
    new_house = house if mode == house.mode else HouseState(mode, now)
    cmds = {}
    for u in units:
        mine = [r for r in rooms if r.unit == u.id]
        cmds[u.id] = unit_command(u, mine, readings, predicted, mode, unit_temps.get(u.id),
                                  states.get(u.id, UnitState()), now, settings)
    return cmds, new_house, {"mode": mode, "reason": reason, "cold": round(cold, 2), "hot": round(hot, 2)}


def regime_changed(old: UnitState, new: UnitState) -> bool:
    """True when a unit starts/stops driving its rooms or changes mode; trends from before no longer apply."""
    return (old.mode, old.active) != (new.mode, new.active)


def apply(state: UnitState, cmd: Command, now: datetime) -> UnitState:
    """Return the unit's state after issuing `cmd`."""
    if cmd.mode == state.mode:
        return UnitState(state.mode, state.since, state.last_run, cmd.active)
    return UnitState(cmd.mode, now, cmd.mode if cmd.mode != "off" else state.last_run, cmd.active)


def diagnose(unit: Unit, rooms: list[Room], readings: dict[str, tuple[float, datetime]],
             unit_temp: Optional[float], state: UnitState, cmd: Optional[Command], now: datetime,
             settings: Settings = Settings()) -> dict:
    """Why a unit is doing what it's doing: its mode/setpoint plus each served room's weight and pull on the error."""
    mine = [r for r in rooms if r.unit == unit.id]
    items, den = [], 0.0
    for r in mine:
        got = readings.get(r.id)
        fresh = bool(got) and (now - got[1]).total_seconds() <= settings.stale_s
        w = max(MIN_PRIORITY, min(MAX_PRIORITY, r.priority))
        items.append({"id": r.id, "name": r.name, "priority": r.priority, "temp": got[0] if got else None,
                      "target": target_temp(r, now), "fresh": fresh, "w": w})
        if fresh:
            den += w
    for it in items:
        w = it.pop("w")
        if it["fresh"]:
            it["delta"] = round(it["temp"] - it["target"], 2)
            it["weight_pct"] = round(100 * w / den, 1)
            it["pull"] = round(w * (it["temp"] - it["target"]) / den, 2)  # this room's share of the weighted error
        else:
            it["delta"], it["weight_pct"], it["pull"] = None, 0.0, 0.0
    items.sort(key=lambda i: (-i["priority"], i["name"]))
    res = weighted_error(mine, readings, now, settings)
    driver = max((i for i in items if i["fresh"]), key=lambda i: abs(i["pull"]), default=None)
    return {
        "unit": unit.id, "mode": state.mode,
        "setpoint": cmd.setpoint if cmd else None, "reason": cmd.reason if cmd else "",
        "held_s": int((now - state.since).total_seconds()) if state.since else None,
        "error": None if res is None else round(res[0], 2),
        "avg": None if res is None else round(res[1], 2),
        "target": None if res is None else round(res[1] - res[0], 2),
        "unit_temp": None if unit_temp is None else round(unit_temp, 1),
        "offset": None if (res is None or unit_temp is None) else round(unit_temp - res[1], 1),
        "rooms": items, "driver": driver["id"] if driver and abs(driver["pull"]) > 0 else None,
    }
