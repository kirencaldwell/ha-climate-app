"""Pure control logic (no Home Assistant imports). All temperatures are degrees F.

A *room* has a wifi sensor, a target (from its schedule) and a priority (fixed, or from its priority schedule).
A *unit* is a Mitsubishi climate entity serving one or more rooms. The
controller picks one house-wide mode (the units share an outdoor compressor) from
the rooms' current and predicted temperatures, then holds each unit at its rooms'
target. Setpoints and modes change rarely and gently so the inverter can modulate
instead of stopping and starting.
"""
from __future__ import annotations

import math
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
class PriorityEntry:
    """From `start` on these days the room has this priority (until the next entry). Steps, no glide."""
    days: list[int]
    start: str
    priority: int

    def minutes(self) -> int:
        h, m = self.start.split(":")
        return int(h) * 60 + int(m)


@dataclass
class Room:
    id: str
    name: str
    unit: Optional[str]  # None = monitor-only (sensor shown, never drives a unit)
    sensor: Optional[str] = None
    priority: int = 3  # used when no priority schedule entry applies
    default_temp: float = 68.0
    schedule: list[ScheduleEntry] = field(default_factory=list)
    priority_schedule: list[PriorityEntry] = field(default_factory=list)  # e.g. bedrooms low during the day
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
    # Comfort leans cool: the household would rather add a sweater than be too warm. So the band sits mostly below the
    # target, a miss above it costs `warm_weight` times a miss below, and heating aims `aim_offset` under the target.
    band_below: float = 1.5
    band_above: float = 1.0
    warm_weight: float = 2.0
    aim_offset: float = 0.5
    # House-wide hard limits: a room past these outweighs any priority.
    floor: float = 63.0  # below every schedule's night target, so a target at 65 doesn't sit on the hard limit
    ceiling: float = 72.0
    hard_weight: float = 20.0
    # Prediction: project each room's recent trend this far ahead.
    horizon_s: int = 2700
    history_s: int = 1800
    max_predicted_change: float = 3.0
    # House mode switching (heat <-> cool): only when the other side is clearly worse, and not too often.
    min_mode_s: int = 1200
    switch_confirm_s: int = 600  # the other mode must stay clearly better this long before switching
    switch_ratio: float = 1.5
    switch_margin: float = 0.5
    min_cost: float = 0.05  # demand below this counts as comfortable
    start_cost: float = 1.0  # demand needed to leave "off" (then confirmed for switch_confirm_s like any mode switch)
    max_offset: float = 6.0  # clamp on sensor-vs-unit-sensor compensation
    # Inverter efficiency: keep setpoints steady so the compressor modulates instead of stopping and starting.
    setpoint_min_change: float = 1.0  # ignore setpoint moves smaller than this (F)
    setpoint_min_interval_s: int = 900  # at most one setpoint change per unit this often (mode changes excepted)
    # Outer loop (see drive): set each unit's setpoint relative to its room so it pushes gently, instead of sending
    # the target and letting the unit blast at full output until it overshoots. Off = the old fixed-setpoint behaviour.
    drive: bool = True
    drive_tau_min: float = 40.0  # how fast the loop closes on target; each unit's gain follows from its learned effect
    drive_kp: float = 1.0  # gain (F of setpoint push per F of error) when the model has nothing learned for a unit
    drive_kp_min: float = 0.3
    drive_kp_max: float = 1.0  # 1 = never push past the target on P alone: gentler than today even for a weak unit
    drive_ti_min: float = 90.0  # integral time: removes the steady-state shortfall a pure P loop would leave
    drive_i_max: float = 2.0  # cap on the integral's share of the push (anti-windup)
    drive_lead_min: float = 10.0  # act on where the room will be this far ahead (anticipates the coast after a stop);
    #                               a unit's learned lag replaces it
    drive_max_push: float = 2.0  # setpoint at most this far past the room reading: limits the blast of hot/cold air
    # Per unit, for rooms that respond very differently: "unit:F,unit:F". A unit not listed gets a push limit from its
    # learned effect (so it moves its rooms at most drive_max_rate F/h), else drive_max_push.
    drive_max_push_units: str = "air_handler:4,minisplit_2:1"
    drive_max_rate: float = 4.0
    drive_push_cap: float = 4.0  # never more than this, set or learned
    drive_stop_push: float = 2.0  # setpoint this far behind the reading to make the unit stop (it idles ~1 C past it)
    drive_off: float = 0.5  # stop the unit once the room is predicted this far past target
    drive_restart: float = 0.6  # once stopped, restart only when the room is predicted this far short of target
    drive_slope_s: int = 600  # the room's rate of change from this much history
    drive_min_interval_s: int = 300  # at most one setpoint change per unit this often (starts and stops excepted)
    drive_min_change: float = 0.7  # ignore setpoint moves smaller than this (most of a 0.5 C step)
    # Fan speed, the loop's second output (units listed here only; others keep whatever fan they're set to). A gentle
    # push runs the fan low: less capacity at the compressor's lowest speed, so a strong unit in a small room runs
    # longer and softer instead of in short blasts, and less draft near it. A hard push steps the fan up; a room far
    # from target gets the unit's own auto fan.
    fan_units: str = "minisplit_1,minisplit_2"  # comma-separated unit ids
    fan_levels: str = "quiet,low,medium"  # slowest first, for pushes up to 1/3, 2/3 and the rest of drive_max_push
    fan_far: float = 2.0  # room predicted this far short of target: auto fan
    fan_min_interval_s: int = 600  # at most one fan change per unit this often (going to auto excepted)
    # Reading older than this is ignored. Long, because Bluetooth sensors (e.g. Govee) only update when the value
    # changes; a sensor that really drops out goes unavailable instead.
    stale_s: int = 7200
    # A sensor that goes unavailable (Bluetooth sensors drop out for seconds to minutes) keeps its last reading this
    # long before the room counts as having no reading.
    dropout_grace_s: int = 900
    # Free cooling/heating (off by default): when it's at least this much colder outside than a room, a room merely
    # above its comfort band is left to cool off on its own, so only the hard ceiling asks for cooling (mirrored for
    # heat on hot days). Off because a schedule drop to a cold target should actively cool, not wait for the weather.
    free_delta: Optional[float] = None
    outdoor_entity: str = ""  # weather.* or sensor.*; empty = the first weather entity
    allow_cool: bool = True
    allow_heat: bool = True
    deadband: float = 1.0  # legacy (card gauge width); kept so stored settings still load
    stop_margin: float = 0.0  # legacy
    min_run_s: int = 300  # legacy
    min_off_s: int = 300  # legacy
    mode_switch_s: int = 900  # legacy


def round_cool(x: float) -> float:
    """Nearest whole degree, ties going down: with setpoints aimed half a degree under whole-degree targets, a tie
    rounding to even would land cool one degree and warm the next (66.5 -> 66 but 67.5 -> 68)."""
    return float(math.floor(x + 0.5 - 1e-9))


def settings_from(d) -> Settings:
    """Settings from stored values, ignoring names this version doesn't know (a removed setting must not crash)."""
    known = set(Settings.__dataclass_fields__)
    return Settings(**{k: v for k, v in (d or {}).items() if k in known})


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
    raw: Optional[float] = None  # setpoint before rounding, so stabilize can tell real moves from rounding jitter
    target: Optional[float] = None  # what the unit's rooms should reach (room scale, aim included), for drive
    room: Optional[float] = None  # the rooms' priority-weighted reading now
    offset: float = 0.0  # the unit's own reading minus `room`
    fan: Optional[str] = None  # fan mode to set (None: leave the unit's fan as it is)


@dataclass
class HouseState:
    """The one mode all units share (they have a common outdoor compressor)."""
    mode: str = "off"  # "off" until something first needs conditioning, then "heat" | "cool"
    since: Optional[datetime] = None
    pending: Optional[str] = None  # a mode switch that is being confirmed
    pending_since: Optional[datetime] = None


def elapsed(later: datetime, earlier: datetime) -> float:
    """Real seconds between two times. Subtracting two local times in the same zone gives wall-clock time, which is
    an hour off across a daylight-saving change (and negative during the repeated fall-back hour), so aware times
    go through their UTC timestamps. Naive times (sim, tests) have no DST to get wrong."""
    if later.tzinfo is None or earlier.tzinfo is None:
        return (later - earlier).total_seconds()
    return later.timestamp() - earlier.timestamp()


def _hold_active(room: Room, now: datetime) -> bool:
    if room.hold_temp is None:
        return False
    return room.hold_until is None or now < room.hold_until


SCHEDULE_RAMP_S = 3600  # glide into each schedule change over this long instead of stepping


def _entry_times(room: Room, now: datetime, days, entries=None, value=lambda e: e.temp):
    for back in days:
        day = now - timedelta(days=back)
        for e in (room.schedule if entries is None else entries):
            if day.weekday() in e.days:
                yield day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=e.minutes()), value(e)


def scheduled_temp(room: Room, now: datetime) -> float:
    """Latest schedule entry at or before `now` (looking back up to a week), gliding linearly toward the next
    entry over the hour before it. A heat pump recovers far more efficiently from a gentle ramp than a step."""
    past = [x for x in _entry_times(room, now, range(8)) if x[0] <= now]
    current = max(past)[1] if past else room.default_temp
    upcoming = [x for x in _entry_times(room, now, (0, -1)) if now < x[0] <= now + timedelta(seconds=SCHEDULE_RAMP_S)]
    if not upcoming:
        return current
    at, nxt = min(upcoming)
    frac = 1 - (at - now).total_seconds() / SCHEDULE_RAMP_S
    return round(current + (nxt - current) * frac, 1)


def target_temp(room: Room, now: datetime) -> float:
    return room.hold_temp if _hold_active(room, now) else scheduled_temp(room, now)


def room_priority(room: Room, now: datetime) -> int:
    """Priority in force now: the latest priority-schedule entry at or before `now` (looking back up to a week),
    else the room's fixed priority. Clamped to MIN_PRIORITY..MAX_PRIORITY."""
    past = [x for x in _entry_times(room, now, range(8), room.priority_schedule, lambda e: e.priority) if x[0] <= now]
    p = max(past)[1] if past else room.priority
    return max(MIN_PRIORITY, min(MAX_PRIORITY, int(p)))


def weighted_error(rooms: list[Room], readings: dict[str, tuple[float, datetime]],
                   now: datetime, settings: Settings):
    """Priority-weighted (reading - target). Returns (error, avg_reading) or None."""
    num = den = rsum = 0.0
    for r in rooms:
        got = readings.get(r.id)
        if not got or elapsed(now, got[1]) > settings.stale_s:
            continue
        w = room_priority(r, now)
        num += w * (got[0] - target_temp(r, now))
        rsum += w * got[0]
        den += w
    if not den:
        return None
    return num / den, rsum / den


def stored_setpoint_c(setpoint_f: float) -> float:
    """The setpoint a Mitsubishi unit actually stores for `setpoint_f`, in C: it keeps 0.5 C steps, so 64 F is stored
    as 18.0 C and reads back as 64.4 F. Compare read-backs against this, not against the F we asked for."""
    return round((setpoint_f - 32) / 1.8 * 2) / 2


def hold_reading(fresh: Optional[tuple[float, datetime]], last: Optional[tuple[float, datetime, datetime]],
                 now: datetime, settings: Settings = Settings()):
    """Bridge a sensor dropout. `fresh` is this tick's (value, time) or None if unavailable; `last` is the previous
    result of this function's second value: (value, time, when it was last seen). Returns (reading to use, new last)."""
    if fresh is not None:
        return fresh, (fresh[0], fresh[1], now)
    if last is not None and elapsed(now, last[2]) <= settings.dropout_grace_s:
        return (last[0], last[1]), last
    return None, last


def predict(history: list[tuple[datetime, float]], now: datetime, settings: Settings = Settings()) -> Optional[float]:
    """Project a room's temperature `horizon_s` ahead from a least-squares trend over the last `history_s`."""
    pts = [(t, v) for t, v in history if elapsed(now, t) <= settings.history_s]
    if len(pts) < 3 or elapsed(pts[-1][0], pts[0][0]) < 600:
        return None
    xs = [elapsed(t, pts[0][0]) for t, _ in pts]
    ys = [v for _, v in pts]
    if len({round(v, 2) for v in ys}) < 3:
        # One step on a coarse sensor (a Mitsubishi unit reports in 0.5 C = 0.9 F steps) or a reading flickering
        # between two values is not a trend; extrapolating it predicts a phantom 2 F swing.
        return ys[-1]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    change = max(-settings.max_predicted_change, min(settings.max_predicted_change, slope * settings.horizon_s))
    return ys[-1] + change


def room_costs(room: Room, current: float, predicted: Optional[float], now: datetime,
               settings: Settings = Settings(), outdoor: Optional[float] = None) -> tuple[float, float]:
    """(cold, hot) discomfort for one room. A room counts as cold if it is cold now *or* will be, same for hot.

    With `free_delta` set and an outdoor temperature, comfort demand the weather will settle by itself is dropped;
    the hard floor/ceiling always count."""
    t = target_temp(room, now)
    w = room_priority(room, now)
    lo = current if predicted is None else min(current, predicted)
    hi = current if predicted is None else max(current, predicted)
    free = settings.free_delta is not None and outdoor is not None
    free_cool = free and outdoor <= current - settings.free_delta
    free_heat = free and outdoor >= current + settings.free_delta
    cold = (0.0 if free_heat else w * max(0.0, (t - settings.band_below) - lo) ** 2) \
        + settings.hard_weight * max(0.0, settings.floor - lo) ** 2
    hot = (0.0 if free_cool else settings.warm_weight * w * max(0.0, hi - (t + settings.band_above)) ** 2) \
        + settings.hard_weight * max(0.0, hi - settings.ceiling) ** 2
    return cold, hot


def choose_mode(cold: float, hot: float, house: HouseState, now: datetime,
                settings: Settings = Settings()) -> tuple[HouseState, str]:
    """Pick the house mode from total cold vs hot demand. Never goes back to off once running.

    Leaving "off" (first start, or after a restart) takes demand of `start_cost`, held for `switch_confirm_s`, so a
    sensor ticking a hair past a limit can't start the compressor.

    Each heat<->cool switch stops the compressor and reverses the refrigerant flow, so a switch needs the other
    side to be clearly worse *continuously* for `switch_confirm_s`, after at least `min_mode_s` in the mode.
    """
    can = {"heat": settings.allow_heat, "cool": settings.allow_cool}
    if house.mode not in ("heat", "cool"):
        want = "heat" if cold >= hot else "cool"
        demand = max(cold, hot)
        if demand < settings.start_cost or not can[want]:
            return HouseState("off"), "house comfortable"
        what = f"{'cold' if want == 'heat' else 'hot'} demand {demand:.2f}"
        if house.pending != want:
            return HouseState("off", None, want, now), f"{want} wanted ({what}); confirming"
        waited = elapsed(now, house.pending_since)
        if waited < settings.switch_confirm_s:
            return HouseState("off", None, want, house.pending_since), \
                f"{want} wanted ({what}); confirming {int(waited // 60)}/{settings.switch_confirm_s // 60} min"
        return HouseState(want, now), what
    mine, theirs = (cold, hot) if house.mode == "heat" else (hot, cold)
    other = "cool" if house.mode == "heat" else "heat"
    stay = HouseState(house.mode, house.since)
    if can[other] and theirs >= settings.min_cost and theirs > mine * settings.switch_ratio + settings.switch_margin:
        held = elapsed(now, house.since) if house.since else 1e9
        if held < settings.min_mode_s:
            return stay, f"{other} wanted ({theirs:.2f} vs {mine:.2f}); holding {house.mode} min time"
        if house.pending != other:
            return HouseState(house.mode, house.since, other, now), f"{other} wanted ({theirs:.2f} vs {mine:.2f}); confirming"
        waited = elapsed(now, house.pending_since)
        if waited < settings.switch_confirm_s:
            return HouseState(house.mode, house.since, other, house.pending_since), \
                f"{other} wanted ({theirs:.2f} vs {mine:.2f}); confirming {int(waited // 60)}/{settings.switch_confirm_s // 60} min"
        return HouseState(other, now), f"switching: {'hot' if other == 'cool' else 'cold'} demand {theirs:.2f} vs {mine:.2f}"
    if max(cold, hot) < settings.min_cost:
        return stay, f"{house.mode} mode, idle: nothing needed"
    return stay, f"{house.mode} (cold {cold:.2f}, hot {hot:.2f})"


def unit_command(unit: Unit, rooms: list[Room], readings: dict[str, tuple[float, datetime]],
                 predicted: dict[str, Optional[float]], mode: str, unit_temp: Optional[float],
                 state: UnitState, now: datetime, settings: Settings = Settings()) -> Command:
    """Setpoint for one unit in the house mode: its rooms' target, held steady.

    An inverter heat pump is most efficient left at one setpoint so it can slow down as the room nears target,
    rather than being flipped between "run" and "idle" setpoints (which makes the compressor stop and restart).
    Rooms already past target simply make the unit idle on its own.
    """
    if mode not in ("heat", "cool"):
        return Command("off", None, "house comfortable")
    res = weighted_error(rooms, readings, now, settings)
    if res is None:
        # No reading (sensors out past the dropout grace, or none yet after a restart): leave the unit's setpoint as
        # it is rather than jumping to a limit; stabilize keeps our last setpoint if we have one.
        return Command(mode, None, "no fresh sensors; holding setpoint")
    err, avg = res
    # Aim a little cool while heating (less heat). Not while cooling: aiming under the target there parks rooms near
    # the cool edge of the band, and the evening cool-down then tips the house into heat. The band's tighter warm
    # edge and the doubled cost of running warm already make cooling start sooner.
    target = avg - err - (settings.aim_offset if mode == "heat" else 0.0)
    offset = 0.0
    if unit_temp is not None:  # make the unit's own thermostat chase the *room* temperature
        offset = max(-settings.max_offset, min(settings.max_offset, unit_temp - avg))
    raw = max(unit.min_setpoint, min(unit.max_setpoint, target + offset))
    needs = avg < target if mode == "heat" else avg > target
    # Whole degrees in the reason: it is logged whenever it changes, and tenths tick over through every ramp.
    why = (("heating" if mode == "heat" else "cooling") + f" toward {target:.0f}°") if needs else "at target; idling"
    return Command(mode, round_cool(raw), why, needs, raw, target, avg, offset)


def stabilize(cmd: Command, prev: Optional[tuple[Command, datetime]], now: datetime,
              settings: Settings = Settings()) -> tuple[Command, bool]:
    """Hold the previous setpoint unless the new one differs by at least `setpoint_min_change` and the last change
    was at least `setpoint_min_interval_s` ago. Mode changes always go through. Returns (command, changed).

    "Small" compares the unrounded setpoint, so a value hovering near x.5 doesn't alternate between two whole
    degrees. A small hold is the normal steady state and isn't annotated; only a real move held back by the
    interval says so."""
    if cmd.setpoint is None:
        if prev is not None and prev[0].mode == cmd.mode and prev[0].setpoint is not None:
            return Command(cmd.mode, prev[0].setpoint, cmd.reason, cmd.active, prev[0].setpoint), False
        return cmd, True
    if prev is None:
        return cmd, True
    pcmd, at = prev
    if pcmd.mode != cmd.mode or pcmd.setpoint is None:
        return cmd, True
    want = cmd.raw if cmd.raw is not None else cmd.setpoint
    if abs(want - pcmd.setpoint) < settings.setpoint_min_change:
        return Command(cmd.mode, pcmd.setpoint, cmd.reason, cmd.active, cmd.raw), False
    if elapsed(now, at) < settings.setpoint_min_interval_s:
        return Command(cmd.mode, pcmd.setpoint, f"{cmd.reason} (holding {pcmd.setpoint:g}°: recently changed)",
                       cmd.active, cmd.raw), False
    return cmd, True


@dataclass
class DriveState:
    """One unit's outer loop: whether it's pushing, its integral, and the last setpoint it sent."""
    mode: str = "off"
    on: bool = False
    i: float = 0.0  # F of push held by the integral
    at: Optional[datetime] = None  # last update
    sp: Optional[float] = None  # last setpoint sent (F, on the unit's 0.5 C grid)
    sent_at: Optional[datetime] = None
    miss: float = 0.0  # last predicted miss (F; > 0 the room still needs this mode's help)
    fan: Optional[str] = None  # last fan mode chosen
    fan_at: Optional[datetime] = None


def slope(history: list[tuple[datetime, float]], now: datetime, window_s: int) -> Optional[float]:
    """Least-squares rate of change (F/h) over the last `window_s`, or None with too little to go on."""
    pts = [(t, v) for t, v in history if elapsed(now, t) <= window_s]
    if len(pts) < 3 or elapsed(pts[-1][0], pts[0][0]) < window_s / 2:
        return None
    xs = [elapsed(t, pts[0][0]) / 3600 for t, _ in pts]
    ys = [v for _, v in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def drive_gain(effect: Optional[float], settings: Settings = Settings(), k1: float = 0.5) -> float:
    """Outer-loop gain for a unit whose full output moves its room `effect` F/h (learned by the model). The inverter
    turns a setpoint push into output at about k1 per F, so the room closes on target at effect*k1*kp per hour; pick
    kp for a time constant of drive_tau_min. A powerful unit in a small room gets a small gain, and the reverse."""
    if not effect or effect <= 0:
        return settings.drive_kp
    kp = 1.0 / (effect * k1 * settings.drive_tau_min / 60)
    return max(settings.drive_kp_min, min(settings.drive_kp_max, kp))


def to_grid(setpoint_f: float) -> float:
    """`setpoint_f` on the unit's 0.5 C grid, in F."""
    return round(stored_setpoint_c(setpoint_f) * 1.8 + 32, 1)


def max_push_for(unit_id: str, effect: Optional[float], settings: Settings = Settings(), k0: float = 0.3,
                 k1: float = 0.5) -> float:
    """How far past the room a unit's setpoint may go: set per unit (drive_max_push_units), else from its learned
    effect on its rooms (F/h at full output; the inverter's output is about k0 + k1 x push), so that it moves them at
    most drive_max_rate F/h, else drive_max_push. At least one 0.5 C step, at most drive_push_cap."""
    for part in settings.drive_max_push_units.split(","):
        k, _, v = part.partition(":")
        if k.strip() == unit_id and v.strip():
            try:
                return max(0.9, min(settings.drive_push_cap, float(v)))
            except ValueError:
                break
    if effect and effect > 0:
        return max(0.9, min(settings.drive_push_cap, (settings.drive_max_rate / effect - k0) / k1))
    return settings.drive_max_push


def drive_law(st: Optional[DriveState], mode: str, room: float, target: float, rate: Optional[float], kp: float,
              now: datetime, settings: Settings = Settings(), lead_min: Optional[float] = None,
              max_push: Optional[float] = None) -> tuple[float, DriveState]:
    """The outer loop: how far past the room reading to put the setpoint (the push, F, positive = in the mode's
    direction), from the room's predicted miss `lead_min` ahead (target minus prediction, signed so that > 0 means
    the room still needs this mode's help). The lead is the unit's learned lag when there is one: with a first-order
    lag, rate x lag is how far the room goes on after the unit stops. Else drive_lead_min.

        push = kp * miss + integral      while running (clamped to -one 0.5 C step .. max_push; at least +one step
                                         while the room is more than drive_restart short, or the unit wouldn't start)
        push = -stop_push                once the room is on track to pass target by drive_off: stop it

    While running, a small or slightly negative push lets the inverter's own loop settle at low output; the integral
    supplies the steady push a cold night needs. An inverter can't go below its minimum output, though, and only
    idles once its reading is about 1 C past its setpoint, so to stop it before an overshoot takes a definite push the
    other way. Restarting needs `drive_restart` of predicted miss, so it doesn't chatter. The integral builds only
    while below the cap, bleeds off while stopped, and resets on a mode change. Returns (push, new state); the
    state's sp/sent_at are the caller's."""
    s = 1.0 if mode == "heat" else -1.0
    fresh = st is None or st.mode != mode
    dt_h = 0.0 if fresh or st.at is None else max(0.0, min(0.25, elapsed(now, st.at) / 3600))
    r = max(-12.0, min(12.0, rate or 0.0))
    lead = settings.drive_lead_min if lead_min is None else lead_min
    miss = s * (target - (room + r * lead / 60))
    i = 0.0 if fresh else st.i
    if fresh:
        on = miss > 0
    elif st.on:
        on = miss > -settings.drive_off
    else:
        on = miss > settings.drive_restart
    ti_h = max(1e-6, settings.drive_ti_min / 60)
    step = 0.9
    if on:
        p = kp * miss
        cap = settings.drive_max_push if max_push is None else max_push
        if p + i < cap:  # anti-windup: don't build while saturated
            i = max(0.0, min(settings.drive_i_max, i + kp * miss * dt_h / ti_h))
        # Clearly short: at least one step, or a push that rounds to the room's own reading never starts the unit
        # (Bedroom 3 sat idle at 65.3 with its setpoint 65.3 for an hour). Near target the unit may settle lower.
        low = step if miss > settings.drive_restart else -step
        push = max(low, min(cap, p + i))
    else:
        i = max(0.0, i + kp * min(0.0, miss) * dt_h / ti_h)  # bleeds off while past target
        push = -settings.drive_stop_push
    keep = DriveState(mode, on, i, now, None if fresh else st.sp, None if fresh else st.sent_at, miss,
                      None if st is None else st.fan, None if st is None else st.fan_at)
    return push, keep


def fan_for(push: float, miss: float, on: bool, settings: Settings = Settings(), max_push: Optional[float] = None) -> Optional[str]:
    """Fan mode for a push (None: no opinion, i.e. stopped: leave it)."""
    if not on:
        return None
    if miss >= settings.fan_far:
        return "auto"
    levels = [x.strip() for x in settings.fan_levels.split(",") if x.strip()]
    if not levels:
        return None
    frac = max(0.0, push) / max(1e-6, settings.drive_max_push if max_push is None else max_push)
    return levels[min(len(levels) - 1, int(frac * len(levels)))]


def choose_fan(st: DriveState, push: float, now: datetime, settings: Settings = Settings(),
               max_push: Optional[float] = None) -> Optional[str]:
    """The fan mode to have now, holding a change for fan_min_interval_s (except to auto, which goes at once).
    Updates st.fan/st.fan_at. None: leave the fan alone."""
    want = fan_for(push, st.miss, st.on, settings, max_push)
    if want is None or want == st.fan:
        return st.fan if want is not None else None
    if st.fan is not None and want != "auto" and st.fan_at is not None \
            and elapsed(now, st.fan_at) < settings.fan_min_interval_s:
        return st.fan
    st.fan, st.fan_at = want, now
    return want


def drive(cmd: Command, st: Optional[DriveState], rate: Optional[float], kp: float, unit: Unit, now: datetime,
          settings: Settings = Settings(), lead_min: Optional[float] = None,
          max_push: Optional[float] = None) -> tuple[Command, DriveState, bool]:
    """Turn a command's target into the setpoint to send this tick -> (command, new state, changed).

    Setpoints sit on the unit's 0.5 C grid. A new setpoint goes out on a start or stop, on a mode change, or when it
    has moved at least drive_min_change and drive_min_interval_s has passed; otherwise the last one is held."""
    if cmd.mode not in ("heat", "cool") or cmd.target is None or cmd.room is None:
        keep = st.sp if st is not None and st.mode == cmd.mode else None
        new = DriveState(cmd.mode, False, 0.0, now, keep, st.sent_at if st and keep is not None else None)
        return Command(cmd.mode, keep, cmd.reason, False, keep, cmd.target, cmd.room, cmd.offset), new, \
            st is None or st.mode != cmd.mode
    push, new = drive_law(st, cmd.mode, cmd.room, cmd.target, rate, kp, now, settings, lead_min, max_push)
    s = 1.0 if cmd.mode == "heat" else -1.0
    raw = max(unit.min_setpoint, min(unit.max_setpoint, cmd.room + cmd.offset + s * push))
    sp = to_grid(raw)
    toggled = st is None or st.mode != cmd.mode or st.on != new.on
    prev = new.sp
    if prev is None or toggled:
        send = True
    elif new.on:
        send = abs(raw - prev) >= settings.drive_min_change and sp != prev and (
            new.sent_at is None or elapsed(now, new.sent_at) >= settings.drive_min_interval_s)
    else:  # stopped: the stop setpoint stays put unless the room has come back close enough to restart the unit
        send = s * (prev - (cmd.room + cmd.offset)) > -settings.drive_stop_push / 2
    if send:
        new.sp, new.sent_at, changed = sp, now, sp != prev or toggled
    else:
        sp, changed = prev, False
    if new.on:
        why = f"{'heating' if cmd.mode == 'heat' else 'cooling'} toward {cmd.target:.0f}°"
    elif s * (cmd.target - cmd.room) > 0:
        why = f"easing off: on track for {cmd.target:.0f}°"
    else:
        why = "at target; idling"
    fan = None
    if unit.id in {x.strip() for x in settings.fan_units.split(",")}:
        fan = choose_fan(new, push, now, settings, max_push)
    return Command(cmd.mode, sp, why, new.on, raw, cmd.target, cmd.room, cmd.offset, fan), new, changed


MANUAL_MODES = ("dry", "fan_only")


def control_step(units: list[Unit], rooms: list[Room], readings: dict[str, tuple[float, datetime]],
                 predicted: dict[str, Optional[float]], unit_temps: dict[str, Optional[float]],
                 states: dict[str, UnitState], house: HouseState, now: datetime,
                 settings: Settings = Settings(), manual: Optional[dict[str, str]] = None,
                 busy: Optional[set[str]] = None, outdoor: Optional[float] = None):
    """One control tick for the whole house -> (commands by unit id, new HouseState, info dict).

    All units always share one mode. `manual` holds the current mode of each unit under thermostat control (not
    driven by us); commands are still computed for those units, as what Room Climate would do. A unit in dry or fan
    takes over: the controller keeps every other unit off so the compressor never gets conflicting modes. A unit in
    heat or cool on its thermostat sets the house mode for the rest, for the same reason. `busy` lists units that are
    actually heating/cooling right now (by us or by hand): their rooms' trends come from the unit, not the room.
    """
    blocking = {u: m for u, m in (manual or {}).items() if m in MANUAL_MODES}
    if blocking:
        m = next(iter(blocking.values()))
        cmds = {u.id: (Command(blocking[u.id], None, f"manual {blocking[u.id]}; holding") if u.id in blocking
                       else Command("off", None, f"blocked: manual {m} on another unit")) for u in units}
        return cmds, house, {"mode": m, "reason": f"manual {m} in use", "cold": None, "hot": None, "rooms": {}}
    forced = next((m for m in (manual or {}).values() if m in ("heat", "cool")), None)

    cold = hot = 0.0
    per_room: dict[str, tuple[float, float]] = {}
    # Only rooms on units we drive pick heat vs cool: a unit left on its thermostat isn't ours to serve, and its room
    # may read from that unit's own (biased) sensor. With no unit under our control every room counts, so what Room
    # Climate *would* do stays meaningful.
    ours = {u.id for u in units} - set(manual or {})
    voters = ours or {u.id for u in units}
    unit_entity = {u.id: u.entity for u in units}
    zones: dict[str, list] = {}  # unit id -> [cold, hot] of its voting rooms
    for r in rooms:
        if r.unit is None or r.unit not in voters:
            continue  # monitor-only rooms (and rooms of thermostat-run units) don't steer the house
        got = readings.get(r.id)
        if not got or elapsed(now, got[1]) > settings.stale_s:
            continue
        # A room whose unit is heating/cooling it (by us or by hand) trends because of the unit, and that stops at
        # its setpoint, so the trend says nothing about the opposite need (it would flip cool->heat as cooling
        # works). Use the reading alone.
        driven = states.get(r.unit, UnitState()).active or r.unit in (busy or ())
        c, h = room_costs(r, got[0], None if driven else predicted.get(r.id), now, settings, outdoor)
        per_room[r.id] = (round(c, 2), round(h, 2))
        if r.sensor is not None and r.sensor == unit_entity.get(r.unit):
            # The room reads its own unit's sensor, which sits in that unit's airflow and swings with it (Bedroom 3:
            # 55-78 F in an hour as its unit went heat -> cool). It would drive the house into a heat/cool limit cycle,
            # so it follows the house mode and just holds its target until it has a sensor of its own.
            continue
        zone = zones.setdefault(r.unit, [0.0, 0.0])
        zone[0] += c
        zone[1] += h
    # A unit can't heat one of its rooms and cool another, so a zone's rooms vote together, by their net demand. Rooms
    # sharing the air handler at different targets (a kitchen set to 65 at night among rooms at 67) would otherwise
    # vote for opposite modes in turn: heating for the zone overheats the kitchen -> cool -> the zone is cold -> heat.
    # Netted, the zone asks for what most of it (by priority) needs, and the kitchen's miss is a trade-off, not a flip.
    for zc, zh in zones.values():
        if zc > zh:
            cold += zc - zh
        else:
            hot += zh - zc
    new_house, reason = choose_mode(cold, hot, house, now, settings)
    if forced and new_house.mode != forced:
        new_house = HouseState(forced, now if house.mode != forced else house.since)
        reason = f"following {forced} set on a thermostat ({reason})"
    mode = new_house.mode
    cmds = {}
    for u in units:
        mine = [r for r in rooms if r.unit == u.id]
        cmds[u.id] = unit_command(u, mine, readings, predicted, mode, unit_temps.get(u.id),
                                  states.get(u.id, UnitState()), now, settings)
    return cmds, new_house, {"mode": mode, "reason": reason, "cold": round(cold, 2), "hot": round(hot, 2), "rooms": per_room,
                             "outdoor": outdoor}


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
        fresh = bool(got) and elapsed(now, got[1]) <= settings.stale_s
        w = room_priority(r, now)
        items.append({"id": r.id, "name": r.name, "priority": w, "temp": got[0] if got else None,
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
        "held_s": int(elapsed(now, state.since)) if state.since else None,
        "error": None if res is None else round(res[0], 2),
        "avg": None if res is None else round(res[1], 2),
        "target": None if res is None else round(res[1] - res[0], 2),
        "unit_temp": None if unit_temp is None else round(unit_temp, 1),
        "offset": None if (res is None or unit_temp is None) else round(unit_temp - res[1], 1),
        "rooms": items, "driver": driver["id"] if driver and abs(driver["pull"]) > 0 else None,
    }
