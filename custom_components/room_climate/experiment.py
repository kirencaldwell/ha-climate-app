"""Experiment days: while nobody's home, run the units in a planned sequence the model can learn more from than from
everyday running, where the units mostly run together and in step with the sun.

A plan is a list of blocks over the time allotted:
- "float": every unit idles, so the house drifts on its own (heat loss, sun by wall, rooms trading heat).
- "solo": one unit runs at a steady push while the others idle (its effect on every room, and its lag). A unit whose
  fan we control sweeps its fan speeds in 15-minute steps during its run (each speed's share of its output), quiet to
  auto, then the other way round next time, so a speed isn't always at the same point in a run.
- "coast": everything idles after a solo (how long its effect lingers).
- "restore": the last hour is normal control, so the house is back on schedule when the time is up.

One mode for the whole experiment (the units share a compressor, and mode switches are what we least want): the
house's current one, else heat when it's cool outside. The house-wide hard floor still holds: a unit heats a room that
has fallen below it. The ceiling holds with CEILING_EXTRA more room (the house is empty on an experiment day, and a unit
that stops at the everyday ceiling teaches the model nothing for the rest of its block): a unit stops pushing a room
past it (heat), or past the floor (cool).

Units whose room warms when another unit's compressor runs (the leak) go first: they are the ones that hit the ceiling
from the others' runs, so later their own block would be idle.
Pure Python, no Home Assistant imports.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

try:
    from . import core
except ImportError:  # offline
    import core  # type: ignore

FLOAT_MIN, SOLO_MIN, FAN_STEP_MIN, COAST_MIN, RESTORE_MIN = 90, 45, 15, 30, 60
FAN_CYCLE = ("quiet", "low", "medium", "auto")
PUSH = 2.0  # F past the room for a solo unit (the outer loop's largest push)
MIN_H, MAX_H = 2, 24
CEILING_EXTRA = 3.0  # F above the everyday ceiling a unit may push a room to during an experiment


def pick_mode(house_mode: str, outdoor: Optional[float]) -> str:
    if house_mode in ("heat", "cool"):
        return house_mode
    return "cool" if outdoor is not None and outdoor >= 65 else "heat"


def order(unit_ids: list, leaky=()) -> list:
    """Unit ids in the order their solo runs go: units with a leak first (their room is warmed by the others' runs),
    else as given."""
    return sorted(unit_ids, key=lambda u: u not in leaky)


def plan(start: datetime, hours: float, unit_ids: list, fan_units=(), mode: str = "heat") -> dict:
    """{"start", "end", "mode", "blocks": [{"start", "end", "kind", "unit", "fan"}]} with ISO times."""
    hours = max(MIN_H, min(MAX_H, hours))
    end = start + timedelta(hours=hours)
    restore = end - timedelta(minutes=RESTORE_MIN)
    blocks, t, rnd = [], start, 0

    def add(kind, minutes, unit=None, fan=None):
        nonlocal t
        stop = min(restore, t + timedelta(minutes=minutes))
        if stop - t >= timedelta(minutes=15):
            blocks.append({"start": t.isoformat(), "end": stop.isoformat(), "kind": kind, "unit": unit, "fan": fan})
        t = stop

    add("float", FLOAT_MIN)
    while t < restore - timedelta(minutes=15):
        for uid in unit_ids:
            if t >= restore - timedelta(minutes=15):
                break
            if uid in fan_units:
                for fan in (FAN_CYCLE if rnd % 2 == 0 else FAN_CYCLE[::-1]):
                    add("solo", FAN_STEP_MIN, uid, fan)
            else:
                add("solo", SOLO_MIN, uid)
            add("coast", COAST_MIN)
        rnd += 1
        add("float", FLOAT_MIN // 2)
    blocks.append({"start": restore.isoformat(), "end": end.isoformat(), "kind": "restore", "unit": None, "fan": None})
    return {"start": start.isoformat(), "end": end.isoformat(), "mode": mode, "blocks": blocks}


def current(exp: dict, now: datetime) -> Optional[dict]:
    """The block in force now (None: before the start or after the end)."""
    for b in exp.get("blocks", []):
        if datetime.fromisoformat(b["start"]) <= now < datetime.fromisoformat(b["end"]):
            return b
    return None


def describe(b: Optional[dict], names: dict = None) -> str:
    if b is None:
        return "finished"
    nm = (names or {}).get(b.get("unit"), b.get("unit"))
    return {"float": "all units idle (the house drifts)",
            "coast": "all units idle (after a run)",
            "restore": "back to normal control before the time is up",
            "solo": f"{nm} only" + (f", fan {b['fan']}" if b.get("fan") else "")}[b["kind"]]


def commands(b: dict, mode: str, cmds: dict, unit_rooms: dict, readings: dict, settings: core.Settings) -> dict:
    """Unit id -> core.Command for an experiment block (not "restore": that's normal control). `cmds` are today's
    controller's commands this tick (their room/offset give each unit's rooms' reading), `unit_rooms` unit id -> its
    rooms' ids, `readings` room id -> F."""
    s = 1.0 if mode == "heat" else -1.0
    out = {}
    for uid, c in cmds.items():
        if c.room is None:
            out[uid] = core.Command(mode, None, "experiment: no reading; holding")
            continue
        temps = [readings[r] for r in unit_rooms.get(uid, []) if readings.get(r) is not None]
        solo = b["kind"] == "solo" and b.get("unit") == uid
        why = f"experiment: {describe(b)}"
        if mode == "heat" and temps and min(temps) < settings.floor:
            solo, why = True, "experiment: a room is below the floor; heating"  # never let the empty house get cold
        elif solo and temps and (max(temps) >= settings.ceiling + CEILING_EXTRA - 0.5 if mode == "heat" else min(temps) <= settings.floor + 0.5):
            solo, why = False, "experiment: at the house limit; idling"
        push = PUSH if solo else -settings.drive_stop_push
        sp = core.to_grid(c.room + c.offset + s * push)
        out[uid] = core.Command(mode, sp, why, solo, sp, c.target, c.room, c.offset, b.get("fan") if solo else None)
    return out
