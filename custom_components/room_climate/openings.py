"""Registers and doors: what the owner reports about them, how far to trust it, and what the registers do to the model.

There are no rules here about where a register or door should be: that is for the planner to say (mpc.advise_openings tries each
register's positions against its own plan's cost). A door has no model yet, so it gets no suggestion. Pure Python, no Home Assistant
imports.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

try:
    from . import core
except ImportError:  # offline
    import core  # type: ignore

# ---- what the registers do to the model ----
# A register's opening scales how much of the unit's air a room gets. These are priors (nobody has varied the registers on
# a measured day yet): a "closed" register still leaks some air, a half-open one passes most. The unit's total output is
# conserved, so closing one room's register sends its share to the rooms still open: that is the effect the owner wants.
REGISTER_GAIN = {"closed": 0.15, "partial": 0.6, "open": 1.0}
DOOR_LEVEL = {"closed": 0.0, "cracked": 0.15, "open": 0.6, "wide": 1.0}
DOOR_DEFAULT = 0.6  # what a room with no door entry counts as (rows logged before doors were tracked)


def register_gain(state) -> float:
    return REGISTER_GAIN.get(state, 1.0)  # an unset register counts as open


def confidence(item: dict, now: Optional[datetime], conf: tuple = (0.5, 0.95, 24.0)) -> float:
    """How likely the item really is in its reported state, 0..1. `conf` = (floor, fresh, tau_h): `fresh` right after the owner
    set or confirmed it, decaying to `floor` (the owner's own estimate: about half) over `tau_h` hours. Not set yet: 0."""
    levels = REGISTER_GAIN if item.get("kind") == "register" else DOOR_LEVEL
    if item.get("state") not in levels:
        return 0.0
    floor, fresh, tau_h = conf
    t = max(item.get("confirmed") or "", item.get("changed") or "")
    if not t or now is None:
        return floor
    try:
        age_h = max(0.0, (now - datetime.fromisoformat(t)).total_seconds() / 3600)
    except (TypeError, ValueError):
        return floor
    return floor + (fresh - floor) * math.exp(-age_h / max(tau_h, 1e-6))


def _blend(levels: dict, state, r: float) -> float:
    """The expected level: `r` on the reported state's, the rest spread evenly over the other states."""
    others = [v for k, v in levels.items() if k != state]
    mean_other = sum(others) / len(others)
    return r * levels[state] + (1 - r) * mean_other if state in levels else sum(levels.values()) / len(levels)


def register_multipliers(openings: list, room_unit: dict, model_rooms: dict, states: dict = None, now=None, conf=None) -> dict:
    """{room id: multiplier on its unit's effect there} from the registers' states (`states`: id -> state overrides, for
    trying alternatives). Per unit, the rooms that have registers share its output in proportion to their fitted gains times
    their register's opening (rooms with no register count as fully open), renormalized so the unit's total stays what it was
    with every register open."""
    states = states or {}
    by_room = {}
    for o in openings:
        if o.get("kind") == "register" and o.get("room"):
            if now is None:
                gval = register_gain(states.get(o["id"], o.get("state")))  # taken at face value
            elif o["id"] in states:  # a state being tried: as if the owner had just set it
                gval = _blend(REGISTER_GAIN, states[o["id"]], (conf or (0.5, 0.95, 24.0))[1])
            else:
                gval = _blend(REGISTER_GAIN, o.get("state"), confidence(o, now, conf or (0.5, 0.95, 24.0)))
            by_room.setdefault(o["room"], []).append(gval)
    out = {}
    for unit in {room_unit.get(r) for r in by_room} - {None}:
        base = {r: (model_rooms[r].heat.get(unit, 0.0) if r in model_rooms else 0.0)
                for r, u in room_unit.items() if u == unit and r in model_rooms}
        g = {r: (sum(by_room[r]) / len(by_room[r]) if r in by_room else 1.0) for r in base}
        total = sum(base.values())
        served = sum(base[r] * g[r] for r in base)
        if total <= 0 or served <= 0:
            continue
        for r in base:  # rooms with no register of their own get their share of the redistributed air too
            out[r] = g[r] * total / served
    return out


def door_level(openings: list, room_id: str, now=None, conf=None) -> float:
    """How open the room's door(s) are, 0..1 (mean of its doors; DOOR_DEFAULT if it has none or they are unset)."""
    mine = [o for o in openings if o.get("kind") == "door" and o.get("room") == room_id and o.get("state") in DOOR_LEVEL]
    if now is None:
        lv = [DOOR_LEVEL[o["state"]] for o in mine]
    else:
        lv = [_blend(DOOR_LEVEL, o["state"], confidence(o, now, conf or (0.5, 0.95, 24.0))) for o in mine]
    return sum(lv) / len(lv) if lv else DOOR_DEFAULT
