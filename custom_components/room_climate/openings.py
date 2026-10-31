"""Suggested register and door positions, from where the rooms are against their targets right now.

A first, transparent rule (the model will replace it once the log has register and door changes to learn from): the air
handler's air should go to the rooms that need it. While the house is heating (cooling), a register in a room that is over
(under) its target by more than the room's allowed error is suggested partly or fully closed, and one in a room that is
short of its target open. A door to a room that is far over is suggested cracked (less of the hall's air flows in), to a
room far short, open. Pure Python, no Home Assistant imports.
"""
from __future__ import annotations

from typing import Optional

try:
    from . import core
except ImportError:  # offline
    import core  # type: ignore

REGISTER_PARTIAL_Z = 0.7  # error / allowed error beyond which a register is suggested half closed
REGISTER_CLOSED_Z = 2.0   # ...and closed
DOOR_Z = 1.5              # a door is suggested cracked / open beyond this


def recommend(openings: list, rooms: list, live: dict, mode: Optional[str], settings: core.Settings = None) -> dict:
    """{opening id: {"rec": state, "why": text, "match": bool}} for each opening whose room has a reading. `rooms`: the
    config's room dicts, `live`: room id -> {"temp", "target", "priority"}; nothing is suggested while the house is off."""
    s = settings or core.Settings()
    if mode not in ("heat", "cool"):
        return {}
    sign = 1.0 if mode == "heat" else -1.0
    unit_of = {r["id"]: r.get("unit") for r in rooms}
    out = {}
    for o in openings:
        rid = o.get("room")
        lv = (live or {}).get(rid) or {}
        if rid is None or not unit_of.get(rid) or lv.get("temp") is None or lv.get("target") is None:
            continue
        delta = lv["temp"] - lv["target"]
        z = sign * delta / core.allowable_error(lv.get("priority") or 1, s.allowable_ref_f)  # > 0: past its goal
        name = next((r.get("name") for r in rooms if r["id"] == rid), rid)
        head = f"{name} is {abs(delta):.1f}° {'over' if delta > 0 else 'under'} its {lv['target']:g}° target"
        if o["kind"] == "register":
            if z >= REGISTER_CLOSED_Z:
                rec, why = "closed", f"{head}: close it so the air goes to the rooms still short"
            elif z >= REGISTER_PARTIAL_Z:
                rec, why = "partial", f"{head}: half close it"
            elif z <= -REGISTER_PARTIAL_Z:
                rec, why = "open", f"{head}: keep it fully open"
            else:
                rec, why = "open", f"{name} is near its target: leave it open"
        else:
            cur = o.get("state")
            if z >= DOOR_Z:  # never suggest opening a door further for a room that is already too warm
                rec, why = (cur, f"{head}: keep the door {cur}") if cur in ("closed", "cracked") else \
                    ("cracked", f"{head}: crack the door, less of the hall's air flows in")
            elif z <= -DOOR_Z:
                rec, why = (cur, f"{head}: keep the door {'wide ' if cur == 'wide' else ''}open") if cur in ("open", "wide") else \
                    ("open", f"{head}: open the door so the hall's air reaches it")
            else:
                continue  # no reason to change a door
        out[o["id"]] = {"rec": rec, "why": why, "match": o.get("state") == rec, "src": "rule"}
    return out


# ---- what the registers do to the model ----
# A register's opening scales how much of the unit's air a room gets. These are priors (nobody has varied the registers on
# a measured day yet): a "closed" register still leaks some air, a half-open one passes most. The unit's total output is
# conserved, so closing one room's register sends its share to the rooms still open: that is the effect the owner wants.
REGISTER_GAIN = {"closed": 0.15, "partial": 0.6, "open": 1.0}
DOOR_LEVEL = {"closed": 0.0, "cracked": 0.15, "open": 0.6, "wide": 1.0}
DOOR_DEFAULT = 0.6  # what a room with no door entry counts as (rows logged before doors were tracked)


def register_gain(state) -> float:
    return REGISTER_GAIN.get(state, 1.0)  # an unset register counts as open


def register_multipliers(openings: list, room_unit: dict, model_rooms: dict, states: dict = None) -> dict:
    """{room id: multiplier on its unit's effect there} from the registers' states (`states`: id -> state overrides, for
    trying alternatives). Per unit, the rooms that have registers share its output in proportion to their fitted gains times
    their register's opening (rooms with no register count as fully open), renormalized so the unit's total stays what it was
    with every register open."""
    states = states or {}
    by_room = {}
    for o in openings:
        if o.get("kind") == "register" and o.get("room"):
            by_room.setdefault(o["room"], []).append(register_gain(states.get(o["id"], o.get("state"))))
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


def door_level(openings: list, room_id: str) -> float:
    """How open the room's door(s) are, 0..1 (mean of its doors; DOOR_DEFAULT if it has none or they are unset)."""
    lv = [DOOR_LEVEL[o["state"]] for o in openings if o.get("kind") == "door" and o.get("room") == room_id and o.get("state") in DOOR_LEVEL]
    return sum(lv) / len(lv) if lv else DOOR_DEFAULT
