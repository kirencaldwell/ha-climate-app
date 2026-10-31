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
            if z >= DOOR_Z:
                rec, why = "cracked", f"{head}: crack the door, less of the hall's air flows in"
            elif z <= -DOOR_Z:
                rec, why = "open", f"{head}: open the door so the hall's air reaches it"
            else:
                continue  # no reason to change a door
        out[o["id"]] = {"rec": rec, "why": why, "match": o.get("state") == rec}
    return out
