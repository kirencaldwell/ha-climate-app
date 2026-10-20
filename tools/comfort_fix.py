"""Bounded, reversible comfort fixes for the night-time review (tools/comfort_watch.py), and their undo.

Every change is recorded in reviews/fixes.jsonl with what it replaced, so it can be undone the next day:

    python3 tools/comfort_fix.py status                      rooms now: reading, target, priority in force, hold
    python3 tools/comfort_fix.py priority ROOM P --reason "..."   change the priority in force now (1-5)
    python3 tools/comfort_fix.py hold ROOM TEMP HOURS --reason "..."  hold a room's target (60-74 F, up to 10 h;
                                                            Room Climate ends the hold by itself)
    python3 tools/comfort_fix.py list                        every fix, newest last, with its id
    python3 tools/comfort_fix.py revert ID [ID ...] | --all  undo fixes (refused for a room changed since, unless
                                                            --force)

Only these two kinds of change exist: no settings, controls, code, restarts or unit commands. A priority change edits
the one entry of the room's priority schedule in force now (so it repeats on later days until reverted); a hold
expires on its own.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime

VENV_PY = os.path.expanduser("~/.local/share/room-climate-watch/venv/bin/python")
try:
    import websockets  # noqa: F401
except ImportError:  # the system python has no websockets: rerun under the watch's venv
    if os.path.exists(VENV_PY) and sys.executable != VENV_PY:
        os.execv(VENV_PY, [VENV_PY] + sys.argv)
    raise

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(os.path.dirname(HERE), "reviews", "fixes.jsonl")
URL = "ws://192.168.4.54:8123/api/websocket"
TEMP_MIN, TEMP_MAX, HOLD_MAX_H = 60.0, 74.0, 10.0


async def _call(msgs: list) -> list:
    tok = open(os.path.expanduser("~/.config/ha-token")).read().strip()
    out = []
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": tok}))
        if json.loads(await ws.recv()).get("type") != "auth_ok":
            raise SystemExit("Home Assistant refused the token")
        for i, m in enumerate(msgs, 1):
            await ws.send(json.dumps({"id": i, **m}))
            while True:
                d = json.loads(await ws.recv())
                if d.get("id") == i:
                    break
            if not d.get("success"):
                raise SystemExit(f"{m['type']} failed: {d.get('error')}")
            out.append(d.get("result"))
    return out


def call(*msgs):
    return asyncio.run(_call(list(msgs)))


def state() -> dict:
    return call({"type": "room_climate/get"})[0]


def room_of(st: dict, rid: str) -> dict:
    r = next((x for x in st["rooms"] if x["id"] == rid), None)
    if r is None:
        raise SystemExit(f"unknown room {rid}; rooms: {', '.join(x['id'] for x in st['rooms'])}")
    return r


def entry_in_force(sched: list, now: datetime):
    """Index of the priority-schedule entry in force now (today's latest start before now, else yesterday's last)."""
    day, hm = now.weekday(), now.strftime("%H:%M")
    today = [(e["start"], i) for i, e in enumerate(sched) if day in e["days"] and e["start"] <= hm]
    if today:
        return max(today)[1]
    yday = [(e["start"], i) for i, e in enumerate(sched) if (day - 1) % 7 in e["days"]]
    return max(yday)[1] if yday else None


def record(kind: str, rid: str, before: dict, after: dict, reason: str, note: str) -> dict:
    fx = {"id": uuid.uuid4().hex[:6], "at": datetime.now().astimezone().isoformat(timespec="seconds"), "kind": kind,
          "room": rid, "before": before, "after": after, "reason": reason, "note": note, "reverted": None}
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(fx) + "\n")
    return fx


def ledger() -> list:
    try:
        with open(LEDGER) as f:
            return [json.loads(x) for x in f if x.strip()]
    except OSError:
        return []


def save_ledger(fixes: list):
    with open(LEDGER, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in fixes)


def cmd_status(_):
    st = state()
    live = st.get("live", {})
    for r in st["rooms"]:
        v = live.get(r["id"], {})
        if v.get("temp") is None:
            continue
        hold = f" hold {r.get('hold_temp')} until {r.get('hold_until')}" if r.get("hold_until") else ""
        print(f"{r['id']:16s} {v['temp']:5.1f} F  target {v.get('target')}  priority {v.get('priority')}{hold}")
    h = st.get("house", {})
    print(f"house {h.get('mode')}: {h.get('reason')}  outdoor {h.get('outdoor')}")


def cmd_priority(a):
    if not 1 <= a.p <= 5:
        raise SystemExit("priority must be 1-5")
    st = state()
    r = room_of(st, a.room)
    now = datetime.now().astimezone()
    if r.get("priority_schedule"):
        sched = [dict(e) for e in r["priority_schedule"]]
        i = entry_in_force(sched, now)
        if i is None:
            raise SystemExit(f"{a.room}: no schedule entry in force now")
        old = sched[i]["priority"]
        sched[i]["priority"] = a.p
        call({"type": "room_climate/set_room", "room_id": a.room, "priority_schedule": sched})
        note = f"schedule entry from {sched[i]['start']} ({len(sched[i]['days'])} days/week): {old} -> {a.p}"
        before, after = {"priority_schedule": r["priority_schedule"]}, {"priority_schedule": sched}
    else:
        old = r.get("priority")
        call({"type": "room_climate/set_room", "room_id": a.room, "priority": a.p})
        note = f"priority {old} -> {a.p}"
        before, after = {"priority": old}, {"priority": a.p}
    fx = record("priority", a.room, before, after, a.reason, note)
    print(f"fix {fx['id']}: {a.room} {note}")


def cmd_hold(a):
    if not TEMP_MIN <= a.temp <= TEMP_MAX:
        raise SystemExit(f"hold temperature must be {TEMP_MIN}-{TEMP_MAX} F")
    if not 0.25 <= a.hours <= HOLD_MAX_H:
        raise SystemExit(f"hold must be 0.25-{HOLD_MAX_H} hours")
    st = state()
    r = room_of(st, a.room)
    before = {"hold_temp": r.get("hold_temp"), "hold_until": r.get("hold_until")}
    call({"type": "room_climate/set_room", "room_id": a.room, "hold_temp": a.temp, "hold_hours": a.hours})
    after = {"hold_temp": a.temp}
    fx = record("hold", a.room, before, after, a.reason, f"hold {a.temp} F for {a.hours} h (ends by itself)")
    print(f"fix {fx['id']}: {a.room} {fx['note']}")


def cmd_list(_):
    for fx in ledger():
        state_ = f"reverted {fx['reverted']}" if fx.get("reverted") else "in place"
        print(f"{fx['id']}  {fx['at'][:16]}  {fx['room']:16s} {fx['note']}  [{state_}]  why: {fx['reason']}")


def cmd_revert(a):
    fixes = ledger()
    todo = [fx for fx in fixes if not fx.get("reverted") and (a.all or fx["id"] in a.ids)]
    if not todo:
        raise SystemExit("nothing to revert")
    st = state()
    for fx in reversed(todo):  # newest first, so stacked fixes on one room unwind in order
        r = room_of(st, fx["room"])
        if fx["kind"] == "priority":
            key = next(iter(fx["after"]))
            if r.get(key) != fx["after"][key] and not a.force:
                print(f"{fx['id']}: {fx['room']} was changed since this fix; left alone (use --force to restore anyway)")
                continue
            call({"type": "room_climate/set_room", "room_id": fx["room"], key: fx["before"][key]})
        else:
            prev = fx["before"]
            call({"type": "room_climate/set_room", "room_id": fx["room"], "hold_temp": None})
            if prev.get("hold_until") and prev["hold_until"] > datetime.now().astimezone().isoformat():
                print(f"{fx['id']}: note: an earlier hold ({prev['hold_temp']} until {prev['hold_until']}) was not restored")
        fx["reverted"] = datetime.now().astimezone().isoformat(timespec="seconds")
        st = state()
        print(f"{fx['id']}: reverted ({fx['room']}: {fx['note']})")
    save_ledger(fixes)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("priority")
    p.add_argument("room"); p.add_argument("p", type=int); p.add_argument("--reason", required=True)
    p.set_defaults(fn=cmd_priority)
    h = sub.add_parser("hold")
    h.add_argument("room"); h.add_argument("temp", type=float); h.add_argument("hours", type=float)
    h.add_argument("--reason", required=True)
    h.set_defaults(fn=cmd_hold)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    rv = sub.add_parser("revert")
    rv.add_argument("ids", nargs="*"); rv.add_argument("--all", action="store_true")
    rv.add_argument("--force", action="store_true")
    rv.set_defaults(fn=cmd_revert)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
