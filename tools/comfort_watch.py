"""Comfort watch: when a room you care about at that hour leaves comfort, have Claude diagnose it for a later review.

Run every 5 minutes (launchd: tools/com.roomclimate.comfortwatch.plist). Each run reads the last readings from the HA
minute log over SSH and checks the triggers below. When one fires (at most once per night / morning window), it
writes a 12-hour summary of the logs and launches Claude Code headless to diagnose what happened. Its tools are
read-only (Read, Grep, Glob), plus, for the night trigger, tools/comfort_fix.py: a few bounded, recorded, reversible
changes (a self-expiring temperature hold, a room's priority). Its review is appended to reviews/comfort-reviews.md and
every fix to reviews/fixes.jsonl; undo them with `python3 tools/comfort_fix.py revert --all` (or by id).

    python3 tools/comfort_watch.py            check now (what launchd runs)
    python3 tools/comfort_watch.py --status   print each trigger's current state, never launch
    python3 tools/comfort_watch.py --force night|morning   run a diagnosis now, as if that trigger had fired

Triggers (calibrated on reported discomfort: "too hot" in Bedroom 3 at 3-5.5 F over target, "a little cold" on the
main floor at 2.5-4.5 F under):
  night    21:00-07:00  Bedroom 3 or Primary bedroom >= 2.5 F over target or >= 3 F under, for 20 min
  morning  06:30-09:00  kitchen, living room or dining area >= 3 F under target, for 30 min
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REVIEWS = os.path.join(REPO, "reviews")
STATE = os.path.expanduser("~/.local/state/room-climate-watch/state.json")
SSH = ["ssh", "-i", os.path.expanduser("~/.ssh/homeassistant"), "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
       "-o", "BatchMode=yes", "root@192.168.4.54"]
LOGDIR = "/config/room_climate_data"
CLAUDE = os.path.expanduser("~/.local/bin/claude")

TRIGGERS = {
    "night": {"rooms": ("bedroom_3", "primary_bedroom"), "from": "21:00", "to": "07:00", "over": 2.5, "under": 3.0,
              "minutes": 20, "fixes": True},
    "morning": {"rooms": ("kitchen", "living_room", "dining_area"), "from": "06:30", "to": "09:00", "over": None,
                "under": 3.0, "minutes": 30, "fixes": False},
}
MAX_FIXES = 2
NAMES = {"bedroom_3": "Bedroom 3", "primary_bedroom": "Primary bedroom", "kitchen": "Kitchen",
         "living_room": "Living room", "dining_area": "Dining area"}


def in_window(t: datetime, a: str, b: str) -> bool:
    hm = t.strftime("%H:%M")
    return a <= hm < b if a < b else (hm >= a or hm < b)


def window_start(t: datetime, a: str, b: str) -> datetime:
    """When the window containing t began (a night window that started yesterday evening counts from then)."""
    h, m = map(int, a.split(":"))
    s = t.replace(hour=h, minute=m, second=0, microsecond=0)
    return s if s <= t else s - timedelta(days=1)


def ssh(cmd: str) -> bytes:
    return subprocess.run(SSH + [cmd], check=True, capture_output=True, timeout=120).stdout


def recent_rows(n: int = 45) -> list:
    out = ssh(f"f=$(ls {LOGDIR}/*.jsonl | tail -1); tail -n {n} $f").decode()
    return [json.loads(x) for x in out.splitlines() if x.startswith("{")]


def rows_since(hours: float) -> list:
    """Minute-log rows of the last `hours` (today's file plus yesterday's when needed)."""
    now = datetime.now().astimezone()
    days = sorted({(now - timedelta(hours=hours)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")})
    rows = []
    for d in days:
        raw = ssh(f"cat {LOGDIR}/{d}.jsonl 2>/dev/null || cat {LOGDIR}/{d}.jsonl.gz 2>/dev/null")
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        rows += [json.loads(x) for x in raw.decode().splitlines() if x.startswith("{")]
    cut = (now - timedelta(hours=hours)).isoformat()
    return sorted((r for r in rows if r["t"] >= cut), key=lambda r: r["t"])


def miss(row: dict, rid: str):
    v = (row.get("rooms") or {}).get(rid) or {}
    if v.get("temp") is None or v.get("target") is None:
        return None
    return v["temp"] - v["target"]


def check(name: str, rows: list, now: datetime):
    """(fired, detail) for one trigger over the newest rows."""
    tr = TRIGGERS[name]
    if not in_window(now, tr["from"], tr["to"]):
        return False, "outside its hours"
    if any(r.get("exp") for r in rows[-tr["minutes"]:]):
        return False, "experiment running"
    recent = [r for r in rows if datetime.fromisoformat(r["t"]) >= now - timedelta(minutes=tr["minutes"])]
    if len(recent) < tr["minutes"] * 0.8:
        return False, f"only {len(recent)} minutes logged"
    hits = []
    for rid in tr["rooms"]:
        ms = [miss(r, rid) for r in recent]
        if any(x is None for x in ms):
            continue
        if tr["over"] is not None and min(ms) >= tr["over"]:
            hits.append(f"{NAMES[rid]} {ms[-1]:+.1f} F from target (over by {tr['over']}+ for {tr['minutes']} min)")
        if tr["under"] is not None and max(ms) <= -tr["under"]:
            hits.append(f"{NAMES[rid]} {ms[-1]:+.1f} F from target (under by {tr['under']}+ for {tr['minutes']} min)")
    return bool(hits), "; ".join(hits) or "within limits"


def summary(rows: list) -> str:
    """A compact table every 10 minutes: rooms (reading/target/priority), units, house mode, weather, outdoor unit."""
    rooms = sorted({r for row in rows for r, v in row.get("rooms", {}).items() if v.get("temp") is not None})
    lines = ["# Minute log summary (every 10 min)", "",
             "Rooms: reading/target/priority. Units: mode setpoint action fan. Power: outdoor unit draw (W).", ""]
    for row in rows:
        t = row["t"]
        if int(t[14:16]) % 10:
            continue
        rs = " ".join(f"{r}={v['temp']:.1f}/{v['target']:.0f}/p{v.get('prio')}" for r in rooms
                      for v in [row["rooms"].get(r) or {}] if v.get("temp") is not None and v.get("target") is not None)
        us = " ".join(f"{u}={x.get('state')} {x.get('sp')} {x.get('action')} {x.get('fan')}"
                      for u, x in (row.get("units") or {}).items())
        pw = max([v for k, v in (row.get("extra") or {}).items() if k.endswith("input_power")
                  and isinstance(v, (int, float))] or [None], key=lambda v: -1 if v is None else v)
        lines.append(f"{t[5:16]} house={row.get('house')} ({row.get('reason', '')[:70]}) out={row.get('outdoor')} "
                     f"sun={(row.get('sun') or {}).get('ghi')} power={pw} | {rs} | {us}")
    return "\n".join(lines) + "\n"


PROMPT = """You are reviewing the Room Climate controller for this house (repo in the working directory; read
CLAUDE.md-style notes in docs/ if present). A comfort trigger fired: {trigger} at {now}: {detail}.

The last 12 hours of the controller's minute log are summarised in {inputs} (read it first). The code is in
custom_components/room_climate/ (core.py is the everyday controller; model.py/fit.py/mpc.py/optimizer.py the learned
model and planner). Since the evening of 2026-10-21 the planner may be in control of the units: where the log's units
show control "model" it picks the house mode and each unit's target, and core.py turns those into setpoints; it hands a
unit back to core.py ("room_climate") on its own when the model stops predicting well, so check the log for that. You
have read-only tools and cannot change anything.

{fix_rules}
Write a short review for the owner to read tomorrow, in plain language:
1. What happened: which room, how far off, for how long, and what the house and units were doing.
2. Why, from the log: the cause you can show (quote times and numbers), and what you can't tell.
3. Proposed fixes, best first: each with what it would change, the trade-off, and how sure you are. Prefer fixes
   that improve the model or settings the owner controls (priorities, targets) over new special-case rules in the
   controller; say so if a fix would be a special case.
Known context: the units share one outdoor compressor (one house mode); Bedroom 3 picks up heat from the compressor
whenever the house heats (the "leak"); the owner prefers Bedroom 3 to track well at night even if the main floor is
cold. Keep it under 400 words. Output only the review, in Markdown, starting with a "### " heading line."""

FIX_RULES = """You may also fix it now, while the owner is asleep, with `python3 tools/comfort_fix.py` (the only command
you can run). Run `python3 tools/comfort_fix.py status` first to see the house now; it may have recovered already,
in which case change nothing. Rules:
- At most {max_fixes} fixes this run, only for the room(s) that tripped the trigger, and only if the log shows the fix
  addresses the cause. When unsure, don't: the controller keeps running and the owner reviews tomorrow.
- Prefer `hold ROOM TEMP HOURS` (it ends by itself; hold until about 07:00 at the latest) over `priority ROOM P`
  (which edits the room's schedule entry in force now and repeats on later days until reverted).
- The owner's preference: Bedroom 3 and the Primary bedroom track well at night; the main floor may run cold. Don't
  lower those two rooms' priorities at night or raise the main floor's.
- Every fix needs a --reason. In the review, list each fix with its id and what it changed, or say "No fixes made".
"""
NO_FIX_RULES = """Diagnose only: you can't change anything on this run.
"""


def _fixes_recorded() -> list:
    try:
        with open(os.path.join(REVIEWS, "fixes.jsonl")) as f:
            return [json.loads(x) for x in f if x.strip()]
    except OSError:
        return []


def diagnose(name: str, detail: str, now: datetime):
    os.makedirs(os.path.join(REVIEWS, "inputs"), exist_ok=True)
    stamp = now.strftime("%Y-%m-%d_%H%M")
    inputs = os.path.join(REVIEWS, "inputs", f"{stamp}_{name}.md")
    with open(inputs, "w") as f:
        f.write(summary(rows_since(12)))
    fixes = TRIGGERS[name]["fixes"]
    prompt = PROMPT.format(trigger=name, now=now.strftime("%Y-%m-%d %H:%M"), detail=detail,
                           inputs=os.path.relpath(inputs, REPO),
                           fix_rules=FIX_RULES.format(max_fixes=MAX_FIXES) if fixes else NO_FIX_RULES)
    allowed = ["Read", "Grep", "Glob"] + (["Bash(python3 tools/comfort_fix.py:*)"] if fixes else [])
    denied = ["Edit", "Write", "WebFetch", "WebSearch", "NotebookEdit"] + ([] if fixes else ["Bash"])
    n_before = len(_fixes_recorded())
    try:
        res = subprocess.run([CLAUDE, "-p", prompt, "--allowedTools", *allowed, "--disallowedTools", *denied,
                              "--max-turns", "30"],
                             cwd=REPO, capture_output=True, text=True, timeout=1800)
        text = res.stdout.strip() or f"(no review; claude exited {res.returncode}: {res.stderr.strip()[:500]})"
    except Exception as e:  # keep a record either way
        text = f"(no review: {type(e).__name__}: {e})"
    # What was really changed, from the fix tool's own record rather than the review's account of it
    made = _fixes_recorded()[n_before:]
    ledger = "".join(f"- `{x['id']}` {x['room']}: {x['note']} (why: {x['reason']}); undo: "
                     f"`python3 tools/comfort_fix.py revert {x['id']}`\n" for x in made) or "- none\n"
    with open(os.path.join(REVIEWS, "comfort-reviews.md"), "a") as f:
        f.write(f"\n---\n\n## {now.strftime('%a %Y-%m-%d %H:%M')}: {name} trigger\n\n{detail}\n\n"
                f"Log summary: `{os.path.relpath(inputs, REPO)}`\n\nFixes recorded:\n{ledger}\n{text}\n")


def load_state() -> dict:
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(st: dict):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(st, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", choices=list(TRIGGERS))
    a = ap.parse_args()
    now = datetime.now().astimezone()
    if a.force:
        diagnose(a.force, "forced run", now)
        return
    rows = recent_rows()
    st = load_state()
    for name, tr in TRIGGERS.items():
        fired, detail = check(name, rows, now)
        started = window_start(now, tr["from"], tr["to"]).isoformat()
        done = st.get(name) == started
        if a.status:
            print(f"{name:8s} {'FIRED' if fired else 'ok':5s} {detail}{' (already reviewed this window)' if done else ''}")
            continue
        if fired and not done:
            st[name] = started
            save_state(st)  # before the run: a slow or failed run still counts, so it can't repeat all night
            diagnose(name, detail, now)


if __name__ == "__main__":
    sys.exit(main())
