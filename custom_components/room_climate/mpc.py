"""The model-based controller in "would do" mode, next to today's controller, plus live tracking of how well the
model predicts. Pure Python (no Home Assistant imports): the integration feeds it every tick and runs the heavy parts
(planning, predicting) in an executor job.

Two things run on their own clocks:
- plan (every 5 min): optimizer.plan picks the mode and setpoints the model-based controller would use, with the same
  safety rules as today's controller (start from off only after the confirmation time, setpoints through
  core.stabilize). Nothing is sent; the decision is shown in the card and written to the minute log.
- track (every 10 min): start a prediction from the current readings and advance it minute by minute with what the
  units, the weather and the sun *actually* do, scoring it against the readings 30/60/120 min later, next to the
  "nothing changes" baseline. Replaying the real inputs (rather than assuming today's settings hold) keeps controller
  decisions out of the score: it measures the physics only. Cheap enough to run every tick.
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta

try:
    from . import core
    from . import model as hm
    from . import optimizer
except ImportError:  # offline
    import core  # type: ignore
    import model as hm  # type: ignore
    import optimizer  # type: ignore

HORIZONS_MIN = (30, 60, 120)


class ShadowPlanner:
    def __init__(self, ps: optimizer.PlanSettings = None, plan_every_s: int = 300, track_every_s: int = 600,
                 keep_h: int = 24):
        self.model: hm.HouseModel = None
        self.ps = ps or optimizer.PlanSettings()
        self.plan_every, self.track_every = timedelta(seconds=plan_every_s), timedelta(seconds=track_every_s)
        self.keep = timedelta(hours=keep_h)
        self.house = core.HouseState()
        self.sent: dict = {}  # unit id -> (Command, when): what it would have sent
        self.decision: dict = {}  # unit id -> {"mode", "setpoint"}
        self.plan: optimizer.Plan = None
        self.plan_at: datetime = None
        self.plan_ms: float = None
        self.error: str = None
        self.next_plan: datetime = None
        self.next_track: datetime = None
        self.records: deque = deque()  # open predictions: {"t0", "t", "start", "temps", "running", "done"}
        self.errors: deque = deque()  # (t_eval, room, horizon_min, model_err, nochange_err)
        self.points: deque = deque()  # (t_eval, room, horizon_min, predicted F): for charts
        self.readings: dict = {}  # room -> deque[(t, F)]

    def set_model(self, m: hm.HouseModel):
        self.model = m
        self.records.clear()  # predictions from the old model would be scored against the new one
        self.errors.clear()
        self.points.clear()
        self.next_plan = self.next_track = None

    # --- every tick (cheap) ---
    def observe(self, now: datetime, temps: dict, actual: dict = None, outdoor=None, sun: float = 0.0):
        """Remember readings (one per minute per room); start, advance and score predictions. actual: unit id ->
        {"mode", "setpoint" (F), "running"} as the units really are now."""
        for rid, v in temps.items():
            d = self.readings.setdefault(rid, deque())
            if not d or (now - d[-1][0]).total_seconds() >= 55:
                d.append((now, v))
            while d and now - d[0][0] > self.keep:
                d.popleft()
        if self.model is None or actual is None:
            return
        try:
            self._advance(now, temps, actual, outdoor, sun)
            if self.next_track is None or now >= self.next_track:
                self.next_track = now + self.track_every
                start = {r: temps[r] for r in self.model.rooms if temps.get(r) is not None}
                if start:
                    self.records.append({"t0": now, "t": now, "start": start, "temps": dict(start), "done": set(),
                                         "running": {u: bool(a.get("running")) for u, a in actual.items()}})
        except Exception as e:  # never let the model take the integration down; show it instead
            self.error = f"{type(e).__name__}: {e}"
        while self.records and now - self.records[0]["t0"] > timedelta(minutes=max(HORIZONS_MIN) + 10):
            self.records.popleft()
        while self.errors and now - self.errors[0][0] > self.keep:
            self.errors.popleft()

    def _advance(self, now, temps, actual, outdoor, sun):
        modes = {u: (a.get("mode") if a.get("mode") in ("heat", "cool") else "off") for u, a in actual.items()}
        sps = {u: a.get("setpoint") for u, a in actual.items()}
        for rec in self.records:
            if len(rec["done"]) == len(HORIZONS_MIN):
                continue
            dt = (now - rec["t"]).total_seconds() / 3600
            n = max(1, int(dt * 60 / 2 + 0.999))  # steps of at most 2 min (ticks can be sparse)
            for _ in range(n if dt > 0 else 0):
                rec["temps"], rec["running"], _ = hm.step(self.model, rec["temps"], outdoor, sun, modes, sps,
                                                          rec["running"], dt / n)
            rec["t"] = now
            age = (now - rec["t0"]).total_seconds() / 60
            for h in HORIZONS_MIN:
                if h in rec["done"] or age < h:
                    continue
                rec["done"].add(h)
                for rid, pred in rec["temps"].items():
                    real = temps.get(rid)
                    if real is None or rid not in rec["start"]:
                        continue
                    self.errors.append((now, rid, h, pred - real, rec["start"][rid] - real))
                    self.points.append((now, rid, h, round(pred, 2)))
        while self.points and now - self.points[0][0] > timedelta(hours=8):
            self.points.popleft()

    def due(self, now: datetime) -> bool:
        return self.model is not None and (self.next_plan is None or now >= self.next_plan)

    # --- heavy part (executor) ---
    def run(self, now: datetime, rooms: list, units: list, temps: dict, actual: dict, outdoor, sun_at,
            settings: core.Settings, ignore=()):
        """Plan, if due. actual: unit id -> {"mode", "setpoint" (F), "running"}; sun_at(datetime) -> 0..1."""
        if self.model is None or not (self.next_plan is None or now >= self.next_plan):
            return
        self.next_plan = now + self.plan_every
        try:
            self._plan(now, rooms, units, temps, actual, outdoor, sun_at, settings, ignore)
            self.error = None
        except Exception as e:  # never let the model take the integration down; show it instead
            self.error = f"{type(e).__name__}: {e}"

    def _plan(self, now, rooms, units, temps, actual, outdoor, sun_at, settings, ignore):
        ps = self.ps
        n = int(round(ps.horizon_h * 60 / ps.step_min))
        times = [now + timedelta(minutes=ps.step_min * k) for k in range(n)]
        running = {u: bool(a.get("running")) for u, a in actual.items()}
        now_sp = {u.id: (self.sent[u.id][0].setpoint if u.id in self.sent else actual.get(u.id, {}).get("setpoint"))
                  for u in units}
        t0 = time.perf_counter()
        p = optimizer.plan(self.model, rooms, units, temps, now, self.house, now_sp, running, [outdoor] * n,
                           [sun_at(t) for t in times], settings, ps, ignore=set(ignore))
        setpoints = p.setpoints
        if self.house.mode not in ("heat", "cool"):
            # Starting from off: the "demand" is how much better the best mode is than staying off; same start
            # threshold and confirmation time as today's controller.
            want = p.mode if p.mode != "off" else "heat"
            benefit = 0.0 if p.mode == "off" else p.by_mode["off"]["cost"] - p.cost
            self.house, _ = core.choose_mode(benefit if want == "heat" else 0.0, benefit if want == "cool" else 0.0,
                                             self.house, now, settings)
            setpoints = p.by_mode.get(self.house.mode, {}).get("setpoints", {})
        elif p.mode != self.house.mode:
            self.house = core.HouseState(p.mode, now)
        self.decision = {}
        for u in units:
            sp = setpoints.get(u.id)
            if self.house.mode not in ("heat", "cool"):
                self.decision[u.id] = {"mode": "off", "setpoint": None}
                continue
            cmd = core.Command(self.house.mode, sp, p.reason, True, sp)
            cmd, changed = core.stabilize(cmd, self.sent.get(u.id), now, settings)
            if changed and cmd.setpoint is not None:
                self.sent[u.id] = (cmd, now)
            self.decision[u.id] = {"mode": cmd.mode, "setpoint": cmd.setpoint}
        self.plan, self.plan_at = p, now
        self.plan_ms = round((time.perf_counter() - t0) * 1000, 1)

    # --- reporting ---
    def log_fields(self) -> dict:
        if self.plan is None:
            return None
        return {"house": self.house.mode, "units": self.decision, "cost": round(self.plan.cost, 3),
                "costs": {k: v["cost"] for k, v in self.plan.by_mode.items()}}

    def accuracy(self) -> dict:
        out = {}
        for _, rid, h, e, b in self.errors:
            out.setdefault(rid, {}).setdefault(str(h), []).append((e, b))
        res = {}
        for rid, by in out.items():
            res[rid] = {}
            for h, pairs in by.items():
                n = len(pairs)
                res[rid][h] = {"n": n, "rmse": round((sum(e * e for e, _ in pairs) / n) ** 0.5, 2),
                               "bias": round(sum(e for e, _ in pairs) / n, 2),
                               "base_rmse": round((sum(b * b for _, b in pairs) / n) ** 0.5, 2)}
        return res

    def history(self, now: datetime, hours: float = 6) -> dict:
        """Per room: readings (every ~5 min); for moments in the past, what the model said 1 h and 2 h before, replaying
        what the units actually did; and the plan's trajectory ahead. [iso time, F] pairs."""
        since = now - timedelta(hours=hours)
        out = {}
        for rid, d in self.readings.items():
            pts, last = [], None
            for t, v in d:
                if t >= since and (last is None or (t - last).total_seconds() >= 290):
                    pts.append([t.isoformat(), v])
                    last = t
            out[rid] = {"actual": pts, "pred60": [], "pred120": []}
        for t, rid, h, v in self.points:
            key = {60: "pred60", 120: "pred120"}.get(h)
            if key and t >= since and rid in out:
                out[rid][key].append([t.isoformat(), v])
        if self.plan is not None:
            for rid, traj in self.plan.predicted.items():
                if rid in out:
                    out[rid]["plan"] = [[(self.plan_at + timedelta(minutes=self.ps.step_min * (k + 1))).isoformat(), v]
                                        for k, v in enumerate(traj)]
        return out

    def debug(self, now: datetime) -> dict:
        p = self.plan
        return {
            "error": self.error,
            "house": self.house.mode,
            "since": self.house.since.isoformat() if self.house.since else None,
            "pending": self.house.pending,
            "plan": None if p is None else {
                "at": self.plan_at.isoformat(), "ms": self.plan_ms, "mode": p.mode, "reason": p.reason,
                "cost": round(p.cost, 3), "by_mode": p.by_mode, "setpoints": p.setpoints, "later": p.later,
                "step_min": self.ps.step_min},
            "decision": self.decision,
            "accuracy": self.accuracy(),
            "history": self.history(now),
            "horizons": list(HORIZONS_MIN),
        }
