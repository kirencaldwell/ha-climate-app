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
    from . import joint
    from . import mlmodel
    from . import model as hm
    from . import observer
    from . import optimizer
except ImportError:  # offline
    import core  # type: ignore
    import joint  # type: ignore
    import mlmodel  # type: ignore
    import model as hm  # type: ignore
    import observer  # type: ignore
    import optimizer  # type: ignore

HORIZONS_MIN = (30, 60, 120)
READY_TOL_F = 0.2   # F a room's score may trail 'nothing changes' by and still count as beating it (noise: sensors
#                     read to 0.1 F, and a room that hardly moves is hard to beat); absolute, so it can't excuse a room that moves


def along(now: datetime, current, points):
    """f(t): a forecast as a function of time. Starts at `current` (the observation now), then follows `points`
    ([(datetime, value)], e.g. hourly forecast values) by straight lines between them, holding the last one. Points at
    or before `now` are ignored, so a forecast hour that's already started doesn't override what's observed. With no
    future points (no forecast), it's `current` throughout. None values are skipped."""
    pts = sorted((t, v) for t, v in (points or []) if v is not None and t > now)
    if current is None and pts:
        current = pts[0][1]
    knots = [(now, current)] + pts

    def f(t):
        if current is None:
            return None
        if t <= knots[0][0] or len(knots) == 1:
            return knots[0][1]
        for (ta, va), (tb, vb) in zip(knots, knots[1:]):
            if t <= tb:
                frac = (t - ta).total_seconds() / max(1.0, (tb - ta).total_seconds())
                return va + (vb - va) * frac
        return knots[-1][1]
    return f


def as_h(sun) -> float:
    """The overhead sun index (0..1) of a sun vector (or a bare 0..1 value)."""
    return float(hm.as_sun(sun).get("h", 0.0)) if sun is not None else 0.0


def _seq(x, times):
    """Weather over the plan: a function of time (a forecast) or a constant (current conditions held)."""
    return [x(t) for t in times] if callable(x) else [x] * len(times)


class ShadowPlanner:
    def __init__(self, ps: optimizer.PlanSettings = None, plan_every_s: int = 300, track_every_s: int = 600,
                 keep_h: int = 24, confirm_plans: int = 1, correct: bool = False):
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
        self.projection: dict = None  # {"at", "temps": {room: [[iso, F]]}}: 2 h ahead if the units keep their settings
        self.next_track: datetime = None
        self.records: deque = deque()  # open predictions: {"t0", "t", "start", "temps", "running", "done"}
        self.errors: deque = deque()  # (t_eval, room, horizon_min, model_err, nochange_err)
        self.points: deque = deque()  # (t_eval, room, horizon_min, predicted F): for charts
        self.readings: dict = {}  # room -> deque[(t, F)]
        self.lagged: dict = {}  # unit id -> effective output now (model.step), followed from what the units really do
        self.masses: dict = {}  # room id -> its mass temperature now (rooms with a mass state), followed from readings
        self.comp: dict = {"h": 0.0}  # hours the shared compressor has been running (any unit running): for the start-up pulse
        self.targets: dict = {}  # unit id -> target last planned (with the outer loop)
        # Feedback on the model (observer.py): learned corrections the plans use. `correct` False: plans use the raw model.
        # Off by default (held back for now): plans use the model as fitted and switch modes as soon as another wins.
        self.correct = correct
        self.observer = observer.Observer(offsets_on=correct, effect_on=correct)
        self.confirm_plans = confirm_plans  # plans in a row that must prefer the other mode before the house switches
        self._switch_votes = 0
        # The ML correction (mlmodel.py): feature rows logged as predictions complete (out-of-sample: made by whatever
        # model was live), the daily-trained corrector, and its scores next to the physics model's.
        self.ml_rows: list = []  # [iso time, room, horizon, real change, features], until drained to the log
        self.corrector = None
        # The joint run/fan planner (joint.py), shadow only: what it would do if it ran the house, every 15 min, with its own
        # hypothetical house mode. Slow (pure-Python search), so it backs off if a plan takes too long.
        self.joint = joint.JointPlanner()
        self.joint_every = timedelta(minutes=15)
        self.next_joint: datetime = None
        self.jhouse = core.HouseState()
        self.jplan = None
        self.jplan_at: datetime = None
        self.jplan_ms: float = None
        self.jerror: str = None
        self.jflips: deque = deque()  # when the joint planner would have switched heat <-> cool
        self.jlast_mode: str = None
        self.ml_errors: deque = deque()  # (t_eval, room, horizon_min, corrected err, physics err)

    def set_model(self, m: hm.HouseModel):
        """Use a (re)fitted model. Accuracy history is kept: a nightly refit changes the model a little, has passed its
        own held-out check, and wiping the scores would knock any unit under model control back out every night."""
        self.model = m
        self.next_plan = None

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
            self._follow_mass(now, temps)
            self._follow_lag(now, temps, actual)
            self._follow_comp(now, actual)
            self._advance(now, temps, actual, outdoor, sun)
            if self.correct:
                self.observer.update(self.model, now, temps, actual, outdoor, sun, self.lagged, self.masses, self.comp)
            if self.next_track is None or now >= self.next_track:
                self.next_track = now + self.track_every
                start = {r: temps[r] for r in self.model.rooms if temps.get(r) is not None}
                if start:
                    self.records.append({"t0": now, "t": now, "start": start, "temps": dict(start), "done": set(),
                                         "running": {u: bool(a.get("running")) for u, a in actual.items()},
                                         "lagged": dict(self.lagged), "masses": dict(self.masses),
                                         "comp": dict(self.comp), "ctx": self._ml_context(now, start, outdoor, sun),
                                         "acc": {"y": {}, "on": 0.0, "heat": 0.0, "o": 0.0, "s": 0.0, "min": 0.0}})
        except Exception as e:  # never let the model take the integration down; show it instead
            self.error = f"{type(e).__name__}: {e}"
        while self.records and now - self.records[0]["t0"] > timedelta(minutes=max(HORIZONS_MIN) + 10):
            self.records.popleft()
        while self.errors and now - self.errors[0][0] > self.keep:
            self.errors.popleft()
        while self.ml_errors and now - self.ml_errors[0][0] > self.keep:
            self.ml_errors.popleft()

    def _follow_mass(self, now, temps):
        """Each room's mass temperature now, driven by its readings (model.mass_step): the model's second state, which
        can't be measured, follows from the room's history. Starts at the reading; after a restart it takes a few of
        its time constants to settle (until then a room's mass sits nearer its air than it really is)."""
        last = getattr(self, "_mass_at", None)
        dt_h = 0.0 if last is None else max(0.0, min(0.5, (now - last).total_seconds() / 3600))
        self._mass_at = now
        for rid, r in self.model.rooms.items():
            if not r.mass_k:
                self.masses.pop(rid, None)
                continue
            t = temps.get(rid)
            if t is not None:
                self.masses[rid] = hm.mass_step(r, self.masses.get(rid), t, dt_h)

    def _ml_context(self, now, start, outdoor, sun):
        """What the ML features need at the origin of a prediction: each room's readings 10 and 30 min ago, its two
        strongest-coupled neighbours' difference from it, the outdoor temperature, the sun, the time of day, how long the
        compressor has run."""
        def back(rid, minutes):
            d = self.readings.get(rid)
            if not d:
                return None
            target = now - timedelta(minutes=minutes)
            val = None
            for t, v in d:
                if t > target:
                    break
                val = v
            return val
        rooms = {}
        for rid, T0 in start.items():
            t10, t30 = back(rid, 10), back(rid, 30)
            if t10 is None or t30 is None:
                continue
            nb = [start[j] - T0 for j in mlmodel.neighbours(self.model, rid) if j in start]
            rooms[rid] = (t10, t30, (nb + [0.0, 0.0])[:2])
        return {"rooms": rooms, "out0": outdoor if outdoor is not None else None, "sun0": as_h(sun),
                "hour": now.hour + now.minute / 60, "t_on0": self.comp.get("h", 0.0) * 60}

    def _ml_log(self, rec, rid, h, pred, real, now):
        """A prediction just completed: log its features and the real change, and score the ML-corrected prediction."""
        ctx, acc = rec.get("ctx"), rec.get("acc")
        if not ctx or not acc or rid not in ctx["rooms"] or acc["min"] <= 0 or ctx["out0"] is None:
            return
        T0 = rec["start"][rid]
        t10, t30, nb = ctx["rooms"][rid]
        own = mlmodel.own_unit(self.model, rid)
        ym = {u: v / acc["min"] for u, v in acc["y"].items()}
        f = mlmodel.make_features(pred - T0, T0, ctx["out0"], acc["o"] / acc["min"], t10, t30, nb[0], nb[1],
                                  ym.get(own, 0.0) if own else 0.0, sum(v for u, v in ym.items() if u != own),
                                  acc["on"] / acc["min"], ctx["t_on0"], acc["heat"] / acc["min"], acc["s"] / acc["min"],
                                  ctx["sun0"], ctx["hour"])
        f = [round(x, 4) for x in f]
        self.ml_rows.append([now.isoformat(), rid, h, round(real - T0, 4), f])
        if self.corrector is not None:
            corr = pred + self.corrector.correct(rid, h, f)
            self.ml_errors.append((now, rid, h, corr - real, pred - real))
        if len(self.ml_rows) > 50000:  # nobody is draining them: keep the newest
            del self.ml_rows[:10000]

    def drain_ml_rows(self) -> list:
        out, self.ml_rows = self.ml_rows, []
        return out

    def set_corrector(self, c):
        self.corrector = c

    def _follow_comp(self, now, actual):
        """How long the shared compressor has been running: any unit running counts."""
        last = getattr(self, "_comp_at", None)
        dt_h = 0.0 if last is None else max(0.0, min(0.5, (now - last).total_seconds() / 3600))
        self._comp_at = now
        self.comp["h"] = self.comp.get("h", 0.0) + dt_h if any(a.get("running") for a in actual.values()) else 0.0

    def _follow_lag(self, now, temps, actual):
        """Each unit's effective output now: its estimated output (running or not, and how hard from its setpoint gap)
        through its learned lag, so a plan made just after a unit stops knows the room will coast on."""
        last = getattr(self, "_lag_at", None)
        dt_h = 0.0 if last is None else max(0.0, min(0.5, (now - last).total_seconds() / 3600))
        self._lag_at = now
        for uid, u in self.model.units.items():
            a = actual.get(uid) or {}
            mode = a.get("mode")
            y = 0.0
            if a.get("running") and mode in ("heat", "cool"):
                rd, sp = hm.unit_reading(self.model, uid, temps), a.get("setpoint")
                y = 1.0
                if rd is not None and sp is not None:
                    err = (sp - rd) if mode == "heat" else (rd - sp)
                    y = max(u.ymin, min(1.0, u.k0 + u.k1 * err))
                y = (y if mode == "heat" else -y) * hm.fan_factor(u, a.get("fan"))
            self.lagged[uid] = hm.lag(u, self.lagged.get(uid, y), y, dt_h)

    def _advance(self, now, temps, actual, outdoor, sun):
        modes = {u: (a.get("mode") if a.get("mode") in ("heat", "cool") else "off") for u, a in actual.items()}
        sps = {u: a.get("setpoint") for u, a in actual.items()}
        # Whether each compressor ran is replayed too, not re-simulated: with units that cycle, a hair of temperature
        # error would otherwise shift every simulated start and stop and swamp the score.
        ran = {u: bool(a.get("running")) for u, a in actual.items()}
        fans = {u: a.get("fan") for u, a in actual.items()}
        for rec in self.records:
            if len(rec["done"]) == len(HORIZONS_MIN):
                continue
            dt = (now - rec["t"]).total_seconds() / 3600
            n = max(1, int(dt * 60 / 2 + 0.999))  # steps of at most 2 min (ticks can be sparse)
            acc = rec.get("acc")
            for _ in range(n if dt > 0 else 0):
                rec["temps"], rec["running"], y = hm.step(self.model, rec["temps"], outdoor, sun, modes, sps,
                                                          ran, dt / n, rec.setdefault("lagged", {}), hold=True,
                                                          fans=fans, masses=rec.setdefault("masses", {}),
                                                          comp=rec.setdefault("comp", {"h": 0.0}))
                if acc is not None:  # what the ML features need: the units' outputs, the compressor, the sun over the horizon
                    mn = dt * 60 / n
                    for u, v in y.items():
                        acc["y"][u] = acc["y"].get(u, 0.0) + v * mn
                    acc["on"] += mn if any(ran.values()) else 0.0
                    acc["heat"] += mn if any(md == "heat" for md in modes.values()) else 0.0
                    acc["o"] += (outdoor if outdoor is not None else 0.0) * mn
                    acc["s"] += as_h(sun) * mn
                    acc["min"] += mn
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
                    self._ml_log(rec, rid, h, pred, real, now)
        while self.points and now - self.points[0][0] > self.keep:
            self.points.popleft()

    def due(self, now: datetime) -> bool:
        return self.model is not None and (self.next_plan is None or now >= self.next_plan)

    # --- heavy part (executor) ---
    def run(self, now: datetime, rooms: list, units: list, temps: dict, actual: dict, outdoor, sun_at,
            settings: core.Settings, ignore=(), fixed=None, drives=None):
        """Plan, if due. actual: unit id -> {"mode", "setpoint" (F), "running"}; outdoor: F now, or outdoor(datetime) ->
        F from a forecast; sun_at(datetime) -> sun vector (model.sun_vector) or 0..1; fixed: unit id -> (mode, setpoint) for units on their own
        thermostats (planned around, not for); drives: unit id -> core.DriveState, the outer loops as they are now."""
        if self.model is None or not (self.next_plan is None or now >= self.next_plan):
            return
        self.next_plan = now + self.plan_every
        try:
            self._plan(now, rooms, units, temps, actual, outdoor, sun_at, settings, ignore, fixed or {}, drives or {})
            self._project(now, temps, actual, outdoor, sun_at)
            self.error = None
        except Exception as e:  # never let the model take the integration down; show it instead
            self.error = f"{type(e).__name__}: {e}"
        if settings.joint_shadow and (self.next_joint is None or now >= self.next_joint):
            slow = self.jplan_ms is not None and self.jplan_ms > 120_000  # a plan over 2 min: try less often
            self.next_joint = now + self.joint_every * (4 if slow else 1)
            try:
                self._joint_plan(now, rooms, units, temps, actual, outdoor, sun_at, settings, ignore, fixed or {})
                self.jerror = None
            except Exception as e:
                self.jerror = f"{type(e).__name__}: {e}"

    def _joint_plan(self, now, rooms, units, temps, actual, outdoor, sun_at, settings, ignore, fixed):
        """What the joint planner would do now, with its own pretend house mode (never sent anywhere)."""
        t0 = time.perf_counter()
        running = {u: bool(a.get("running")) for u, a in actual.items()}
        p = self.joint.plan(self.model, rooms, units, temps, now, self.jhouse, running, outdoor, sun_at, settings,
                            ignore=set(ignore), fixed=fixed, lagged=dict(self.lagged),
                            fans={u: a.get("fan") for u, a in actual.items()}, masses=dict(self.masses),
                            comp=dict(self.comp), **self.observer.corrections())
        mode = self.jhouse.mode
        if mode not in ("heat", "cool"):
            want = p.mode if p.mode != "off" else "heat"
            benefit = 0.0 if p.mode == "off" else p.by_mode["off"]["cost"] - p.cost
            self.jhouse, _ = core.choose_mode(benefit if want == "heat" else 0.0, benefit if want == "cool" else 0.0,
                                              self.jhouse, now, settings)
        elif p.mode != mode:
            self.jhouse = core.HouseState(p.mode, now)
        new = self.jhouse.mode
        if new in ("heat", "cool"):
            if self.jlast_mode in ("heat", "cool") and new != self.jlast_mode:
                self.jflips.append(now)
            self.jlast_mode = new
        while self.jflips and now - self.jflips[0] > self.keep:
            self.jflips.popleft()
        self.jplan, self.jplan_at = p, now
        self.jplan_ms = round((time.perf_counter() - t0) * 1000, 1)

    def _project(self, now, temps, actual, outdoor, sun_at, horizon_min=120, step_min=10):
        """What the model expects over the next 2 h if every unit keeps its current mode and setpoint."""
        start = {r: temps[r] for r in self.model.rooms if temps.get(r) is not None}
        n = horizon_min // step_min
        times = [now + timedelta(minutes=step_min * (k + 1)) for k in range(n)]
        modes = {u: (a.get("mode") if a.get("mode") in ("heat", "cool") else "off") for u, a in actual.items()}
        traj, _, _ = hm.rollout(self.model, dict(start), modes, {u: a.get("setpoint") for u, a in actual.items()},
                                {u: bool(a.get("running")) for u, a in actual.items()}, _seq(outdoor, times),
                                [sun_at(t) for t in times], step_min / 60, substeps=2, lagged=dict(self.lagged),
                                fans={u: a.get("fan") for u, a in actual.items()}, masses=dict(self.masses),
                                comp=dict(self.comp), **self.observer.corrections())
        self.projection = {"at": now.isoformat(), "temps": {
            r: [[t.isoformat(), round(step[r], 2)] for t, step in zip(times, traj)] for r in start}}

    def _plan(self, now, rooms, units, temps, actual, outdoor, sun_at, settings, ignore, fixed, drives):
        ps = self.ps
        n = int(round(ps.horizon_h * 60 / ps.step_min))
        times = [now + timedelta(minutes=ps.step_min * k) for k in range(n)]
        running = {u: bool(a.get("running")) for u, a in actual.items()}
        if settings.drive:  # what's planned is each unit's target; moving it is what the change cost charges
            now_sp = {u.id: self.targets.get(u.id) for u in units}
        else:
            now_sp = {u.id: (self.sent[u.id][0].setpoint if u.id in self.sent else actual.get(u.id, {}).get("setpoint"))
                      for u in units}
        t0 = time.perf_counter()
        p = optimizer.plan(self.model, rooms, units, temps, now, self.house, now_sp, running, _seq(outdoor, times),
                           [sun_at(t) for t in times], settings, ps, ignore=set(ignore), fixed=fixed,
                           drives=drives if settings.drive else None, lagged=dict(self.lagged),
                           fans={u: a.get("fan") for u, a in actual.items()}, masses=dict(self.masses),
                           comp=dict(self.comp), **self.observer.corrections())
        setpoints, first = p.setpoints, p.first
        forced = next((f[0] for f in fixed.values() if f[0] in ("heat", "cool")), None)
        if forced and self.house.mode != forced:
            self.house = core.HouseState(forced, now)  # a unit on its own thermostat sets the shared mode
        elif self.house.mode not in ("heat", "cool"):
            # Starting from off: the "demand" is how much better the best mode is than staying off; same start
            # threshold and confirmation time as today's controller.
            want = p.mode if p.mode != "off" else "heat"
            benefit = 0.0 if p.mode == "off" else p.by_mode["off"]["cost"] - p.cost
            self.house, _ = core.choose_mode(benefit if want == "heat" else 0.0, benefit if want == "cool" else 0.0,
                                             self.house, now, settings)
            setpoints = p.by_mode.get(self.house.mode, {}).get("setpoints", {})
            first = p.by_mode.get(self.house.mode, {}).get("first", {})
        elif p.mode != self.house.mode:
            # Switching heat <-> cool takes the other mode winning several plans in a row (on top of the planner's own
            # margin and dwell): one plan's blip, e.g. a room's jump the model didn't foresee, doesn't flip the house.
            self._switch_votes += 1
            if self._switch_votes >= self.confirm_plans:
                self.house = core.HouseState(p.mode, now)
                self._switch_votes = 0
            elif self.house.mode in p.by_mode:
                setpoints, first = p.by_mode[self.house.mode]["setpoints"], p.by_mode[self.house.mode]["first"]
        else:
            self._switch_votes = 0
        self.decision = {}
        for u in units:
            if u.id in fixed:
                self.decision[u.id] = {"mode": "thermostat", "setpoint": None}
                continue
            sp = setpoints.get(u.id)
            if self.house.mode not in ("heat", "cool"):
                self.decision[u.id] = {"mode": "off", "setpoint": None}
                continue
            if settings.drive:
                # The plan is a target; the integration's outer loop turns it into setpoints. Shown: the setpoint it
                # would send first.
                self.targets[u.id] = sp
                self.decision[u.id] = {"mode": self.house.mode, "target": sp, "setpoint": (first or {}).get(u.id)}
                continue
            cmd = core.Command(self.house.mode, sp, p.reason, True, sp)
            cmd, changed = core.stabilize(cmd, self.sent.get(u.id), now, settings)
            if changed and cmd.setpoint is not None:
                self.sent[u.id] = (cmd, now)
            self.decision[u.id] = {"mode": cmd.mode, "setpoint": cmd.setpoint}
        self.plan, self.plan_at = p, now
        self.plan_ms = round((time.perf_counter() - t0) * 1000, 1)

    def readiness(self, now: datetime, rooms_needed, strict: bool = True, min_n: int = 36):
        """(ready to control units?, reasons it isn't). Strict, for handing a unit over: a usable fit, a fresh plan,
        and for every room it scores, 60- and 120-min predictions (at least `min_n` each, about 6 h) beating "nothing
        changes" (to within READY_TOL_F). Not strict, for staying in control: 10% slack on top and fewer scores, so it doesn't
        flap."""
        why = []
        m = self.model
        if m is None:
            return False, ["no model yet"]
        if not (m.meta or {}).get("usable"):
            why.append("the latest fit doesn't beat 'nothing changes' on held-out data")
        if self.error:
            why.append(f"planner error: {self.error}")
        if self.plan_at is None or now - self.plan_at > timedelta(minutes=15):
            why.append("no fresh plan")
        acc = self.accuracy()
        slack, need = (1.0, min_n) if strict else (1.1, max(1, min_n // 3))
        for rid in sorted(rooms_needed):
            for h in ("60", "120"):
                v = acc.get(rid, {}).get(h)
                if not v or v["n"] < need:
                    why.append(f"{rid}: {0 if not v else v['n']}/{need} {h}-min predictions scored so far")
                elif v["rmse"] >= v["base_rmse"] * slack + READY_TOL_F:
                    why.append(f"{rid}: {h}-min predictions off by {v['rmse']}°, no better than 'nothing changes' "
                               f"({v['base_rmse']}°)")
        return not why, why

    # --- surviving restarts ---
    def to_state(self) -> dict:
        """The scores, chart points and recent readings, JSON-ready. Predictions still in progress are left out: after
        a restart they'd be advanced across the gap without the real inputs, so scoring them wouldn't be fair."""
        iso = lambda t: t.isoformat()
        return {"version": 1,
                "errors": [[iso(t), r, h, round(e, 3), round(b, 3)] for t, r, h, e, b in self.errors],
                "points": [[iso(t), r, h, v] for t, r, h, v in self.points],
                "readings": {r: [[iso(t), v] for t, v in d] for r, d in self.readings.items()},
                "observer": self.observer.to_state(),
                "ml_errors": [[iso(t), r, h, round(e, 3), round(p, 3)] for t, r, h, e, p in self.ml_errors]}

    def load_state(self, d: dict, now: datetime):
        """Restore what to_state saved, dropping anything older than the windows it's kept for."""
        if not d or d.get("version") != 1:
            return
        p = datetime.fromisoformat
        keep_since = chart_since = now - self.keep
        self.errors = deque((p(t), r, h, e, b) for t, r, h, e, b in d.get("errors", []) if p(t) >= keep_since)
        self.points = deque((p(t), r, h, v) for t, r, h, v in d.get("points", []) if p(t) >= chart_since)
        self.readings = {r: deque((p(t), v) for t, v in pts if p(t) >= keep_since) for r, pts in d.get("readings", {}).items()}
        self.observer.load_state(d.get("observer"))
        self.ml_errors = deque((p(t), r, h, e, b) for t, r, h, e, b in d.get("ml_errors", []) if p(t) >= keep_since)

    # --- reporting ---
    def joint_fields(self) -> dict:
        p = self.jplan
        if p is None:
            return None
        return {"house": self.jhouse.mode, "cost": round(p.cost, 3), "comfort": p.comfort, "fans": p.fans,
                "run": {u: (v[0] if v else "off") for u, v in p.pattern.items()},
                "next": {u: (v[1] if len(v) > 1 else v[0]) for u, v in p.pattern.items() if v}}

    def log_fields(self) -> dict:
        if self.plan is None:
            return None
        return {"joint": self.joint_fields(), "house": self.house.mode, "units": self.decision, "cost": round(self.plan.cost, 3),
                "comfort": self.plan.comfort, "costs": {k: v["cost"] for k, v in self.plan.by_mode.items()}}

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
        mlp = {}
        for _, rid, h, e, p in self.ml_errors:  # the ML-corrected prediction on the same scale, where there is one
            mlp.setdefault((rid, str(h)), []).append((e, p))
        for (rid, h), pairs in mlp.items():
            if rid in res and h in res[rid]:
                n = len(pairs)
                res[rid][h]["ml_rmse"] = round((sum(e * e for e, _ in pairs) / n) ** 0.5, 2)
                res[rid][h]["ml_n"] = n
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
            "joint": None if self.jplan is None else {
                "at": self.jplan_at.isoformat(), "ms": self.jplan_ms, "house": self.jhouse.mode, "mode": self.jplan.mode,
                "reason": self.jplan.reason, "cost": round(self.jplan.cost, 3), "pattern": self.jplan.pattern,
                "fans": self.jplan.fans, "flips_24h": len(self.jflips), "error": self.jerror},
            "observer": self.observer.to_state(),
            "ml": {"corrector": self.corrector is not None, "rows_waiting": len(self.ml_rows),
                   "enabled": {k: v for k, v in (self.corrector.enabled.items() if self.corrector else [])}},
            "projection": self.projection,
            "accuracy": self.accuracy(),
            "history": self.history(now),
            "horizons": list(HORIZONS_MIN),
        }


# --- history for the Model page's charts (pure, so they're testable offline) ---
def compact(row: dict) -> dict:
    """The bits of a minute-log row the charts need: room temps/targets/priorities; per unit its real mode and setpoint,
    what today's controller commanded and what the model would have; both house modes; the model's predicted comfort."""
    mp = row.get("mpc") or {}
    mu = mp.get("units") or {}
    return {"t": row["t"], "h": row.get("house"), "mh": mp.get("house"), "mc": mp.get("comfort"),
            "r": {rid: [v.get("temp"), v.get("target"), v.get("prio")] for rid, v in (row.get("rooms") or {}).items()
                  if v.get("temp") is not None},
            "u": {uid: [v.get("state"), v.get("sp"), v.get("cmd"), v.get("cmd_sp"), (mu.get(uid) or {}).get("mode"),
                        (mu.get(uid) or {}).get("setpoint")] for uid, v in (row.get("units") or {}).items()}}


def comfort_series(samples: list, rooms, settings: core.Settings, ahead_min: int = 120) -> list:
    """[[iso, actual comfort score over the next `ahead_min` (band cost per hour, as an average), the model's predicted
    score for its own plan at that moment]] for compact samples, scoring `rooms` the way the optimizer does."""
    rate = []
    for smp in samples:
        c = 0.0
        for rid in rooms:
            v = smp["r"].get(rid)
            if v and v[0] is not None and v[1] is not None:
                c += optimizer.band_cost(v[0], v[1], v[2] or 3, settings)
        rate.append((datetime.fromisoformat(smp["t"]), c))
    out, j, acc, n = [], 0, 0.0, 0
    for i, (t, _) in enumerate(rate):  # forward-looking average over [t, t + ahead)
        while j < len(rate) and rate[j][0] < t + timedelta(minutes=ahead_min):
            acc, n, j = acc + rate[j][1], n + 1, j + 1
        if rate[j - 1][0] - t >= timedelta(minutes=ahead_min - 10):  # only once the whole window is in the past
            out.append([samples[i]["t"], round(acc / n, 3), samples[i].get("mc")])
        else:
            out.append([samples[i]["t"], None, samples[i].get("mc")])
        acc, n = acc - rate[i][1], n - 1
    return out


def accuracy_by_hour(errors) -> dict:
    """room -> horizon -> [[iso hour, rmse, nothing-changes rmse, n]] from (t, room, horizon, err, base) tuples."""
    groups = {}
    for t, rid, h, e, b in errors:
        hour = t.replace(minute=0, second=0, microsecond=0)
        groups.setdefault((rid, h, hour), []).append((e, b))
    out = {}
    for (rid, h, hour), pairs in sorted(groups.items(), key=lambda kv: kv[0][2]):
        n = len(pairs)
        out.setdefault(rid, {}).setdefault(str(h), []).append(
            [hour.isoformat(), round((sum(e * e for e, _ in pairs) / n) ** 0.5, 3),
             round((sum(b * b for _, b in pairs) / n) ** 0.5, 3), n])
    return out
