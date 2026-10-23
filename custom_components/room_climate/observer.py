"""Online correction of the thermal model from what the rooms are really doing: the planner's disturbance observer.

The planner replans from the measured temperatures every few minutes, which corrects where the rooms are, but not how
they respond: if the kitchen warms at a third of the modelled rate, or Bedroom 3 jumps twice as far when the shared
compressor starts, every replan makes the same wrong bet. This watches the model's own short-term predictions and
learns corrections, then hands them to the planner:

- effect_scale: room id -> {unit id: multiplier on that unit's heating/cooling effect on the room}
- leak_scale:   room id -> multiplier on the room's leak term (heat a unit picks up while it idles and the compressor
                runs for the others)
- offsets:      room id -> extra F/h nothing in the model explains (sun, envelope, people)

How they are learned: every `every_min` a short prediction starts from the readings and advances with what the units,
weather and sun really did (as mpc.ShadowPlanner does for its scores). When one is `window_min` old, the gap between
the reading and the prediction, per hour, is the error rate. For each room that rate is regressed on what each cause
(its significant units, its leak, a constant) was expected to contribute in that window: the error rate is
(scale - 1) x expected contribution for a cause with the wrong gain, and a constant for a drift. Recursive ridge
regression with forgetting: all causes are fitted together, so several acting at once need no attribution rule, and a
window where a cause did not act says nothing about that cause's gain.

Why this is stable. A gain error is not a constant: learned as a constant drift it would also apply while the unit is
off, the planner would see a phantom cooling, plan more heat, and hunt (the first version, kept as `naive`, did exactly
that in simulation). Here a gain only acts through the cause it belongs to. Each estimate is a ridge solution (always
bounded; with no data it returns to "no correction"), updated with exponential forgetting, and projected into fixed
bounds, so the estimator is a stable filter. The remaining loop, estimate -> plan -> which causes act -> information,
is the usual adaptive-control one; the prior and the bounds keep it from running away, and tools/monte_carlo.py measures
it. Pure Python.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta

try:
    from . import model as hm
except ImportError:  # offline
    import model as hm  # type: ignore


def _solve(a, b):
    """Solve a x = b for a small dense system (Gaussian elimination, partial pivoting)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-12:
            return [0.0] * n
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


class Observer:
    def __init__(self, window_min: int = 20, every_min: int = 5, forget_h: float = 12.0, noise: float = 0.8,
                 scale_std: float = 0.3, offset_std: float = 0.25, d_max: float = 1.0, s_min: float = 0.25,
                 s_max: float = 3.0, min_coeff: float = 1.5, min_leak: float = 0.5,
                 offsets_on: bool = True, leak_on: bool = False, effect_on: bool = True, naive: bool = False):
        self.window = timedelta(minutes=window_min)
        self.every = timedelta(minutes=every_min)
        self.forget = math.exp(-(every_min / 60) / forget_h)  # per update; data older than forget_h fades
        self.noise = noise  # F/h, per window; large because windows overlap (data counted ~4 times)
        self.scale_prec, self.offset_prec = 1.0 / scale_std ** 2, 1.0 / offset_std ** 2  # ridge prior: no correction
        self.d_max, self.s_min, self.s_max = d_max, s_min, s_max
        self.min_coeff = min_coeff  # F/h at full output a unit must have on a room to get its own gain
        self.min_leak = min_leak
        self.offsets_on, self.leak_on, self.effect_on = offsets_on, leak_on, effect_on
        self.naive = naive  # the first version, kept as a negative control: a drift learned from every window
        # leak_on is off by default: learning the leak's scale helped in the simulator but made Bedroom 3's real
        # predictions worse in a causal replay of 40 h of the real log (tools/observer_replay.py --score), so the leak's
        # shape isn't just a scale. The defaults above (priors, bounds) were chosen on that same replay.
        self.zones: dict = {}  # room id -> the unit that serves it; when given, only that unit's gain is learned for the room
        self.offsets: dict = {}  # room id -> F/h
        self.leak_scale: dict = {}  # room id -> multiplier
        self.effect_scale: dict = {}  # room id -> {unit id: multiplier}
        self.stats: dict = {}  # room id -> {"keys", "A", "b"}: the regression's running sums
        self.records: deque = deque()
        self.next_start = None

    def reset(self):
        self.offsets, self.leak_scale, self.effect_scale, self.stats = {}, {}, {}, {}
        self.records.clear()
        self.next_start = None

    def corrections(self) -> dict:
        """Keyword arguments for optimizer.plan / model.rollout: only the corrections that differ from nothing."""
        return {"offsets": {r: v for r, v in self.offsets.items() if abs(v) >= 0.02},
                "leak_scale": {r: v for r, v in self.leak_scale.items() if abs(v - 1.0) >= 0.02},
                "effect_scale": {r: {u: s for u, s in by.items() if abs(s - 1.0) >= 0.02}
                                 for r, by in self.effect_scale.items()
                                 if any(abs(s - 1.0) >= 0.02 for s in by.values())}}

    def update(self, m: hm.HouseModel, now: datetime, temps: dict, actual: dict, outdoor, sun, lagged: dict = None,
               masses: dict = None, comp: dict = None):
        """Call every tick with the readings, the units as they really are ({unit id: {"mode", "setpoint", "running",
        "fan"}}), the weather now, and the model's carried states (lagged outputs, masses, compressor run time) for starting
        predictions."""
        if m is None or not actual:
            return
        modes = {u: (a.get("mode") if a.get("mode") in ("heat", "cool") else "off") for u, a in actual.items()}
        sps = {u: a.get("setpoint") for u, a in actual.items()}
        ran = {u: bool(a.get("running")) for u, a in actual.items()}
        fans = {u: a.get("fan") for u, a in actual.items()}
        for rec in self.records:
            span = (now - rec["t"]).total_seconds() / 3600
            n = max(1, int(span * 60 / 2 + 0.999))  # steps of at most 2 min (ticks can be sparse)
            for _ in range(n if span > 0 else 0):
                rec["temps"], _, _ = hm.step(m, rec["temps"], outdoor, sun, modes, sps, ran, span / n, rec["lagged"],
                                             hold=True, fans=fans, masses=rec["masses"], offsets=rec["offsets"],
                                             leak_scale=rec["leak_scale"], effect_scale=rec["effect_scale"],
                                             contrib=rec["contrib"], comp=rec["comp"])
            rec["t"] = now
        while self.records and now - self.records[0]["t0"] >= self.window:
            self._learn(m, self.records.popleft(), now, temps)
        if self.next_start is None or now >= self.next_start:
            self.next_start = now + self.every
            start = {r: temps[r] for r in m.rooms if temps.get(r) is not None}
            if start:
                self.records.append({"t0": now, "t": now, "start": dict(start), "temps": dict(start),
                                     "lagged": dict(lagged or {}), "masses": dict(masses or {}),
                                     "offsets": dict(self.offsets), "leak_scale": dict(self.leak_scale),
                                     "effect_scale": {r: dict(by) for r, by in self.effect_scale.items()},
                                     "contrib": {}, "comp": dict(comp) if comp is not None else None})
        # a long gap in the readings (a restart) makes the open predictions unfair: drop them
        while self.records and now - self.records[0]["t0"] > 3 * self.window:
            self.records.popleft()

    def _causes(self, rid: str, r: hm.RoomModel) -> tuple:
        """The causes worth a gain of their own in a room: its leak (if it has one), the units that matter to it."""
        keys = []
        if self.leak_on and sum(abs(c) for c in r.leak.values()) >= self.min_leak:
            keys.append(("leak", None))
        if self.effect_on:
            for uid in sorted(set(r.heat) | set(r.cool)):
                if self.zones.get(rid) not in (None, uid):
                    continue
                if max(abs(r.heat.get(uid, 0.0)), abs(r.cool.get(uid, 0.0))) >= self.min_coeff:
                    keys.append(("unit", uid))
        return tuple(keys)

    def _learn(self, m: hm.HouseModel, rec: dict, now: datetime, temps: dict):
        w_h = (now - rec["t0"]).total_seconds() / 3600
        if w_h < 0.5 * self.window.total_seconds() / 3600:
            return
        for rid, r in m.rooms.items():
            real, pred = temps.get(rid), rec["temps"].get(rid)
            if real is None or pred is None or rid not in rec["start"]:
                continue
            resid = (real - pred) / w_h  # F/h the prediction missed by over the window
            if self.naive:  # a drift from every window, whatever was acting
                d = self.offsets.get(rid, 0.0)
                resid -= d - rec["offsets"].get(rid, 0.0)
                self.offsets[rid] = max(-self.d_max, min(self.d_max, d * self.forget + 0.3 * resid))
                continue
            keys = self._causes(rid, r)
            n = len(keys) + (1 if self.offsets_on else 0)
            if n == 0:
                continue
            # The error relative to the uncorrected model, and what each cause was expected to contribute (at scale 1)
            y = resid + (rec["offsets"].get(rid, 0.0) if self.offsets_on else 0.0)
            x = []
            for kind, uid in keys:
                if kind == "leak":
                    v, s0 = rec["contrib"].get("leak", {}).get(rid, 0.0), rec["leak_scale"].get(rid, 1.0)
                else:
                    v, s0 = rec["contrib"].get("unit", {}).get(rid, {}).get(uid, 0.0), rec["effect_scale"].get(rid, {}).get(uid, 1.0)
                base = v / w_h / s0 if s0 > 1e-6 else 0.0
                x.append(base)
                y += (s0 - 1.0) * base
            if self.offsets_on:
                x.append(1.0)
            st = self.stats.get(rid)
            if st is None or st["keys"] != keys or len(st["b"]) != n:
                st = self.stats[rid] = {"keys": keys, "A": [[0.0] * n for _ in range(n)], "b": [0.0] * n}
            f, inv = self.forget, 1.0 / self.noise ** 2
            for i in range(n):
                st["b"][i] = st["b"][i] * f + x[i] * y * inv
                for j in range(n):
                    st["A"][i][j] = st["A"][i][j] * f + x[i] * x[j] * inv
            prior = [self.scale_prec] * len(keys) + ([self.offset_prec] if self.offsets_on else [])
            a = [[st["A"][i][j] + (prior[i] if i == j else 0.0) for j in range(n)] for i in range(n)]
            theta = _solve(a, st["b"])
            for i, (kind, uid) in enumerate(keys):
                s = max(self.s_min, min(self.s_max, 1.0 + theta[i]))
                if kind == "leak":
                    self.leak_scale[rid] = s
                else:
                    self.effect_scale.setdefault(rid, {})[uid] = s
            if self.offsets_on:
                self.offsets[rid] = max(-self.d_max, min(self.d_max, theta[-1]))

    # --- surviving restarts, and reporting ---
    def to_state(self) -> dict:
        return {"offsets": {r: round(v, 3) for r, v in self.offsets.items()},
                "leak_scale": {r: round(v, 3) for r, v in self.leak_scale.items()},
                "effect_scale": {r: {u: round(s, 3) for u, s in by.items()} for r, by in self.effect_scale.items()},
                "stats": {r: {"keys": [list(k) for k in st["keys"]], "A": [[round(v, 4) for v in row] for row in st["A"]],
                              "b": [round(v, 4) for v in st["b"]]} for r, st in self.stats.items()}}

    def load_state(self, d: dict):
        """Corrections and the regression's running sums, as to_state saved them."""
        if not d:
            return
        self.stats = {r: {"keys": tuple(tuple(k) for k in st["keys"]), "A": st["A"], "b": st["b"]}
                      for r, st in (d.get("stats") or {}).items()}
        self.offsets = {r: float(v) for r, v in (d.get("offsets") or {}).items()}
        self.leak_scale = {r: float(v) for r, v in (d.get("leak_scale") or {}).items()}
        self.effect_scale = {r: {u: float(s) for u, s in by.items()} for r, by in (d.get("effect_scale") or {}).items()}
