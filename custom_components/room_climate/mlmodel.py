"""A machine-learned predictor of the rooms' temperatures, trained daily from the minute log, scored next to the physics model.

Why next to, not instead of. The planner needs to ask "what if this unit runs for the next hour" (physics.py's model
answers that for any plan); a model trained on the log alone sees only what the controller happened to do. So the ML here
is a *correction layer*: for each room and horizon (30/60/120 min) it predicts what the physics model gets wrong, from
features the physics model doesn't use directly (time of day, how long the compressor has run, the room's recent trend, its
neighbours, the sun), and the corrected prediction is the physics model's plus that. It is trained on the physics
model's out-of-sample misses, and is used only where it beats the physics model on held-out data.

Pure Python (the integration has no numpy): features here, a ridge regression here. tools/ml_benchmark.py trains
gradient-boosted trees (scikit-learn) on the same features to see whether a stronger learner is worth deploying.
"""
from __future__ import annotations

import math
from datetime import datetime

try:
    from . import model as hm
except ImportError:  # offline
    import model as hm  # type: ignore

HORIZONS = (30, 60, 120)
STEP_MIN = 2  # replay step
FEATURES = ["ode", "dT_out", "out_h", "trend10", "trend30", "nbr1", "nbr2", "own_y", "oth_y", "comp_frac", "t_on0",
            "heat_frac", "sun_mean", "sun0", "hsin", "hcos", "inlet_gap", "inlet_d10"]
NF = len(FEATURES)
NF_V1 = 16  # models saved before the inlet features have no `nf`: they were trained on the first 16


def pad(rows: list) -> list:
    """Rows logged before a feature existed get 0 for it (rows are [time, room, horizon, change, features])."""
    for r in rows:
        if len(r[4]) < NF:
            r[4] = list(r[4]) + [0.0] * (NF - len(r[4]))
    return rows


def make_features(ode, T0, out0, out_mean, T10, T30, nb1, nb2, own_y, oth_y, comp_frac, t_on0_min, heat_frac, sun_mean,
                  sun0, hour, inlet_gap=0.0, inlet_d10=0.0) -> list:
    """The feature vector (FEATURES order), shared by the offline benchmark and the live logger so they can't drift:
    ode: the physics model's predicted change over the horizon; dT_out: room minus outdoor now; out_h: mean outdoor over
    the horizon minus the room; trend10/30: the room's change over the last 10/30 min; nbr1/2: its two strongest-coupled
    neighbours minus the room; own_y/oth_y: mean signed output of the room's own unit / of the other units over the
    horizon; comp_frac: share of the horizon the compressor ran; t_on0: hours it had already run (cap 1.5); heat_frac:
    share of it in heat mode; sun_mean/sun0: overhead sun index over the horizon / now; hsin/hcos: time of day;
    inlet_gap: the room's own unit's inlet sensor minus the room (0 while the unit is fed the room sensor, which it then
    echoes); inlet_d10: that inlet's change over the last 10 min."""
    return [ode, T0 - out0, out_mean - T0, T0 - T10, T0 - T30, nb1, nb2, own_y, oth_y, comp_frac, min(t_on0_min, 90) / 60,
            heat_frac, sun_mean, sun0, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24), inlet_gap, inlet_d10]


class Inputs:
    """The minute log as per-minute arrays: readings, weather, sun, the units as they were, the compressor's run time."""

    def __init__(self, rows: list):
        self.rows = rows
        self.n = len(rows)
        self.ts = [datetime.fromisoformat(r["t"]) for r in rows]
        self.T = {}
        for r in rows:
            for rid, v in (r.get("rooms") or {}).items():
                self.T.setdefault(rid, [None] * self.n)
        for i, r in enumerate(rows):
            for rid, v in (r.get("rooms") or {}).items():
                self.T[rid][i] = v.get("temp")
        out, last = [], None
        for r in rows:
            o = r.get("outdoor")
            last = o if o is not None else last
            out.append(last)
        self.out = out
        self.sun, self.ghi = [], []  # ghi: the overhead sun index (0..1) of each minute's sun vector
        for r in rows:
            sn = r.get("sun") or {}
            sv = hm.sun_from_irradiance(sn.get("elev"), sn.get("az"), sn.get("ghi"), sn.get("dni"), sn.get("dhi"))
            self.sun.append(sv)
            self.ghi.append(float(sv.get("h", 0.0)))
        self.modes, self.sps, self.ran, self.fans = [], [], [], []
        for r in rows:
            us = {u: x for u, x in (r.get("units") or {}).items() if x.get("state") not in (None, "unavailable")}
            self.modes.append({u: (x["state"] if x["state"] in ("heat", "cool") else "off") for u, x in us.items()})
            self.sps.append({u: x.get("sp") for u, x in us.items()})
            self.ran.append({u: x.get("action") in ("heating", "cooling") for u, x in us.items()})
            self.fans.append({u: x.get("fan") for u, x in us.items()})
        self.t_on = []  # minutes the compressor has been running at each minute (0 when no unit runs)
        t = 0
        for ran in self.ran:
            t = t + 1 if any(ran.values()) else 0
            self.t_on.append(t)
        self.exp = [bool(r.get("exp")) for r in rows]
        op, cur = [], {}
        for r in rows:  # register / door positions (0..1) as they were each minute, carried forward from the last logged
            cur = r.get("op") or cur
            op.append(cur)
        self.op = op
        self.utemp = [{u: x.get("temp") for u, x in (r.get("units") or {}).items()} for r in rows]  # unit inlet sensors


def own_unit(m: hm.HouseModel, rid: str):
    r = m.rooms[rid]
    return max(r.heat, key=lambda u: r.heat[u], default=None) if r.heat else None


def neighbours(m: hm.HouseModel, rid: str, k: int = 2) -> list:
    c = m.rooms[rid].coupling
    return sorted(c, key=lambda j: -c[j])[:k]


def rollout(m: hm.HouseModel, inp: Inputs, i: int, horizon: int = max(HORIZONS)):
    """The physics model from the readings at minute i, with what the units, weather and sun then really did:
    ({room: {h: predicted temperature at minute i + h}}, {h: {unit: mean signed output}}, {h: share of minutes the
    compressor ran}, {h: share of minutes in heat mode})."""
    T = {rid: v[i] for rid, v in inp.T.items() if rid in m.rooms and v[i] is not None}
    lag, mass, comp = {}, {}, {"h": inp.t_on[i] / 60}
    pred, ysum, on_min, heat_min = {}, {}, 0, 0
    ymean, onf, heatf = {}, {}, {}
    for s in range(0, horizon, STEP_MIN):
        j = min(inp.n - 1, i + s)
        if inp.out[j] is None:
            return None
        T, _, y = hm.step(m, T, inp.out[j], inp.sun[j], inp.modes[j], inp.sps[j], inp.ran[j], STEP_MIN / 60, lag, hold=True,
                          fans=inp.fans[j], masses=mass, comp=comp)
        for u, v in y.items():
            ysum[u] = ysum.get(u, 0.0) + v * STEP_MIN
        on_min += STEP_MIN if any(inp.ran[j].values()) else 0
        heat_min += STEP_MIN if any(md == "heat" for md in inp.modes[j].values()) else 0
        done = s + STEP_MIN
        if done in HORIZONS:
            pred[done] = dict(T)
            ymean[done] = {u: v / done for u, v in ysum.items()}
            onf[done], heatf[done] = on_min / done, heat_min / done
    return pred, ymean, onf, heatf


def feature_rows(m: hm.HouseModel, inp: Inputs, i: int, with_target: bool = True):
    """Rows (room, horizon, target, features) for the origin at minute i. target: the room's real change over the
    horizon, minus the physics model's predicted change when the model is given (`ode` feature is the latter)."""
    if i < 30 or i + max(HORIZONS) >= inp.n or any(inp.exp[i:i + max(HORIZONS)]):
        return []
    r = rollout(m, inp, i)
    if r is None:
        return []
    pred, ymean, onf, heatf = r
    out = []
    hour = inp.ts[i].hour + inp.ts[i].minute / 60
    for rid in m.rooms:
        T0 = inp.T[rid][i]
        if T0 is None or inp.T[rid][i - 10] is None or inp.T[rid][i - 30] is None:
            continue
        nb = [inp.T[j][i] - T0 if inp.T[j][i] is not None else 0.0 for j in neighbours(m, rid)]
        nb += [0.0] * (2 - len(nb))
        own = own_unit(m, rid)
        u0, u10 = (inp.utemp[i].get(own), inp.utemp[i - 10].get(own)) if own else (None, None)
        gap = (u0 - T0) if u0 is not None else 0.0
        d10 = (u0 - u10) if (u0 is not None and u10 is not None) else 0.0
        for h in HORIZONS:
            if h not in pred or rid not in pred[h]:
                continue
            real = inp.T[rid][i + h]
            if with_target and real is None:
                continue
            ym = ymean[h]
            own_y = ym.get(own, 0.0) if own else 0.0
            oth_y = sum(v for u, v in ym.items() if u != own)
            o_mean = sum(x for x in inp.out[i:i + h:5] if x is not None) / max(1, len([x for x in inp.out[i:i + h:5] if x is not None]))
            g = inp.ghi[i:i + h:5]
            f = make_features(pred[h][rid] - T0, T0, inp.out[i] if inp.out[i] is not None else T0, o_mean, inp.T[rid][i - 10],
                              inp.T[rid][i - 30], nb[0], nb[1], own_y, oth_y, onf[h], inp.t_on[i], heatf[h],
                              sum(g) / len(g), inp.ghi[i], hour, gap, d10)
            out.append((rid, h, None if real is None else real - T0, f))
    return out


# ---- a ridge regression in pure Python (what the integration trains and runs) ----
class Ridge:
    def __init__(self, lam: float = 5.0):
        self.lam, self.mu, self.sd, self.w, self.b = lam, [], [], [], 0.0

    def fit(self, X: list, y: list):
        n, p = len(X), len(X[0])
        self.mu = [sum(r[j] for r in X) / n for j in range(p)]
        self.sd = [max(1e-6, (sum((r[j] - self.mu[j]) ** 2 for r in X) / n) ** 0.5) for j in range(p)]
        Z = [[(r[j] - self.mu[j]) / self.sd[j] for j in range(p)] for r in X]
        self.b = sum(y) / n
        yc = [v - self.b for v in y]
        A = [[0.0] * p for _ in range(p)]
        g = [0.0] * p
        for z, t in zip(Z, yc):
            for a in range(p):
                za = z[a]
                g[a] += za * t
                row = A[a]
                for c in range(a, p):
                    row[c] += za * z[c]
        for a in range(p):
            A[a][a] += self.lam
            for c in range(a):
                A[a][c] = A[c][a]
        self.w = _solve(A, g)
        return self

    def predict(self, x: list) -> float:
        return self.b + sum(w * (v - m) / s for w, v, m, s in zip(self.w, x, self.mu, self.sd))

    def to_dict(self):
        return {"lam": self.lam, "mu": self.mu, "sd": self.sd, "w": self.w, "b": self.b}

    @classmethod
    def from_dict(cls, d):
        r = cls(d["lam"])
        r.mu, r.sd, r.w, r.b = d["mu"], d["sd"], d["w"], d["b"]
        return r


def _solve(a: list, b: list) -> list:
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


# ---- the daily-trained correction ----
class MLCorrector:
    """Per horizon, one pooled ridge (rooms as one-hot features) predicting what the physics model gets wrong. A room's
    correction at a horizon is used only if it beat the physics model on that room's most recent held-out rows."""

    def __init__(self, rooms: list, models: dict, enabled: dict, report: dict, nf: int = NF):
        self.rooms, self.models, self.enabled, self.report, self.nf = rooms, models, enabled, report, nf

    def correct(self, room: str, h: int, f: list) -> float:
        """Add this to the physics model's predicted change (0 when the correction isn't enabled there)."""
        mdl = self.models.get(h)
        if mdl is None or not self.enabled.get(f"{room}|{h}") or room not in self.rooms:
            return 0.0
        return mdl.predict(list(f[:self.nf]) + [1.0 if room == q else 0.0 for q in self.rooms])

    def to_dict(self):
        return {"rooms": self.rooms, "models": {str(h): m.to_dict() for h, m in self.models.items()},
                "enabled": self.enabled, "report": self.report, "nf": self.nf}

    @classmethod
    def from_dict(cls, d):
        return cls(d["rooms"], {int(h): Ridge.from_dict(m) for h, m in d["models"].items()}, d["enabled"], d.get("report", {}),
                   d.get("nf", NF_V1))


def train(rows: list, lam: float = 20.0, holdout: float = 0.25, min_rows: int = 150, min_gain: float = 0.03):
    """rows: [[iso time, room, horizon, real change, features]] logged by the live planner (the physics model's predictions
    at the time, so out-of-sample). Time-ordered split: fit on the first 1 - holdout, enable a room's correction at a
    horizon only if it cuts that room's held-out RMSE by `min_gain`; then refit on everything. -> MLCorrector or None."""
    rows = sorted(pad(rows), key=lambda r: r[0])
    rooms = sorted({r[1] for r in rows})
    if not rooms:
        return None
    oh = {q: [1.0 if q == x else 0.0 for x in rooms] for q in rooms}
    models, enabled, report = {}, {}, {}
    for h in HORIZONS:
        data = [(r[0], r[1], r[3], r[4]) for r in rows if r[2] == h]
        if len(data) < min_rows:
            continue
        cut = int(len(data) * (1 - holdout))
        X = [f + oh[room] for _, room, _, f in data]
        res = [y - f[0] for _, _, y, f in data]
        m1 = Ridge(lam).fit(X[:cut], res[:cut])
        errs = {}
        for k in range(cut, len(data)):
            room = data[k][1]
            e0 = -res[k]  # physics prediction minus real
            e1 = e0 + m1.predict(X[k])
            a = errs.setdefault(room, [[], []])
            a[0].append(e0)
            a[1].append(e1)
        for room, (e0, e1) in errs.items():
            if len(e0) < 20:
                continue
            r0, r1 = (sum(x * x for x in e0) / len(e0)) ** 0.5, (sum(x * x for x in e1) / len(e1)) ** 0.5
            enabled[f"{room}|{h}"] = r1 <= r0 * (1 - min_gain)
            report[f"{room}|{h}"] = [round(r0, 3), round(r1, 3), len(e0)]
        models[h] = Ridge(lam).fit(X, res)
    return MLCorrector(rooms, models, enabled, report) if models else None


# ---- trees trained elsewhere (scikit-learn on the Mac), evaluated here in pure Python ----
class Trees:
    """A gradient-boosted tree ensemble exported from scikit-learn (tools/ml_train_local.py): the baseline plus, for each
    tree, the value of the leaf the features fall in (leaf values already include the learning rate)."""

    def __init__(self, base: float, trees: list):
        self.base, self.trees = base, trees

    def predict(self, x: list) -> float:
        s = self.base
        for t in self.trees:
            f, th, left, right, leaf, v = t["f"], t["t"], t["l"], t["r"], t["leaf"], t["v"]
            i = 0
            while not leaf[i]:
                i = left[i] if x[f[i]] <= th[i] else right[i]
            s += v[i]
        return s

    def to_dict(self):
        return {"base": self.base, "trees": self.trees}

    @classmethod
    def from_dict(cls, d):
        return cls(d["base"], d["trees"])


class MLBlend:
    """The best walk-forward predictor so far: half the physics model plus the ridge correction, half trees that predict the
    change directly (from every feature but the physics prediction). Same interface as MLCorrector: correct() is what to
    add to the physics model's predicted change, so the planner's scoring and the logging need no change."""

    kind = "blend"

    def __init__(self, rooms: list, ridge, trees: dict, weight: float = 0.5, report: dict = None, nf: int = NF):
        self.rooms, self.ridge, self.trees, self.weight, self.report = rooms, ridge, trees, weight, report or {}
        self.nf = nf  # how many features the trees were trained on (the physics prediction + the rest)
        self.enabled = ridge.enabled if ridge is not None else {}

    def correct(self, room: str, h: int, f: list) -> float:
        rc = self.ridge.correct(room, h, f) if self.ridge is not None else 0.0
        t = self.trees.get(h)
        if t is None or room not in self.rooms:
            return rc
        pure = t.predict(list(f[1:self.nf]) + [1.0 if room == q else 0.0 for q in self.rooms])
        return (1.0 - self.weight) * rc + self.weight * (pure - f[0])

    def to_dict(self):
        return {"kind": "blend", "rooms": self.rooms, "weight": self.weight, "report": self.report, "nf": self.nf,
                "ridge": self.ridge.to_dict() if self.ridge is not None else None,
                "trees": {str(h): t.to_dict() for h, t in self.trees.items()}}

    @classmethod
    def from_dict(cls, d):
        return cls(d["rooms"], MLCorrector.from_dict(d["ridge"]) if d.get("ridge") else None,
                   {int(h): Trees.from_dict(t) for h, t in d["trees"].items()}, d.get("weight", 0.5), d.get("report"),
                   d.get("nf", NF_V1))


def load(d: dict):
    """A saved correction of either kind."""
    return MLBlend.from_dict(d) if d.get("kind") == "blend" else MLCorrector.from_dict(d)
