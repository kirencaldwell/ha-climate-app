"""A moisture model next to the temperature model: each room's water-vapour pressure from the outdoors, its neighbours, a daily
pattern of indoor sources, and whether a unit is cooling (a cooling coil dehumidifies).

State: the vapour pressure pv (kPa) of each room, from its temperature and relative humidity. Vapour pressure, not relative
humidity, is the thing that is conserved when a room warms or cools (RH falls as air warms; pv does not), so the temperature
model can drive RH separately: RH = pv / psat(T).

    d pv_r / dt = a_r (pv_out - pv_r)                    ventilation / infiltration, a_r >= 0 (per hour)
                + sum_k c_rk (pv_k - pv_r)               exchange with the other rooms, c_rk >= 0 (per hour)
                + s_r + s_r^sin sin(wt) + s_r^cos cos(wt)  indoor sources: showers, cooking, people (kPa per hour)
                - sum_u d_ru * cooling_u                  a unit that is cooling dries the rooms it serves, d_ru >= 0

Fitted per room by non-negative least squares on 10-minute steps (pure Python: the integration has no numpy). Pure Python, no
Home Assistant imports.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

STEP_MIN = 10


def psat_kpa(temp_f: float) -> float:
    c = (temp_f - 32.0) * 5.0 / 9.0
    return 0.6108 * math.exp(17.27 * c / (c + 237.3))


def pv_from(temp_f: float, rh: float) -> float:
    return psat_kpa(temp_f) * rh / 100.0


def rh_from(temp_f: float, pv: float) -> float:
    return max(0.0, min(100.0, 100.0 * pv / psat_kpa(temp_f)))


def dew_point_f(temp_f: float, rh: float) -> Optional[float]:
    if rh is None or rh <= 0:
        return None
    pv = pv_from(temp_f, rh)
    lnr = math.log(pv / 0.6108)
    c = 237.3 * lnr / (17.27 - lnr)
    return c * 9.0 / 5.0 + 32.0


def _harm(t: datetime):
    w = 2 * math.pi * (t.hour + t.minute / 60.0) / 24.0
    return math.sin(w), math.cos(w)


def nnls_gram(G: list, c: list, lo: list, ridge: float = 1e-6, iters: int = 400) -> list:
    """min 0.5 b'Gb - c'b + 0.5 ridge |b|^2 with b_j >= lo_j (lo_j = 0 or -inf), by cyclic coordinate descent on the Gram matrix."""
    n = len(c)
    b = [max(0.0, lo[j]) if lo[j] > -1e9 else 0.0 for j in range(n)]
    for _ in range(iters):
        moved = 0.0
        for j in range(n):
            gjj = G[j][j] + ridge
            if gjj <= 0:
                continue
            r = c[j] - sum(G[j][k] * b[k] for k in range(n) if k != j)
            v = r / gjj
            if v < lo[j]:
                v = lo[j]
            moved = max(moved, abs(v - b[j]))
            b[j] = v
        if moved < 1e-9:
            break
    return b


class MoistureModel:
    def __init__(self, rooms: dict = None, units: list = None, fitted: str = None, report: dict = None):
        # rooms: id -> {"a", "c": {other: c}, "s0", "ss", "sc", "d": {unit: d}, "n"}
        self.rooms, self.units, self.fitted, self.report = rooms or {}, units or [], fitted, report or {}

    def deriv(self, pv: dict, pv_out: float, t: datetime, cooling: dict) -> dict:
        sn, cs = _harm(t)
        out = {}
        for rid, p in self.rooms.items():
            if rid not in pv:
                continue
            d = p["a"] * (pv_out - pv[rid])
            d += sum(c * (pv[k] - pv[rid]) for k, c in p["c"].items() if k in pv)
            d += p["s0"] + p["ss"] * sn + p["sc"] * cs
            d -= sum(dd * cooling.get(u, 0.0) for u, dd in p["d"].items())
            out[rid] = d
        return out

    def step(self, pv: dict, pv_out: float, t: datetime, cooling: dict, dt_h: float) -> dict:
        d = self.deriv(pv, pv_out, t, cooling)
        return {rid: max(0.0, pv[rid] + dt_h * d.get(rid, 0.0)) for rid in pv}

    def rollout(self, pv0: dict, pv_out_seq: list, times: list, cooling_seq: list, step_min: int = STEP_MIN) -> list:
        pv, out = dict(pv0), []
        for po, t, cl in zip(pv_out_seq, times, cooling_seq):
            pv = self.step(pv, po, t, cl, step_min / 60.0)
            out.append(dict(pv))
        return out

    def to_dict(self) -> dict:
        return {"rooms": self.rooms, "units": self.units, "fitted": self.fitted, "report": self.report}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d.get("rooms"), d.get("units"), d.get("fitted"), d.get("report"))


def series(rows: list, rooms: list, units: list, step_min: int = STEP_MIN) -> dict:
    """Resample minute-log rows (each with rooms.<id>.temp and .rh, outdoor, out_rh, units.<id>.action) onto a step_min grid, one
    sample per bucket from its last row: {"t": [datetime], "pv": {room: [..]}, "temp": {room: [..]}, "pv_out": [..],
    "cool": {unit: [share of the bucket's minutes the unit was cooling]}}. Buckets missing a reading are skipped."""
    ts, pv, tp, pvo, cool = [], {r: [] for r in rooms}, {r: [] for r in rooms}, [], {u: [] for u in units}

    def flush(buf):
        if not buf:
            return
        t_last, last = buf[-1]
        cur_pv, cur_tp = {}, {}
        for rid in rooms:
            v = (last.get("rooms") or {}).get(rid) or {}
            if v.get("temp") is None or v.get("rh") is None:
                return
            cur_pv[rid], cur_tp[rid] = pv_from(v["temp"], v["rh"]), v["temp"]
        orh = last.get("out_rh") if last.get("out_rh") is not None else (last.get("weather") or {}).get("humidity")
        if last.get("outdoor") is None or orh is None:
            return
        ts.append(t_last)
        for rid in rooms:
            pv[rid].append(cur_pv[rid])
            tp[rid].append(cur_tp[rid])
        pvo.append(pv_from(last["outdoor"], orh))
        for u in units:
            cool[u].append(sum(1 for _, r in buf if ((r.get("units") or {}).get(u) or {}).get("action") == "cooling") / len(buf))

    key, buf = None, []
    for row in rows:
        t = datetime.fromisoformat(row["t"])
        k = (t.date(), t.hour, t.minute // step_min)
        if key is not None and k != key:
            flush(buf)
            buf = []
        key = k
        buf.append((t, row))
    flush(buf)
    return {"t": ts, "pv": pv, "temp": tp, "pv_out": pvo, "cool": cool}


def fit(S: dict, rooms: list, units: list, ridge: float = 1e-4, min_steps: int = 200) -> Optional[MoistureModel]:
    """Fit every room from a resampled series (see series()). None if there is not enough data."""
    n = len(S["t"])
    if n < min_steps:
        return None
    dt_h = STEP_MIN / 60.0
    out = {}
    for rid in rooms:
        pv = S["pv"][rid]
        others = [k for k in rooms if k != rid]
        names = ["a"] + [("c", k) for k in others] + ["s0", "ss", "sc"] + [("d", u) for u in units]
        X, y = [], []
        for i in range(n - 1):
            d = (pv[i + 1] - pv[i]) / dt_h
            sn, cs = _harm(S["t"][i])
            row = [S["pv_out"][i] - pv[i]] + [S["pv"][k][i] - pv[i] for k in others] + [1.0, sn, cs] + [-S["cool"][u][i] for u in units]
            X.append(row)
            y.append(d)
        m = len(names)
        G = [[0.0] * m for _ in range(m)]
        c = [0.0] * m
        for row, yy in zip(X, y):
            for a in range(m):
                ra = row[a]
                if ra == 0.0:
                    continue
                c[a] += ra * yy
                Ga = G[a]
                for b in range(a, m):
                    Ga[b] += ra * row[b]
        for a in range(m):
            for b in range(a):
                G[a][b] = G[b][a]
        lo = []
        for nm in names:
            lo.append(-1e12 if nm in ("s0", "ss", "sc") else 0.0)
        # standardize: the columns are on very different scales (pressure differences of ~0.05 kPa, a constant of 1), and one ridge
        # strength should shrink them all alike
        sd = [max(1e-9, (G[j][j] / len(X)) ** 0.5) for j in range(m)]
        Gs = [[G[a][b] / (sd[a] * sd[b]) for b in range(m)] for a in range(m)]
        cs_ = [c[a] / sd[a] for a in range(m)]
        bs = nnls_gram(Gs, cs_, [lo[j] * sd[j] if lo[j] > -1e9 else lo[j] for j in range(m)], ridge=ridge * len(X))
        b = [bs[j] / sd[j] for j in range(m)]
        p = {"a": b[0], "c": {k: b[1 + j] for j, k in enumerate(others) if b[1 + j] > 1e-4}, "s0": b[1 + len(others)],
             "ss": b[2 + len(others)], "sc": b[3 + len(others)],
             "d": {u: b[4 + len(others) + j] for j, u in enumerate(units) if b[4 + len(others) + j] > 1e-5}, "n": len(X)}
        out[rid] = p
    return MoistureModel(out, list(units), datetime.now().astimezone().isoformat(timespec="seconds"), {"steps": n})


def evaluate(model: MoistureModel, S: dict, start: int, horizons=(60, 120), stride: int = 3) -> dict:
    """Multi-step rollouts from every `stride`th origin at or after `start`, with the true outdoors and cooling, against
    "nothing changes": RMSE in relative-humidity points (using each room's true temperature at the target time), per room and horizon."""
    rooms = list(model.rooms)
    n = len(S["t"])
    res = {h: {r: ([], []) for r in rooms} for h in horizons}
    hmax = max(horizons) // STEP_MIN
    for i in range(start, n - hmax - 1, stride):
        pv0 = {r: S["pv"][r][i] for r in rooms}
        seq = list(range(i + 1, i + hmax + 1))
        traj = model.rollout(pv0, [S["pv_out"][j] for j in seq], [S["t"][j] for j in seq],
                             [{u: S["cool"][u][j] for u in model.units} for j in seq])
        for h in horizons:
            k = h // STEP_MIN
            j = i + k
            for r in rooms:
                tt = S["temp"][r][j]
                true_rh = rh_from(tt, S["pv"][r][j])
                res[h][r][0].append(rh_from(tt, traj[k - 1][r]) - true_rh)
                res[h][r][1].append(rh_from(tt, pv0[r]) - true_rh)
    out = {}
    for h in horizons:
        out[h] = {}
        for r in rooms:
            e, b = res[h][r]
            if e:
                out[h][r] = (round((sum(x * x for x in e) / len(e)) ** 0.5, 2), round((sum(x * x for x in b) / len(b)) ** 0.5, 2), len(e))
    return out
