"""Fit the thermal model (model.py) to minute-log rows. Pure Python, so Home Assistant can refit it every night.

For each room, over windows of `window` minutes (overlapping, every `stride`):
    (T_end - T_start) / hours  =  loss*(out - T) + sum c_j*(T_j - T) + solar*sun + bias + sum heat_u*y_u+ - sum cool_u*y_u-
with every quantity averaged over the window and y_u+ / y_u- a unit's estimated heating / cooling output (0..1: on
when the log says heating / cooling, scaled by the inverter's response to its setpoint gap). Ridge least squares with
every coefficient except bias held >= 0, so the result stays physical; windows long enough to rise above sensor
quantization (H5100 0.18 F, unit sensors 0.9 F) matter more than more points.

Which room(s) each unit regulates on is detected from the log: the room whose reading the unit's own reported
temperature tracks (a remote sensor), else its zone's mean plus an offset.

The fit is checked by simulating the held-out last part of the log 2 h ahead from real starting temperatures with the
units' logged outputs, against "nothing changes". A model is only `usable` if it beats that for every room.
"""
from __future__ import annotations

from datetime import datetime

try:
    from . import model as hm
except ImportError:  # offline
    import model as hm  # type: ignore


def _mean(xs):
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def _median(xs):
    v = sorted(xs)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def series(rows: list, rooms: list, units: list):
    """Per-minute lists: times, temps[room], outdoor, sun, y[unit] (signed estimated output), unit readings."""
    d = hm.UnitModel()
    ts, out, sun = [], [], []
    T = {r: [] for r in rooms}
    Y = {u: [] for u in units}
    U = {u: [] for u in units}
    for row in rows:
        ts.append(datetime.fromisoformat(row["t"]))
        for r in rooms:
            v = (row.get("rooms", {}).get(r) or {}).get("temp")
            T[r].append(None if v is None else float(v))
        o = row.get("outdoor")
        if o is None:
            o = (row.get("weather") or {}).get("temp")
        out.append(None if o is None else float(o))
        sun.append(hm.solar_index((row.get("sun") or {}).get("elev"), (row.get("weather") or {}).get("cloud")))
        for u in units:
            x = row.get("units", {}).get(u) or {}
            act, sp, rd = x.get("action"), x.get("sp"), x.get("temp")
            U[u].append(None if rd is None else float(rd))
            if act in ("heating", "cooling"):
                y = 1.0
                if sp is not None and rd is not None:
                    err = (sp - rd) if act == "heating" else (rd - sp)
                    y = max(d.ymin, min(1.0, d.k0 + d.k1 * err))
                Y[u].append(y if act == "heating" else -y)
            else:
                Y[u].append(0.0)
    return ts, T, out, sun, Y, U


def detect_sensor_rooms(T: dict, U: dict, zones: dict) -> dict:
    """unit -> (sensor rooms, offset). A unit fed a room's sensor (remote temperature) reports within about half a
    degree of that room (its 0.5 C rounding); otherwise it reads its own air, taken as its zone's mean plus a bias."""
    res = {}
    for uid, readings in U.items():
        best = None
        for rid, temps in T.items():
            diffs = [a - b for a, b in zip(readings, temps) if a is not None and b is not None]
            if len(diffs) < 60:
                continue
            mad = _median([abs(x) for x in diffs])
            if best is None or mad < best[0]:
                best = (mad, rid, _median(diffs))
        if best is not None and best[0] <= 0.6:
            res[uid] = ([best[1]], round(best[2], 2))
            continue
        zone = [r for r in zones.get(uid, []) if r in T]
        diffs = []
        for k, a in enumerate(readings):
            m = _mean([T[r][k] for r in zone])
            if a is not None and m is not None:
                diffs.append(a - m)
        res[uid] = (zone, round(_median(diffs), 2) if diffs else 0.0)
    return res


def solve(A: list, b: list, free: set, lam: float = 1e-3, iters: int = 3000) -> list:
    """min |Ax - b|^2 + lam*|x|^2 (standardized columns) with x_j >= 0 unless j in `free`: projected coordinate
    descent on the normal equations (a dozen or two columns, so plain Python is quick enough)."""
    p, n = len(A[0]), len(A)
    scale = [max(1e-12, (sum(row[j] ** 2 for row in A) / n) ** 0.5) for j in range(p)]
    G = [[0.0] * p for _ in range(p)]
    h = [0.0] * p
    for row, y in zip(A, b):
        z = [row[j] / scale[j] for j in range(p)]
        for i in range(p):
            zi = z[i]
            if zi == 0.0:
                continue
            h[i] += zi * y
            Gi = G[i]
            for j in range(p):
                Gi[j] += zi * z[j]
    for i in range(p):
        h[i] /= n
        for j in range(p):
            G[i][j] /= n
        G[i][i] += lam
    x = [0.0] * p
    for _ in range(iters):
        delta = 0.0
        for j in range(p):
            Gj = G[j]
            v = (h[j] - sum(Gj[k] * x[k] for k in range(p)) + Gj[j] * x[j]) / Gj[j]
            if j not in free and v < 0.0:
                v = 0.0
            delta = max(delta, abs(v - x[j]))
            x[j] = v
        if delta < 1e-10:
            break
    return [x[j] / scale[j] for j in range(p)]


def fit(rows: list, layout: dict, window: int = 30, stride: int = 10, holdout: float = 0.25, lam: float = 1e-3):
    """-> (HouseModel, report). layout: {"rooms": {id: {"units": [...], "neighbors": [...]}},
    "units": {id: {"sensor_rooms": [...] (optional: detected when missing), "zone": [...]}}}"""
    rooms, units = list(layout["rooms"]), list(layout["units"])
    ts, T, out, sun, Y, U = series(rows, rooms, units)
    n = len(ts)
    split = int(n * (1 - holdout))
    m = hm.HouseModel()
    report = {}
    # Rooms with (almost) no readings yet - a sensor that isn't installed, or arrived only today - are left out, as
    # rooms and as anyone's neighbour; otherwise a missing neighbour would void every window of every room.
    min_cover = 0.5
    cover = {r: (sum(v is not None for v in T[r]) / n if n else 0.0) for r in rooms}
    for r in rooms:
        if cover[r] < min_cover:
            report[r] = {"skipped": f"no data yet ({cover[r]:.0%} of the time has readings)"}
    T = {r: v for r, v in T.items() if cover[r] >= min_cover}
    for rid, spec in layout["rooms"].items():
        if rid not in T:
            continue
        nbrs = [j for j in spec.get("neighbors", []) if j in T and j != rid]
        us = [u for u in spec.get("units", units) if u in Y]
        cols = ["loss"] + [f"c:{j}" for j in nbrs] + ["solar", "bias"] + [f"h:{u}" for u in us] + [f"k:{u}" for u in us]
        A, b = [], []
        for a in range(0, max(0, split - window), stride):
            z = a + window
            Ti = T[rid][a:z]
            if Ti[0] is None or Ti[-1] is None or sum(v is None for v in Ti) > 0.2 * window:
                continue
            o = _mean(out[a:z])
            nb = [_mean(T[j][a:z]) for j in nbrs]
            if o is None or any(v is None for v in nb):
                continue
            Tm = _mean(Ti)
            row = [o - Tm] + [v - Tm for v in nb] + [sum(sun[a:z]) / window, 1.0]
            row += [sum(max(0.0, y) for y in Y[u][a:z]) / window for u in us]
            row += [sum(min(0.0, y) for y in Y[u][a:z]) / window for u in us]  # negative: cooling
            A.append(row)
            b.append((Ti[-1] - Ti[0]) / (window / 60))
        if len(A) < 3 * len(cols):
            report[rid] = {"skipped": f"only {len(A)} usable windows (need {3 * len(cols)})"}
            continue
        x = solve(A, b, free={cols.index("bias")}, lam=lam)
        pred = [sum(r[j] * x[j] for j in range(len(x))) for r in A]
        mb = sum(b) / len(b)
        ss = sum((y - mb) ** 2 for y in b)
        r2 = 1 - sum((y - q) ** 2 for y, q in zip(b, pred)) / ss if ss > 0 else 0.0
        p = dict(zip(cols, x))
        m.rooms[rid] = hm.RoomModel(loss=round(p["loss"], 4), solar=round(p["solar"], 3), bias=round(p["bias"], 3),
                                    heat={u: round(p[f"h:{u}"], 3) for u in us},
                                    cool={u: round(p[f"k:{u}"], 3) for u in us},
                                    coupling={j: round(p[f"c:{j}"], 4) for j in nbrs})
        report[rid] = {"windows": len(A), "r2_rate": round(r2, 3)}
    zones = {u: spec.get("zone", []) for u, spec in layout["units"].items()}
    detected = detect_sensor_rooms(T, U, zones)
    for uid, spec in layout["units"].items():
        sr, off = (spec["sensor_rooms"], 0.0) if spec.get("sensor_rooms") is not None else detected.get(uid, ([], 0.0))
        m.units[uid] = hm.UnitModel(sensor_rooms=[r for r in sr if r in m.rooms], sensor_offset=off)
    report["holdout"] = validate(m, ts, T, out, sun, Y, split)
    hold = [v for v in report["holdout"].values()]
    usable = bool(m.rooms) and bool(hold) and all(
        v["model_rmse_F"] is not None and v["no_change_rmse_F"] is not None and v["model_rmse_F"] < v["no_change_rmse_F"]
        for v in hold)
    m.meta = {"fitted": datetime.now().astimezone().isoformat(timespec="seconds"), "usable": usable, "rows": n,
              "window_min": window, "from": ts[0].isoformat() if ts else None,
              "to": ts[-1].isoformat() if ts else None, "report": report}
    return m, report


def validate(m: hm.HouseModel, ts, T, out, sun, Y, split: int, horizon_min: int = 120, step_min: int = 5, every_min: int = 30):
    """2 h ahead on the held-out part with the units at their logged output: RMSE of the model vs 'nothing changes'."""
    errs = {r: [] for r in m.rooms}
    base = {r: [] for r in m.rooms}
    n = len(ts)
    for a in range(split, n - horizon_min, every_min):
        start = {r: T[r][a] for r in m.rooms}
        if any(v is None for v in start.values()):
            continue
        cur = dict(start)
        for k in range(a, a + horizon_min, step_min):
            seg = slice(k, k + step_min)
            o = _mean(out[seg])
            y = {u: sum(Y[u][seg]) / step_min for u in Y}
            d = hm.derivative(m, cur, o, sum(sun[seg]) / step_min, y)
            cur = {r: cur[r] + d.get(r, 0.0) * step_min / 60 for r in cur}
        z = a + horizon_min
        for r in m.rooms:
            if T[r][z] is not None:
                errs[r].append(cur[r] - T[r][z])
                base[r].append(start[r] - T[r][z])

    def rmse(v):
        return round((sum(x * x for x in v) / len(v)) ** 0.5, 2) if v else None

    def bias(v):
        return round(sum(v) / len(v), 2) if v else None

    return {r: {"model_rmse_F": rmse(errs[r]), "model_bias_F": bias(errs[r]), "no_change_rmse_F": rmse(base[r]),
                "n": len(errs[r])} for r in m.rooms}


def layout_for(rooms: list, units: list) -> dict:
    """Fit layout from the controller's config. Every room with a sensor is modelled, with every other such room as a
    possible neighbour and every unit as a possible influence (non-negative ridge sorts out which matter). A room that
    reads its own unit's sensor is modelled (its unit regulates on it) but isn't anyone's neighbour: it measures the
    unit's airflow, not a room. Unit sensor rooms are detected from the log."""
    unit_entity = {u.id: u.entity for u in units}
    sensed = [r for r in rooms if r.sensor]
    own = {r.id for r in sensed if r.sensor == unit_entity.get(r.unit)}
    lay = {"rooms": {}, "units": {}}
    for r in sensed:
        lay["rooms"][r.id] = {"units": [u.id for u in units],
                              "neighbors": [o.id for o in sensed if o.id != r.id and o.id not in own]}
    for u in units:
        lay["units"][u.id] = {"zone": [r.id for r in sensed if r.unit == u.id]}
    return lay
