"""Fit the thermal model (model.py) to minute-log rows. Pure Python, so Home Assistant can refit it every night.

For each room, over windows of `window` minutes (overlapping, every `stride`):
    (T_end - T_start) / hours  =  loss*(out - T) + sum c_j*(T_j - T) + solar*sun + bias + sum heat_u*y_u+ - sum cool_u*y_u-
with every quantity averaged over the window and y_u+ / y_u- a unit's estimated heating / cooling output (0..1: on
when the log says heating / cooling, scaled by the inverter's response to its setpoint gap). Ridge least squares with
every coefficient except bias held >= 0, so the result stays physical; windows long enough to rise above sensor
quantization (H5100 0.18 F, unit sensors 0.9 F) matter more than more points.

Each unit's lag (how long its effect on the rooms trails its output: a warm coil and ducts keep heating after the
compressor stops) is chosen from a few candidates by how well the training windows are explained.

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


LAGS_MIN = (0, 5, 10, 20, 30)  # candidate unit lags


def lagged(y: list, lag_min: float) -> list:
    """A minute series of unit output seen through a first-order lag of `lag_min` minutes."""
    if lag_min <= 0:
        return list(y)
    a = 1.0 - pow(2.718281828459045, -1.0 / lag_min)
    out, z = [], 0.0
    for v in y:
        z += (v - z) * a
        out.append(z)
    return out


def with_lags(Y: dict, m: hm.HouseModel) -> dict:
    """The units' output series through the model's fitted lags."""
    return {u: lagged(v, (m.units[u].lag_h * 60) if u in m.units else 0.0) for u, v in Y.items()}


def detect_sensor_rooms(T: dict, U: dict, zones: dict) -> dict:
    """unit -> (sensor rooms, offset). A unit fed a room's sensor (remote temperature) reports within about half a
    degree of that room (its 0.5 C rounding); otherwise it reads its own air, taken as its zone's mean plus a bias."""
    res = {}
    for uid, readings in U.items():
        best = None
        own = set(zones.get(uid, []))
        for rid, temps in T.items():
            diffs = [a - b for a, b in zip(readings, temps) if a is not None and b is not None]
            if len(diffs) < 60:
                continue
            mad = _median([abs(x) for x in diffs])
            # With rooms held near the same temperature several can track the unit's reading; its own zone's wins.
            key = (mad > 0.6, rid not in own, mad)
            if best is None or key < best[3]:
                best = (mad, rid, _median(diffs), key)
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
    # A room's readings from before its `since` (a sensor that was replaced or moved) aren't the room: leave them out.
    for rid, spec in layout["rooms"].items():
        cut = spec.get("since")
        if cut and rid in T:
            cut = datetime.fromisoformat(cut)
            T[rid] = [None if t < cut else v for t, v in zip(ts, T[rid])]
    split = int(n * (1 - holdout))
    m = hm.HouseModel()
    report = {}
    # A room is fitted once it has `min_hours` of readings (a new or moved sensor joins after a day, not after half the
    # fit window); it's anyone's neighbour only once it covers half the log, since a neighbour missing from a window
    # voids that window for every room it neighbours.
    min_cover, min_hours = 0.5, 24
    have = {r: sum(v is not None for v in T[r]) for r in rooms}
    cover = {r: (have[r] / n if n else 0.0) for r in rooms}
    for r in rooms:
        if have[r] < min_hours * 60:
            report[r] = {"skipped": f"not enough data yet ({have[r] / 60:.0f} of {min_hours} h of readings)"}
    T = {r: v for r, v in T.items() if have[r] >= min_hours * 60}
    neighbour_ok = {r for r in T if cover[r] >= min_cover}

    def rooms_fit(Yl):
        """Fit every room with the units' outputs `Yl` -> (room models, per-room report, training SSE)."""
        rooms_m, rep, sse = {}, {}, 0.0
        for rid, spec in layout["rooms"].items():
            if rid not in T:
                continue
            nbrs = [j for j in spec.get("neighbors", []) if j in neighbour_ok and j != rid]
            us = [u for u in spec.get("units", units) if u in Yl]
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
                row += [sum(max(0.0, y) for y in Yl[u][a:z]) / window for u in us]
                row += [sum(min(0.0, y) for y in Yl[u][a:z]) / window for u in us]  # negative: cooling
                A.append(row)
                b.append((Ti[-1] - Ti[0]) / (window / 60))
            if len(A) < 3 * len(cols):
                rep[rid] = {"skipped": f"only {len(A)} usable windows (need {3 * len(cols)})"}
                continue
            x = solve(A, b, free={cols.index("bias")}, lam=lam)
            pred = [sum(r[j] * x[j] for j in range(len(x))) for r in A]
            mb = sum(b) / len(b)
            ss = sum((y - mb) ** 2 for y in b)
            res = sum((y - q) ** 2 for y, q in zip(b, pred))
            sse += res
            r2 = 1 - res / ss if ss > 0 else 0.0
            p = dict(zip(cols, x))
            rooms_m[rid] = hm.RoomModel(loss=round(p["loss"], 4), solar=round(p["solar"], 3), bias=round(p["bias"], 3),
                                        heat={u: round(p[f"h:{u}"], 3) for u in us},
                                        cool={u: round(p[f"k:{u}"], 3) for u in us},
                                        coupling={j: round(p[f"c:{j}"], 4) for j in nbrs})
            rep[rid] = {"windows": len(A), "r2_rate": round(r2, 3)}
        return rooms_m, rep, sse

    # Each unit's lag: coordinate search over the candidates, by how well the training windows are explained.
    lags = {u: 0 for u in units}
    cache = {}

    def trial(lg):
        key = tuple(sorted(lg.items()))
        if key not in cache:
            cache[key] = rooms_fit({u: lagged(Y[u], lg.get(u, 0)) for u in Y})
        return cache[key]

    if any(Y[u] and any(Y[u]) for u in Y):
        for u in units:
            best = min(LAGS_MIN, key=lambda L: trial({**lags, u: L})[2])
            lags[u] = best
    fitted, rep, _ = trial(lags)
    m.rooms.update(fitted)
    report.update(rep)
    zones = {u: spec.get("zone", []) for u, spec in layout["units"].items()}
    detected = detect_sensor_rooms(T, U, zones)
    for uid, spec in layout["units"].items():
        sr, off = (spec["sensor_rooms"], 0.0) if spec.get("sensor_rooms") is not None else detected.get(uid, ([], 0.0))
        m.units[uid] = hm.UnitModel(sensor_rooms=[r for r in sr if r in m.rooms], sensor_offset=off,
                                    lag_h=round(lags.get(uid, 0) / 60, 4))
    report["lags_min"] = lags
    report["holdout"] = validate(m, ts, T, out, sun, with_lags(Y, m), split)
    report["split"] = ts[split].isoformat() if 0 <= split < n else None  # trained before this, tested after
    # Every room with enough held-out checks must beat "nothing changes"; at least one must have been checked. A room
    # whose sensor only just started may have none yet: the live accuracy gate (mpc.readiness) still covers it.
    hold = [v for v in report["holdout"].values() if v["n"] >= 5]
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
        start = {r: T[r][a] for r in m.rooms if T[r][a] is not None}  # rooms without a reading sit this one out
        if not start:
            continue
        cur = dict(start)
        for k in range(a, a + horizon_min, step_min):
            seg = slice(k, k + step_min)
            o = _mean(out[seg])
            y = {u: sum(Y[u][seg]) / step_min for u in Y}
            d = hm.derivative(m, cur, o, sum(sun[seg]) / step_min, y)
            cur = {r: cur[r] + d.get(r, 0.0) * step_min / 60 for r in cur}
        z = a + horizon_min
        for r in start:
            if T[r][z] is not None:
                errs[r].append(cur[r] - T[r][z])
                base[r].append(start[r] - T[r][z])

    def rmse(v):
        return round((sum(x * x for x in v) / len(v)) ** 0.5, 2) if v else None

    def bias(v):
        return round(sum(v) / len(v), 2) if v else None

    return {r: {"model_rmse_F": rmse(errs[r]), "model_bias_F": bias(errs[r]), "no_change_rmse_F": rmse(base[r]),
                "n": len(errs[r])} for r in m.rooms}


def layout_for(rooms: list, units: list, since: dict = None) -> dict:
    """Fit layout from the controller's config. Every room with a sensor is modelled, with every other such room as a
    possible neighbour and every unit as a possible influence (non-negative ridge sorts out which matter). A room that
    reads its own unit's sensor is modelled (its unit regulates on it) but isn't anyone's neighbour: it measures the
    unit's airflow, not a room. Unit sensor rooms are detected from the log. `since`: room id -> ISO time before
    which that room's readings are left out (its sensor was replaced or moved)."""
    unit_entity = {u.id: u.entity for u in units}
    sensed = [r for r in rooms if r.sensor]
    own = {r.id for r in sensed if r.sensor == unit_entity.get(r.unit)}
    lay = {"rooms": {}, "units": {}}
    for r in sensed:
        lay["rooms"][r.id] = {"units": [u.id for u in units],
                              "neighbors": [o.id for o in sensed if o.id != r.id and o.id not in own]}
        if (since or {}).get(r.id):
            lay["rooms"][r.id]["since"] = since[r.id]
    for u in units:
        lay["units"][u.id] = {"zone": [r.id for r in sensed if r.unit == u.id]}
    return lay


def backtest(m: hm.HouseModel, rows: list, horizons=(60, 120), every_min: int = 30, step_min: int = 5) -> dict:
    """The fitted model run over logged history: from the real readings every `every_min`, replay what the units, the
    weather and the sun did and predict each room `horizons` minutes ahead. -> {room: {"60": [[iso, F]], ...}}, each
    point at the time it's a prediction for, so it plots against the actual reading at that moment."""
    rooms, units = list(m.rooms), sorted({u for row in rows for u in row.get("units", {})})
    ts, T, out, sun, Y, _ = series(rows, rooms, units)
    Y = with_lags(Y, m)
    res = {r: {str(h): [] for h in horizons} for r in rooms}
    hmax = max(horizons)
    for a in range(0, len(ts) - hmax, every_min):
        cur = {r: T[r][a] for r in rooms if T[r][a] is not None}
        if not cur:
            continue
        for k in range(a, a + hmax, step_min):
            seg = slice(k, k + step_min)
            d = hm.derivative(m, cur, _mean(out[seg]), sum(sun[seg]) / step_min, {u: sum(Y[u][seg]) / step_min for u in Y})
            cur = {r: cur[r] + d.get(r, 0.0) * step_min / 60 for r in cur}
            done = k + step_min - a
            if done in horizons:
                for r in cur:
                    res[r][str(done)].append([ts[a + done].isoformat(), round(cur[r], 2)])
    return res
