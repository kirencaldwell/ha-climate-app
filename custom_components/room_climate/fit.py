"""Fit the thermal model (model.py) to minute-log rows. Pure Python, so Home Assistant can refit it every night.

For each room, over windows of `window` minutes (overlapping, every `stride`):
    (T_end - T_start) / hours  =  loss*(out - T) + sum c_j*(T_j - T) + solar*sun_h + sum_k sun_k*sun_k (east/south/west
                                  walls) + bias + sum heat_u*y_u+ - sum cool_u*y_u-
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


POWER_REF_W = 3000.0  # outdoor unit draw counted as full compressor drive (about its most, heating, at 50 F out)


def compressor_drive(extra: dict):
    """0..1 drive of the shared compressor from the units' telemetry, or None when it isn't reported: its frequency
    over 100 Hz when a unit reports one above 0 (these CN105 units report 0 always, so 0 isn't a reading), else the
    largest input power a unit reports over POWER_REF_W (one of the units reports the whole outdoor unit's draw; the
    others report a few watts or nothing)."""
    hz = [v for k, v in (extra or {}).items() if "compressor_frequency" in k and isinstance(v, (int, float)) and v > 0]
    if hz:
        return min(1.0, max(hz) / 100.0)
    pw = [v for k, v in (extra or {}).items() if k.endswith("input_power") and isinstance(v, (int, float))]
    if pw:
        return min(1.0, max(0.0, max(pw)) / POWER_REF_W)
    return None


def _gap(x: dict):
    """A logged unit's setpoint gap in the direction it's working (None when idle or unknown)."""
    act, sp, rd = x.get("action"), x.get("sp"), x.get("temp")
    if act not in ("heating", "cooling") or sp is None or rd is None:
        return None
    return (sp - rd) if act == "heating" else (rd - sp)


OUT_MIN_SOLO = 60  # minutes of a unit running alone with the outdoor unit's power logged before its output is learned
OUT_SKIP_MIN = 3  # first minutes of a run left out: the compressor is still spinning up


def fit_outputs(rows: list, units: list) -> dict:
    """Each unit's output while running, learned from the outdoor unit's measured draw (compressor_drive) in minutes
    when it runs alone: output = clip(k0 + k1 * gap, ymin, 1), k1 >= 0 (gap: setpoint past its reading in the
    direction it works). -> {unit: {"k0", "k1", "ymin", "minutes"}} for units with enough such minutes."""
    pts = {u: [] for u in units}
    since = {u: None for u in units}  # minute index each unit's current run started
    for k, row in enumerate(rows):
        xs = {u: (row.get("units", {}).get(u) or {}) for u in units}
        run = [u for u in units if xs[u].get("action") in ("heating", "cooling")]
        for u in units:
            since[u] = (since[u] if since[u] is not None else k) if u in run else None
        if len(run) != 1:
            continue
        u = run[0]
        d, g = compressor_drive(row.get("extra")), _gap(xs[u])
        if d is None or g is None or k - since[u] < OUT_SKIP_MIN:
            continue
        pts[u].append((max(-1.0, min(4.0, g)), d))
    res = {}
    for u, v in pts.items():
        if len(v) < OUT_MIN_SOLO:
            continue
        n = len(v)
        mg, md = sum(g for g, _ in v) / n, sum(d for _, d in v) / n
        sgg = sum((g - mg) ** 2 for g, _ in v)
        k1 = max(0.0, sum((g - mg) * (d - md) for g, d in v) / sgg) if sgg > 1e-9 else 0.0
        k0 = md - k1 * mg
        ds = sorted(d for _, d in v)
        ymin = max(0.02, min(max(k0, 0.02), ds[n // 10]))
        res[u] = {"k0": round(max(0.02, k0), 3), "k1": round(k1, 3), "ymin": round(ymin, 3), "minutes": n}
    return res


def series(rows: list, rooms: list, units: list, outputs: dict = None):
    """Per-minute lists: times, temps[room], outdoor, sun, y[unit] (signed output, before any fan scaling), unit
    readings, fan mode[unit], leak drive[unit].

    A running unit's output is the outdoor unit's measured draw (compressor_drive) when it's logged, split between the
    units running by what each would give on its own; otherwise what it would give by its output rule (`outputs`:
    unit -> UnitModel with learned k0/k1/ymin, else the defaults), so minutes with and without the power logged are on
    the same scale."""
    ts, out, sun = [], [], []
    T = {r: [] for r in rooms}
    Y = {u: [] for u in units}
    U = {u: [] for u in units}
    F = {u: [] for u in units}
    L = {u: [] for u in units}
    rule = {u: (outputs or {}).get(u) or hm.UnitModel() for u in units}
    for row in rows:
        ts.append(datetime.fromisoformat(row["t"]))
        for r in rooms:
            v = (row.get("rooms", {}).get(r) or {}).get("temp")
            T[r].append(None if v is None else float(v))
        o = row.get("outdoor")
        if o is None:
            o = (row.get("weather") or {}).get("temp")
        out.append(None if o is None else float(o))
        sn = row.get("sun") or {}
        if sn.get("ghi") is not None:  # estimated sunshine (HRRR) when logged; else from the forecast's cloud cover
            sun.append(hm.sun_from_irradiance(sn.get("elev"), sn.get("az"), sn.get("ghi"), sn.get("dni"), sn.get("dhi")))
        else:
            sun.append(hm.sun_vector(sn.get("elev"), sn.get("az"), (row.get("weather") or {}).get("cloud")))
        drive = compressor_drive(row.get("extra"))
        est = {}
        for u in units:
            x = row.get("units", {}).get(u) or {}
            rd = x.get("temp")
            U[u].append(None if rd is None else float(rd))
            F[u].append(x.get("fan"))
            if x.get("action") in ("heating", "cooling"):
                g, d = _gap(x), rule[u]
                est[u] = (1.0 if g is None else max(d.ymin, min(1.0, d.k0 + d.k1 * g)), x["action"] == "heating")
        tot = sum(y for y, _ in est.values())
        for u in units:
            if u not in est:
                Y[u].append(0.0)
                continue
            y, heating = est[u]
            if drive is not None and tot > 0:
                y = drive * y / tot
            Y[u].append(y if heating else -y)
        # Leak drive for idle units (model.leak_inputs): how hard the shared compressor works, else the other units'
        # output in that mode. See compressor_drive.
        modes = {u: (row.get("units", {}).get(u) or {}).get("state") for u in units}
        running = {u: u in est for u in units}
        lk = hm.leak_inputs(modes, running, {u: Y[u][-1] for u in units}, units, drive)
        for u in units:
            L[u].append(lk[u])
    return ts, T, out, sun, Y, U, F, L


FAN_REF = ("auto", "high")  # fan modes counted as full output: the others' factors are relative to these
FAN_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)  # relative to the reference speed


def scaled(Y: dict, F: dict, scales: dict) -> dict:
    """Unit outputs with each minute's fan factor applied (scales: unit -> {fan mode: factor})."""
    return {u: [y * scales.get(u, {}).get(f, 1.0) if y else y for y, f in zip(v, F.get(u, [None] * len(v)))]
            for u, v in Y.items()}


def effective(Y: dict, F: dict, m: hm.HouseModel) -> dict:
    """The units' outputs as the model sees them: fan factors, then lags."""
    return with_lags(scaled(Y, F, {u: x.fan_scale for u, x in m.units.items()}), m)


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


def _lin_solve(A: list, b: list) -> list:
    """Solve A x = b (small, dense) by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-15:
            return [0.0] * n
        M[c], M[piv] = M[piv], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                Mr, Mc = M[r], M[c]
                for k in range(c, n + 1):
                    Mr[k] -= f * Mc[k]
    x = [0.0] * n
    for c in range(n - 1, -1, -1):
        x[c] = (M[c][n] - sum(M[c][k] * x[k] for k in range(c + 1, n))) / M[c][c]
    return x


OE_HORIZON_MIN, OE_STEP_MIN, OE_EVERY_MIN, OE_ITERS = 120, 5, 30, 25
OE_PRIOR = 0.01  # pull toward the slope fit, relative to each parameter's curvature there: keeps poorly seen ones put


def refine_paths(rid: str, rm: hm.RoomModel, T: dict, out: list, sun: list, Ye: dict, Lk: dict, end: int,
                 horizon: int = OE_HORIZON_MIN, step: int = OE_STEP_MIN, every: int = OE_EVERY_MIN,
                 iters: int = OE_ITERS):
    """Refine one room's parameters (from the slope fit) so that its simulated paths match its readings.

    From every `every` minutes of its training part (before `end`), the room is simulated `horizon` minutes ahead the
    way validate does (Euler steps of `step` minutes, inputs averaged over each step), with its neighbours, the
    weather, the sun and the units at what was logged, and its own temperature simulated. The squared miss at every
    step along every path is minimised by projected Levenberg-Marquardt with forward sensitivities (the model is linear
    in its parameters but a path isn't: the room's own temperature feeds back through loss and coupling). Parameters
    stay >= 0 except bias. -> (room model, report)."""
    nbrs, us, own = list(rm.coupling), list(rm.heat), list(rm.leak)
    names = (["loss"] + [f"c:{j}" for j in nbrs] + ["solar"] + [f"sun:{k}" for k in hm.WALLS] + ["bias"]
             + [f"h:{u}" for u in us] + [f"k:{u}" for u in us] + [f"l:{u}" for u in own])
    th0 = ([rm.loss] + [rm.coupling[j] for j in nbrs] + [rm.solar] + [rm.sun_dir.get(k, 0.0) for k in hm.WALLS]
           + [rm.bias] + [rm.heat[u] for u in us] + [rm.cool[u] for u in us] + [rm.leak[u] for u in own])
    p, nb_n, bias_i = len(names), len(nbrs), names.index("bias")
    h = step / 60.0
    Tr = T[rid]
    segs = []  # [(start temp, [(outdoor, [neighbour temps], other features, target or None)])]
    for a in range(0, max(0, end - horizon), every):
        if Tr[a] is None:
            continue
        steps = []
        for k in range(a, a + horizon, step):
            sl = slice(k, k + step)
            o = _mean(out[sl])
            nb = [_mean(T[j][sl]) for j in nbrs]
            if o is None or any(v is None for v in nb):
                break
            sv = hm.mean_sun(sun[sl])
            g = ([sv.get("h", 0.0)] + [sv.get(w, 0.0) for w in hm.WALLS] + [1.0]
                 + [sum(max(0.0, y) for y in Ye[u][sl]) / step for u in us]
                 + [sum(min(0.0, y) for y in Ye[u][sl]) / step for u in us]
                 + [sum(Lk[u][sl]) / step for u in own])
            steps.append((o, nb, g, Tr[k + step] if k + step < len(Tr) else None))
        if sum(st[3] is not None for st in steps) >= 3:
            segs.append((Tr[a], steps))
    if not segs:
        return rm, {"paths": 0}

    def evaluate(th, need_j=True):
        """(cost, J'J, J'r, residual count) of the paths at th."""
        H = [[0.0] * p for _ in range(p)] if need_j else None
        gr = [0.0] * p
        cost, cnt = 0.0, 0
        a_self = th[0] + sum(th[1:1 + nb_n])
        for t0, steps in segs:
            t = t0
            S = [0.0] * p
            for o, nb, g, tgt in steps:
                phi = [o - t] + [v - t for v in nb] + g
                f = sum(th[j] * phi[j] for j in range(p))
                if need_j:
                    S = [S[j] + h * (phi[j] - a_self * S[j]) for j in range(p)]
                t = t + h * f
                if abs(t) > 500:
                    return float("inf"), H, gr, cnt
                if tgt is None:
                    continue
                r = t - tgt
                cost += r * r
                cnt += 1
                if need_j:
                    for i in range(p):
                        Si = S[i]
                        if Si:
                            gr[i] += Si * r
                            Hi = H[i]
                            for j in range(i, p):
                                Hi[j] += Si * S[j]
        if need_j:
            for i in range(p):
                for j in range(i):
                    H[i][j] = H[j][i]
        return cost, H, gr, cnt

    c0, H0, g0, cnt = evaluate(th0)
    if not cnt or c0 == float("inf"):
        return rm, {"paths": len(segs)}
    prior = [OE_PRIOR * max(H0[j][j], 1e-9) for j in range(p)]

    def total(c, th):
        return c + sum(prior[j] * (th[j] - th0[j]) ** 2 for j in range(p))

    th, c, H, gr = list(th0), c0, H0, g0
    best = total(c, th)
    mu, it = 1e-3, 0
    for it in range(1, iters + 1):
        grad = [gr[j] + prior[j] * (th[j] - th0[j]) for j in range(p)]
        free = [j for j in range(p) if j == bias_i or th[j] > 1e-12 or grad[j] < 0]
        A = [[H[i][j] + ((prior[i] + mu * max(H[i][i], 1e-9)) if i == j else 0.0) for j in free] for i in free]
        dx = _lin_solve(A, [-grad[i] for i in free])
        cand = list(th)
        for i, d in zip(free, dx):
            cand[i] = th[i] + d if i == bias_i else max(0.0, th[i] + d)
        cn, Hn, gn, _ = evaluate(cand)
        tn = total(cn, cand)
        if tn < best:
            gain = (best - tn) / best if best > 0 else 0.0
            th, c, H, gr, best = cand, cn, Hn, gn, tn
            mu = max(1e-7, mu / 3)
            if gain < 1e-4:
                break
        else:
            mu *= 4
            if mu > 1e4:
                break
    q = dict(zip(names, th))
    new = hm.RoomModel(loss=round(q["loss"], 4), solar=round(q["solar"], 3), bias=round(q["bias"], 3),
                       sun_dir={k: round(q[f"sun:{k}"], 3) for k in hm.WALLS},
                       heat={u: round(q[f"h:{u}"], 3) for u in us}, cool={u: round(q[f"k:{u}"], 3) for u in us},
                       coupling={j: round(q[f"c:{j}"], 4) for j in nbrs}, leak={u: round(q[f"l:{u}"], 3) for u in own})
    return new, {"paths": len(segs), "iters": it, "path_rmse_slope_fit_F": round((c0 / cnt) ** 0.5, 2),
                 "path_rmse_F": round((c / cnt) ** 0.5, 2)}


def fit(rows: list, layout: dict, window: int = 30, stride: int = 10, holdout: float = 0.25, lam: float = 1e-3,
        paths: bool = True):
    """-> (HouseModel, report). layout: {"rooms": {id: {"units": [...], "neighbors": [...]}},
    "units": {id: {"sensor_rooms": [...] (optional: detected when missing), "zone": [...]}}}"""
    rooms, units = list(layout["rooms"]), list(layout["units"])
    learned_out = fit_outputs(rows, units)
    rule = {u: hm.UnitModel(k0=v["k0"], k1=v["k1"], ymin=v["ymin"]) for u, v in learned_out.items()}
    ts, T, out, sun, Y, U, F, Lk = series(rows, rooms, units, rule)
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
    # Each room trains on the first part of its own readings and is tested on the rest: a room whose sensor started
    # recently has all its readings near the end of the log, inside a split made by time alone.
    splits = {}
    for r, v in T.items():
        idx = [k for k, x in enumerate(v) if x is not None]
        splits[r] = idx[min(len(idx) - 1, int(len(idx) * (1 - holdout)))] if idx else n

    def rooms_fit(Yl):
        """Fit every room with the units' outputs `Yl` -> (room models, per-room report, training SSE)."""
        rooms_m, rep, sse = {}, {}, 0.0
        for rid, spec in layout["rooms"].items():
            if rid not in T:
                continue
            nbrs = [j for j in spec.get("neighbors", []) if j in neighbour_ok and j != rid]
            us = [u for u in spec.get("units", units) if u in Yl]
            own = [u for u in units if rid in layout["units"].get(u, {}).get("zone", [])]  # units serving this room
            cols = (["loss"] + [f"c:{j}" for j in nbrs] + ["solar"] + [f"sun:{k}" for k in hm.WALLS] + ["bias"]
                    + [f"h:{u}" for u in us] + [f"k:{u}" for u in us] + [f"l:{u}" for u in own])
            A, b = [], []
            for a in range(0, max(0, splits[rid] - window), stride):
                z = a + window
                Ti = T[rid][a:z]
                if Ti[0] is None or Ti[-1] is None or sum(v is None for v in Ti) > 0.2 * window:
                    continue
                o = _mean(out[a:z])
                nb = [_mean(T[j][a:z]) for j in nbrs]
                if o is None or any(v is None for v in nb):
                    continue
                Tm = _mean(Ti)
                sv = hm.mean_sun(sun[a:z])
                row = [o - Tm] + [v - Tm for v in nb] + [sv.get("h", 0.0)] + [sv.get(k, 0.0) for k in hm.WALLS] + [1.0]
                row += [sum(max(0.0, y) for y in Yl[u][a:z]) / window for u in us]
                row += [sum(min(0.0, y) for y in Yl[u][a:z]) / window for u in us]  # negative: cooling
                row += [sum(Lk[u][a:z]) / window for u in own]
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
                                        sun_dir={k: round(p[f"sun:{k}"], 3) for k in hm.WALLS},
                                        heat={u: round(p[f"h:{u}"], 3) for u in us},
                                        cool={u: round(p[f"k:{u}"], 3) for u in us},
                                        coupling={j: round(p[f"c:{j}"], 4) for j in nbrs},
                                        leak={u: round(p[f"l:{u}"], 3) for u in own})
            rep[rid] = {"windows": len(A), "r2_rate": round(r2, 3)}
        return rooms_m, rep, sse

    # Each unit's lag: coordinate search over the candidates, by how well the training windows are explained.
    lags = {u: 0 for u in units}
    cache = {}

    def trial(lg, sc=None):
        sc = sc or {}
        key = (tuple(sorted(lg.items())), tuple(sorted((u, tuple(sorted(v.items()))) for u, v in sc.items())))
        if key not in cache:
            Ys = scaled(Y, F, sc)
            cache[key] = rooms_fit({u: lagged(Ys[u], lg.get(u, 0)) for u in Ys})
        return cache[key]

    # Only a unit with one of its own rooms in the fit: otherwise the "best" lag would come from how it happens to
    # line up with other rooms' changes (Bedroom 3's unit got 20 min from the kitchen and Primary bedroom). Those
    # keep 0, and the controller uses its default look-ahead.
    own_fitted = {u for u in units if any(r in T for r in layout["units"].get(u, {}).get("zone", []))}
    if any(Y[u] and any(Y[u]) for u in Y):
        for u in units:
            if u in own_fitted:
                lags[u] = min(LAGS_MIN, key=lambda L: trial({**lags, u: L})[2])
    # Each fan speed's output relative to a reference speed: full fan (auto/high) if the unit has run an hour at it,
    # else the speed it ran at most (whose factor the unit's effect then includes). Only speeds with an hour of running;
    # others count as the reference. By the same training fit.
    scales = {u: {} for u in units}
    ref_fan = {}
    for u in units:
        if u not in own_fitted:
            continue
        minutes = {}
        for y, f in zip(Y[u], F[u]):
            if y:
                minutes[f] = minutes.get(f, 0) + 1
        full = sum(minutes.get(f, 0) for f in FAN_REF)
        ref = "auto" if full >= 60 else max(minutes, key=minutes.get, default=None)
        if ref is None:
            continue
        ref_fan[u] = ref
        refs = set(FAN_REF) if ref in FAN_REF else {ref}
        levels = sorted(f for f, k in minutes.items() if f and f not in refs and k >= 60)
        for _ in range(3):  # coordinate search: a speed's best factor depends on the others', so a few passes
            moved = False
            for level in levels:
                best = min(FAN_GRID, key=lambda g: trial(lags, {**scales, u: {**scales[u], level: g}})[2])
                moved |= scales[u].get(level) != best
                scales[u][level] = best
            if not moved:
                break
    fitted, rep, _ = trial(lags, scales)
    # The slope fit explains each half hour on its own; refine each room so its 2 h simulated paths match too (what
    # the planner and the accuracy checks use). A fast room can otherwise explain its half hours with effects so large
    # that they run away over two hours.
    if paths:
        Ye = {u: lagged(v, lags.get(u, 0)) for u, v in scaled(Y, F, scales).items()}
        for rid in list(fitted):
            fitted[rid], rep[rid]["paths"] = refine_paths(rid, fitted[rid], T, out, sun, Ye, Lk, splits[rid])
    m.rooms.update(fitted)
    report.update(rep)
    zones = {u: spec.get("zone", []) for u, spec in layout["units"].items()}
    detected = detect_sensor_rooms(T, U, zones)
    for uid, spec in layout["units"].items():
        sr, off = (spec["sensor_rooms"], 0.0) if spec.get("sensor_rooms") is not None else detected.get(uid, ([], 0.0))
        r0 = rule.get(uid) or hm.UnitModel()
        m.units[uid] = hm.UnitModel(sensor_rooms=[r for r in sr if r in m.rooms], sensor_offset=off,
                                    lag_h=round(lags.get(uid, 0) / 60, 4), fan_scale=scales.get(uid, {}),
                                    k0=r0.k0, k1=r0.k1, ymin=r0.ymin)
    report["outputs"] = learned_out
    report["lags_min"] = lags
    report["fan_scale"] = {u: {"relative_to": ref_fan.get(u), **v} for u, v in scales.items() if v}
    report["holdout"] = validate(m, ts, T, out, sun, effective(Y, F, m), {r: splits[r] for r in m.rooms}, Lk)
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


def validate(m: hm.HouseModel, ts, T, out, sun, Y, split, L=None, horizon_min: int = 120, step_min: int = 5,
             every_min: int = 30):
    """2 h ahead on the held-out part with the units at their logged output: RMSE of the model vs 'nothing changes'."""
    errs = {r: [] for r in m.rooms}
    base = {r: [] for r in m.rooms}
    n = len(ts)
    splits = split if isinstance(split, dict) else {r: split for r in m.rooms}  # room -> first held-out minute
    for a in range(min(splits.values(), default=n), n - horizon_min, every_min):
        start = {r: T[r][a] for r in m.rooms if T[r][a] is not None}  # rooms without a reading sit this one out
        if not start:
            continue
        cur = dict(start)
        for k in range(a, a + horizon_min, step_min):
            seg = slice(k, k + step_min)
            o = _mean(out[seg])
            y = {u: sum(Y[u][seg]) / step_min for u in Y}
            lk = {u: sum(v[seg]) / step_min for u, v in (L or {}).items()}
            d = hm.derivative(m, cur, o, hm.mean_sun(sun[seg]), y, lk)
            cur = {r: cur[r] + d.get(r, 0.0) * step_min / 60 for r in cur}
        z = a + horizon_min
        for r in start:
            if a >= splits.get(r, n) and T[r][z] is not None:
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
    ts, T, out, sun, Y, _, F, Lk = series(rows, rooms, units, m.units)
    Y = effective(Y, F, m)
    res = {r: {str(h): [] for h in horizons} for r in rooms}
    hmax = max(horizons)
    for a in range(0, len(ts) - hmax, every_min):
        cur = {r: T[r][a] for r in rooms if T[r][a] is not None}
        if not cur:
            continue
        for k in range(a, a + hmax, step_min):
            seg = slice(k, k + step_min)
            d = hm.derivative(m, cur, _mean(out[seg]), hm.mean_sun(sun[seg]), {u: sum(Y[u][seg]) / step_min for u in Y},
                              {u: sum(v[seg]) / step_min for u, v in Lk.items()})
            cur = {r: cur[r] + d.get(r, 0.0) * step_min / 60 for r in cur}
            done = k + step_min - a
            if done in horizons:
                for r in cur:
                    res[r][str(done)].append([ts[a + done].isoformat(), round(cur[r], 2)])
    return res
