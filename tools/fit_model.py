"""Fit the structured thermal model (custom_components/room_climate/model.py) to minute logs.

    python tools/fit_model.py <log dir or .jsonl> <layout.json> -o model.json [--window 30] [--holdout 0.25]

Logs are the HA long-term log (/config/room_climate_data, plain or gzipped days) or simulator rows in the same shape.
The layout says which rooms to fit, which units can affect each room, which rooms exchange heat (neighbours), and
which rooms each unit regulates on:

    {"rooms": {"kitchen": {"units": ["air_handler"], "neighbors": ["living_room"]}, ...},
     "units": {"air_handler": {"sensor_rooms": ["kitchen"]}, ...}}

For each room, over windows of `window` minutes (overlapping, every 10 min):
    (T_end - T_start) / hours  =  loss*(out - T) + sum c_j*(T_j - T) + solar*sun + bias + sum heat_u*y_u+ - sum cool_u*y_u-
with every quantity averaged over the window, y_u+ / y_u- the unit's estimated heating / cooling output (0..1: on
when the log says heating / cooling, scaled by the inverter's response to its setpoint gap). Solved by ridge least
squares with every coefficient except bias held >= 0, so the result stays physical. Windows long enough to rise
above sensor quantization (H5100 0.18 F, unit sensors 0.9 F) matter more than more data points.

The fit is checked by simulating the held-out last part of the log 2 hours ahead from real starting temperatures,
with the units' logged outputs, against simply assuming nothing changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
import datalog  # noqa: E402
import model as hm  # noqa: E402


def load_rows(src: str) -> list:
    if os.path.isdir(src):
        return list(datalog.read_rows(src))
    with open(src) as f:
        return [json.loads(line) for line in f if line.strip()]


def series(rows: list, layout: dict):
    """Per-minute arrays: times, temps[room], outdoor, sun, y[unit] (signed estimated output)."""
    rooms = list(layout["rooms"])
    units = list(layout["units"])
    defaults = hm.UnitModel()
    T = {r: [] for r in rooms}
    out, sun, Y, ts = [], [], {u: [] for u in units}, []
    for row in rows:
        ts.append(datetime.fromisoformat(row["t"]))
        for r in rooms:
            v = (row.get("rooms", {}).get(r) or {}).get("temp")
            T[r].append(np.nan if v is None else float(v))
        o = row.get("outdoor")
        if o is None:
            o = (row.get("weather") or {}).get("temp")
        out.append(np.nan if o is None else float(o))
        sun.append(hm.solar_index((row.get("sun") or {}).get("elev"), (row.get("weather") or {}).get("cloud")))
        for u in units:
            d = row.get("units", {}).get(u) or {}
            act, sp, rd = d.get("action"), d.get("sp"), d.get("temp")
            if act in ("heating", "cooling") and sp is not None and rd is not None:
                err = (sp - rd) if act == "heating" else (rd - sp)
                y = max(defaults.ymin, min(1.0, defaults.k0 + defaults.k1 * err))
                Y[u].append(y if act == "heating" else -y)
            elif act in ("heating", "cooling"):
                Y[u].append(1.0 if act == "heating" else -1.0)
            else:
                Y[u].append(0.0)
    return ts, {r: np.array(v) for r, v in T.items()}, np.array(out), np.array(sun), {u: np.array(v) for u, v in Y.items()}


def nonneg_ridge(A: np.ndarray, b: np.ndarray, free: list, lam: float = 1e-3, iters: int = 3000) -> np.ndarray:
    """min |Ax - b|^2 + lam*|x|^2 (on standardized columns) with x_j >= 0 unless j in `free`. Projected coordinate
    descent on the normal equations; small problems (a dozen columns), so this is quick and dependable."""
    scale = np.sqrt((A ** 2).mean(axis=0))
    scale[scale == 0] = 1.0
    As = A / scale
    with np.errstate(all="ignore"):  # numpy 2.0 + macOS Accelerate raises spurious matmul warnings
        G = As.T @ As / len(b) + lam * np.eye(A.shape[1])
        h = As.T @ b / len(b)
    if not (np.isfinite(G).all() and np.isfinite(h).all()):
        raise ValueError("non-finite regression data")
    x = np.zeros(A.shape[1])
    for _ in range(iters):
        delta = 0.0
        for j in range(A.shape[1]):
            v = (h[j] - G[j] @ x + G[j, j] * x[j]) / G[j, j]
            if j not in free:
                v = max(0.0, v)
            delta = max(delta, abs(v - x[j]))
            x[j] = v
        if delta < 1e-12:
            break
    return x / scale


def windows(n: int, w: int, stride: int, start: int = 0, stop: int = None):
    stop = n if stop is None else stop
    return [(k, k + w) for k in range(start, stop - w, stride)]


def fit(rows: list, layout: dict, window: int = 30, stride: int = 10, holdout: float = 0.25, lam: float = 1e-3):
    ts, T, out, sun, Y = series(rows, layout)
    n = len(ts)
    split = int(n * (1 - holdout))
    units = list(layout["units"])
    m = hm.HouseModel()
    report = {}
    for rid, spec in layout["rooms"].items():
        nbrs = [j for j in spec.get("neighbors", []) if j in T]
        us = [u for u in spec.get("units", units) if u in Y]
        cols = ["loss"] + [f"c:{j}" for j in nbrs] + ["solar", "bias"] + [f"h:{u}" for u in us] + [f"k:{u}" for u in us]
        A, b = [], []
        for a, z in windows(split, window, stride):
            Ti = T[rid][a:z]
            if np.isnan(Ti[0]) or np.isnan(Ti[-1]) or np.isnan(Ti).mean() > 0.2 or np.isnan(out[a:z]).all():
                continue
            if any(np.isnan(T[j][a:z]).mean() > 0.2 for j in nbrs):
                continue
            Tm = np.nanmean(Ti)
            row = [np.nanmean(out[a:z]) - Tm] + [np.nanmean(T[j][a:z]) - Tm for j in nbrs] + [sun[a:z].mean(), 1.0]
            row += [np.clip(Y[u][a:z], 0, None).mean() for u in us] + [np.clip(Y[u][a:z], None, 0).mean() for u in us]
            A.append(row)
            b.append((Ti[-1] - Ti[0]) / ((z - a) / 60))
        if len(A) < len(cols) * 3:
            report[rid] = {"skipped": f"only {len(A)} usable windows"}
            continue
        A, b = np.array(A), np.array(b)
        x = nonneg_ridge(A, b, free=[cols.index("bias")], lam=lam)
        with np.errstate(all="ignore"):
            pred = A @ x
        r2 = 1 - ((b - pred) ** 2).sum() / max(1e-9, ((b - b.mean()) ** 2).sum())
        p = {c: float(v) for c, v in zip(cols, x)}
        m.rooms[rid] = hm.RoomModel(loss=round(p["loss"], 4), solar=round(p["solar"], 3), bias=round(p["bias"], 3),
                                    heat={u: round(p[f"h:{u}"], 3) for u in us},
                                    cool={u: round(p[f"k:{u}"], 3) for u in us},  # the cooling column is negative
                                    coupling={j: round(p[f"c:{j}"], 4) for j in nbrs})
        report[rid] = {"windows": len(A), "r2_rate": round(float(r2), 3)}
    for uid, spec in layout["units"].items():
        sr = [r for r in spec.get("sensor_rooms", []) if r in T]
        diffs = []
        for k, row in enumerate(rows):
            rd = (row.get("units", {}).get(uid) or {}).get("temp")
            vals = [T[r][k] for r in sr if not np.isnan(T[r][k])]
            if rd is not None and vals:
                diffs.append(rd - sum(vals) / len(vals))
        m.units[uid] = hm.UnitModel(sensor_rooms=sr, sensor_offset=round(float(np.median(diffs)), 2) if diffs else 0.0)
    report["holdout"] = validate(m, ts, T, out, sun, Y, split, horizon_min=120)
    usable = all(v["model_rmse_F"] is not None and v["no_change_rmse_F"] is not None
                 and v["model_rmse_F"] < v["no_change_rmse_F"] for v in report["holdout"].values())
    m.meta = {"fitted": datetime.now().isoformat(timespec="seconds"), "usable": usable, "rows": n, "window_min": window,
              "from": ts[0].isoformat() if ts else None, "to": ts[-1].isoformat() if ts else None, "report": report}
    return m, report


def validate(m: hm.HouseModel, ts, T, out, sun, Y, split: int, horizon_min: int = 120):
    """2 h ahead on the held-out part, units at their logged output: RMSE of the model vs 'nothing changes'."""
    errs, base = {r: [] for r in m.rooms}, {r: [] for r in m.rooms}
    n = len(ts)
    for a in range(split, n - horizon_min, 30):
        temps = {r: T[r][a] for r in m.rooms if not np.isnan(T[r][a])}
        if len(temps) < len(m.rooms):
            continue
        cur = dict(temps)
        for k in range(a, a + horizon_min):
            o = out[k] if not np.isnan(out[k]) else np.nanmean(out[a:a + horizon_min])
            d = hm.derivative(m, cur, o, float(sun[k]), {u: float(Y[u][k]) for u in Y})
            cur = {r: cur[r] + d.get(r, 0.0) / 60 for r in cur}
        z = a + horizon_min
        for r in m.rooms:
            if not np.isnan(T[r][z]):
                errs[r].append(cur[r] - T[r][z])
                base[r].append(temps[r] - T[r][z])
    rmse = lambda v: round(float(np.sqrt(np.mean(np.square(v)))), 2) if v else None
    return {r: {"model_rmse_F": rmse(errs[r]), "no_change_rmse_F": rmse(base[r]), "n": len(errs[r])} for r in m.rooms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("layout")
    ap.add_argument("-o", "--out", default="model.json")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--lam", type=float, default=1e-3)
    a = ap.parse_args()
    rows = load_rows(a.logs)
    layout = json.load(open(a.layout))
    m, rep = fit(rows, layout, a.window, holdout=a.holdout, lam=a.lam)
    json.dump(m.to_dict(), open(a.out, "w"), indent=2)
    print(f"{len(rows)} rows -> {a.out}")
    for rid, r in m.rooms.items():
        print(f"  {rid:16s} loss {r.loss:.3f}/h  solar {r.solar:5.2f}  bias {r.bias:+.2f}  heat {r.heat}  cool {r.cool}"
              f"  coupling {r.coupling}  | {rep.get(rid)}")
    for rid, v in rep.items():
        if rid not in m.rooms and rid != "holdout":
            print(f"  {rid:16s} {v}")
    print("  2 h ahead on held-out data:", json.dumps(rep["holdout"]))
    bad = [r for r, v in rep["holdout"].items()
           if v["model_rmse_F"] is None or v["no_change_rmse_F"] is None or v["model_rmse_F"] >= v["no_change_rmse_F"]]
    if bad:
        print(f"  NOT USABLE YET: predicts no better than 'nothing changes' for {', '.join(bad)}. More (and steadier) "
              "data needed before planning with this model.")


if __name__ == "__main__":
    main()
