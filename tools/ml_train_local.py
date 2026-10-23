"""Train the ML correction on this machine from the logs mirrored off the Home Assistant Green.

    ~/.local/share/room-climate-watch/analysis-venv/bin/python tools/ml_train_local.py [--sync] [--eval-days 2]

1. --sync: tools/ha_sync.py pulls new logs and the live planner's ML rows from the Green.
2. Training rows = the live rows (the Green's planner logs one per completed prediction; out-of-sample by construction)
   plus, for earlier days, rows rebuilt here from the minute logs: for each day D the physics model is refitted on the days
   before D and replayed over D (ml_benchmark.fit_day / dataset_day; cached under the work directory, one new day a day).
3. Fit the correction (mlmodel.train: the pure-Python ridge the Green can run) and save it with a report.
4. Walk-forward score on the latest complete days (models trained only on the days before): physics alone, + ridge, and,
   if scikit-learn is installed, + trees and trees alone, so we see whether a heavier learner is worth deploying.

Work directory: ~/.local/share/room-climate-watch/ml  (corrector.json, report.json, history.jsonl, ode.json, bootstrap/).
Nothing is sent to the Green.
"""
import argparse, glob, gzip, json, os, sys, time
from datetime import datetime
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: select.py must not shadow stdlib
import ha_sync  # noqa: E402
import local_fit as lf  # noqa: E402
import ml_benchmark as mb  # noqa: E402
import ml_pure  # noqa: E402
import mlmodel  # noqa: E402

MIRROR = os.path.expanduser("~/.local/share/room-climate-watch/ha")
WORK = os.path.expanduser("~/.local/share/room-climate-watch/ml")


def read_logs(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "room_climate_data", "*.jsonl*"))):
        raw = gzip.open(p, "rt").read() if p.endswith(".gz") else open(p).read()
        rows += [json.loads(x) for x in raw.splitlines() if x.startswith("{")]
    rows.sort(key=lambda r: r["t"])
    return rows


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def rmse(v):
    return (sum(x * x for x in v) / len(v)) ** 0.5 if v else float("nan")


def evaluate(rows, eval_days, pure=None):
    """Walk-forward: for each of the latest complete days, train on the rows before it, score on it. `pure`: {day: rows}
    for every day in the log (ml_pure): "trees (all days)" is trained on those, which need no out-of-sample physics."""
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        np = None
    days = sorted({r[0][:10] for r in rows})[-(eval_days + 1):-1] if len(rows) else []  # latest day may be incomplete
    err = {k: {h: [] for h in mlmodel.HORIZONS} for k in ("nothing", "physics", "+ ridge", "+ trees", "trees", "trees (all days)", "blend")}
    for day in days:
        tr = [r for r in rows if r[0][:10] < day]
        te = [r for r in rows if r[0][:10] == day]
        c = mlmodel.train(tr)
        for h in mlmodel.HORIZONS:
            Xtr = [r[4] for r in tr if r[2] == h]
            rooms = sorted({r[1] for r in rows})
            oh = lambda room: [1.0 if room == q else 0.0 for q in rooms]
            if np is not None and len(Xtr) > 300:
                Xa = np.array([r[4] + oh(r[1]) for r in tr if r[2] == h])
                res = np.array([r[3] - r[4][0] for r in tr if r[2] == h])
                gb = dict(max_depth=3, max_iter=120, learning_rate=0.05, l2_regularization=5.0, min_samples_leaf=30, random_state=0)
                tres = HistGradientBoostingRegressor(**gb).fit(Xa, res)
                tpure = HistGradientBoostingRegressor(**gb).fit(Xa[:, 1:], np.array([r[3] for r in tr if r[2] == h]))
            else:
                tres = tpure = None
            tall = None
            if np is not None and pure:
                prows = [r for d_, v in pure.items() if d_ < day for r in v if r[2] == h]  # every earlier day, no physics needed
                if len(prows) > 300:
                    Xp = np.array([r[4] + oh(r[1]) for r in prows])[:, 1:]
                    gb2 = dict(max_depth=3, max_iter=120, learning_rate=0.05, l2_regularization=5.0, min_samples_leaf=30, random_state=0)
                    tall = HistGradientBoostingRegressor(**gb2).fit(Xp, np.array([r[3] for r in prows]))
            for r in te:
                if r[2] != h:
                    continue
                y, f = r[3], r[4]
                err["nothing"][h].append(-y)
                err["physics"][h].append(f[0] - y)
                err["+ ridge"][h].append(f[0] + (c.correct(r[1], h, f) if c else 0.0) - y)
                if tres is not None:
                    x = np.array([f + oh(r[1])])
                    err["+ trees"][h].append(f[0] + float(tres.predict(x)[0]) - y)
                    err["trees"][h].append(float(tpure.predict(x[:, 1:])[0]) - y)
                if tall is not None:
                    xa = np.array([f + oh(r[1])])[:, 1:]
                    pt = float(tall.predict(xa)[0])
                    err["trees (all days)"][h].append(pt - y)
                    err["blend"][h].append(0.5 * (f[0] + (c.correct(r[1], h, f) if c else 0.0)) + 0.5 * pt - y)
    return days, {k: {str(h): rmse(v) for h, v in d.items()} for k, d in err.items() if any(d[h] for h in d)}, \
        {str(h): len(err["physics"][h]) for h in mlmodel.HORIZONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--eval-days", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    if a.sync:
        sys.argv = [sys.argv[0]]
        ha_sync.main()
    os.makedirs(os.path.join(WORK, "bootstrap"), exist_ok=True)
    cfg = json.load(open(os.path.join(MIRROR, ".storage", "room_climate")))
    cfg = cfg.get("data", cfg)
    layout = lf.layout_from_config(cfg)
    logs = read_logs(MIRROR)
    dates = sorted({r["t"][:10] for r in logs})
    counts = {d: sum(1 for r in logs if r["t"][:10] == d) for d in dates}
    live = read_jsonl(os.path.join(MIRROR, "room_climate_ml_rows.jsonl"))
    first_live = min((r[0][:10] for r in live), default=None)
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    # rebuilt rows for the complete days before the live rows begin (the first two days have no history to fit on)
    days = [d for d in dates[2:] if counts[d] >= 1000 and d < today and (first_live is None or d < first_live)]
    ode_path = os.path.join(WORK, "ode.json")
    models = json.load(open(ode_path)) if os.path.exists(ode_path) else {}
    boot = {d: os.path.join(WORK, "bootstrap", f"{d}.json") for d in days}
    todo = [d for d in days if not os.path.exists(boot[d])]
    if todo:
        need_fit = [d for d in todo if d not in models]
        with Pool(min(a.workers, max(1, len(need_fit))), initializer=mb.init, initargs=(logs, layout, models)) as pool:
            for d, md in pool.imap_unordered(mb.fit_day, need_fit):
                models[d] = md
                json.dump(models, open(ode_path, "w"))
                print(f"  physics model fitted on the days before {d} ({time.time() - t0:.0f}s)", flush=True)
        with Pool(min(a.workers, len(todo)), initializer=mb.init, initargs=(logs, layout, models)) as pool:
            for d, rws in pool.imap_unordered(mb.dataset_day, todo):
                json.dump([[iso, rid, h, y, f] for rid, h, y, f, iso in rws], open(boot[d], "w"))
                print(f"  {d}: {len(rws)} rebuilt rows ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for d in days:
        rows += json.load(open(boot[d]))
    n_boot = len(rows)
    rows += live
    print(f"training rows: {n_boot} rebuilt from {len(days)} day(s) + {len(live)} live", flush=True)
    c = mlmodel.train(rows)
    report = {"at": datetime.now().astimezone().isoformat(timespec="seconds"), "rows_rebuilt": n_boot, "rows_live": len(live),
              "days_rebuilt": days, "first_live_day": first_live}
    if c is not None:
        json.dump({**c.to_dict(), "trained_on_rows": len(rows), "at": report["at"]}, open(os.path.join(WORK, "corrector.json"), "w"))
        report["enabled"] = {k: v for k, v in c.enabled.items()}
        report["held_out_rmse_physics_vs_corrected"] = c.report
        print(f"corrector: enabled for {sum(c.enabled.values())} of {len(c.enabled)} room/horizon pairs", flush=True)
    else:
        print("not enough rows yet for a correction", flush=True)
    model_path = os.path.join(MIRROR, "room_climate_model.json")
    pure = {}
    if os.path.exists(model_path):
        pure_days = [d for d in dates if counts[d] >= 1000 and d < today]
        pure = ml_pure.rows(logs, json.load(open(model_path)), layout, pure_days, a.workers, os.path.join(WORK, "pure"))
        report["rows_pure"] = sum(len(v) for v in pure.values())
        print(f"pure rows (no physics prediction needed): {report['rows_pure']} from {len(pure)} day(s)", flush=True)
    ev_days, rm, n = evaluate(rows, a.eval_days, pure)
    report["walk_forward"] = {"days": ev_days, "rmse_F": rm, "n": n}
    for k, v in rm.items():
        print(f"  {k:10s}" + "".join(f"{v[str(h)]:8.3f}" for h in mlmodel.HORIZONS), flush=True)
    print(f"  (rmse F at {', '.join(str(h) for h in mlmodel.HORIZONS)} min over {ev_days}, n={n})", flush=True)
    json.dump(report, open(os.path.join(WORK, "report.json"), "w"), indent=1)
    with open(os.path.join(WORK, "history.jsonl"), "a") as f:
        f.write(json.dumps({k: report[k] for k in ("at", "rows_rebuilt", "rows_live", "walk_forward")}) + "\n")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
