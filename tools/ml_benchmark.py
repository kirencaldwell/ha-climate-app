"""Is a daily-retrained ML model better than the physics model? A walk-forward test on the real minute log.

    ~/.local/share/room-climate-watch/analysis-venv/bin/python tools/ml_benchmark.py [--days 9]

For each test day T the models only know the days before T:
  physics   the grey-box model fitted on the log before T (tools/local_fit.py does the same fit as the HA nightly job)
  + ridge   physics + a pure-Python ridge correction (what the integration can train and run: mlmodel.Ridge)
  + trees   physics + gradient-boosted trees (scikit-learn) on the same features
  ridge, trees   the same learners predicting the change directly, without the physics model's prediction as an input
  nothing changes   the baseline
The correction models are trained on the physics model's OUT-OF-SAMPLE misses: for each earlier day D, the physics model
fitted on the days before D, scored on D. (Training them on in-sample misses would teach them that the physics model
is better than it is.) Predictions are 30/60/120 min ahead from every 10th minute, rooms pooled, replaying what the units,
weather and sun really did. Results go to reviews/ml/results.json.
"""
import argparse, json, os, sys, time
from datetime import timedelta
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: select.py must not shadow stdlib
import local_fit as lf  # noqa: E402
import fit  # noqa: E402
import mlmodel as ml  # noqa: E402
import model as hm  # noqa: E402

OUT = os.path.join(HERE, "..", "reviews", "ml")
G = {}


def init(rows, layout, models):
    G["rows"], G["layout"], G["models"] = rows, layout, models
    G["inp"] = ml.Inputs(rows)


def date_of(row):
    return row["t"][:10]


def fit_day(day):
    rows = [r for r in G["rows"] if date_of(r) < day]
    m, rep = fit.fit(rows, G["layout"])
    return day, m.to_dict()


def dataset_day(day):
    """Rows for origins on `day` (every 10 min), with the physics model fitted on the days before it."""
    m = hm.HouseModel.from_dict(G["models"][day])
    inp = G["inp"]
    out = []
    for i in range(inp.n):
        if date_of(inp.rows[i]) != day or inp.ts[i].minute % 10:
            continue
        for rid, h, y, f in ml.feature_rows(m, inp, i):
            out.append((rid, h, y, f, (inp.ts[i] + timedelta(minutes=h)).isoformat()))  # stamped when the prediction completes
    return day, out


def rmse(v):
    return (sum(x * x for x in v) / len(v)) ** 0.5 if v else float("nan")


def main():
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=9)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(os.path.join(HERE, "..", "reviews", "config_snapshot.json")))
    cfg = cfg.get("data", cfg)
    layout = lf.layout_from_config(cfg)
    rows = lf.load_rows(a.days)
    dates = sorted({date_of(r) for r in rows})
    counts = {d: sum(1 for r in rows if date_of(r) == d) for d in dates}
    days = [d for d in dates[2:] if counts[d] >= 300]  # physics model needs about two days of history before it
    print("days", {d: counts[d] for d in dates}, flush=True)
    t0 = time.time()
    path = os.path.join(OUT, "ode.json")
    models = json.load(open(path)) if os.path.exists(path) else {}
    todo = [d for d in days if d not in models]
    with Pool(a.workers, initializer=init, initargs=(rows, layout, models)) as pool:
        for d, md in pool.imap_unordered(fit_day, todo):
            models[d] = md
            json.dump(models, open(path, "w"))
            print(f"  physics model for {d}: fitted on the days before it ({time.time() - t0:.0f}s)", flush=True)
    data = {}
    with Pool(a.workers, initializer=init, initargs=(rows, layout, models)) as pool:
        for d, rws in pool.imap_unordered(dataset_day, days):
            data[d] = rws
            print(f"  {d}: {len(rws)} prediction rows ({time.time() - t0:.0f}s)", flush=True)
    room_ids = sorted({r[0] for d in data.values() for r in d})
    onehot = {r: [1.0 if r == q else 0.0 for q in room_ids] for r in room_ids}
    ode_i = ml.FEATURES.index("ode")

    def arrays(day_list, h):
        X, y, ode, rid_l, day_l = [], [], [], [], []
        for d in day_list:
            for rid, hh, yy, f, *_ in data[d]:
                if hh == h:
                    X.append(f + onehot[rid]); y.append(yy); ode.append(f[ode_i]); rid_l.append(rid); day_l.append(d)
        return X, y, ode, rid_l, day_l

    tests = [d for d in days[2:]]
    err = {m_: {h: [] for h in ml.HORIZONS} for m_ in ("nothing", "physics", "+ ridge", "+ trees", "ridge", "trees")}
    by_day = {}
    by_room = {}
    for T in tests:
        train_days = [d for d in days if d < T]
        for h in ml.HORIZONS:
            Xtr, ytr, otr, _, _ = arrays(train_days, h)
            Xte, yte, ote, rte, _ = arrays([T], h)
            if len(Xtr) < 200 or not Xte:
                continue
            res_tr = [y_ - o_ for y_, o_ in zip(ytr, otr)]
            drop = [j for j in range(len(Xtr[0])) if j != ode_i]  # features without the physics prediction (pure ML)
            pure = lambda X: [[r[j] for j in drop] for r in X]
            ridge_res = ml.Ridge(20.0).fit(Xtr, res_tr)
            ridge_pure = ml.Ridge(20.0).fit(pure(Xtr), ytr)
            gb = dict(max_depth=3, max_iter=120, learning_rate=0.05, l2_regularization=5.0, min_samples_leaf=30, random_state=0)
            trees_res = HistGradientBoostingRegressor(**gb).fit(np.array(Xtr), np.array(res_tr))
            trees_pure = HistGradientBoostingRegressor(**gb).fit(np.array(pure(Xtr)), np.array(ytr))
            p_tr = trees_res.predict(np.array(Xte))
            p_tp = trees_pure.predict(np.array(pure(Xte)))
            for k in range(len(Xte)):
                y_ = yte[k]
                preds = {"nothing": 0.0, "physics": ote[k], "+ ridge": ote[k] + ridge_res.predict(Xte[k]),
                         "+ trees": ote[k] + float(p_tr[k]), "ridge": ridge_pure.predict(pure([Xte[k]])[0]), "trees": float(p_tp[k])}
                for m_, v in preds.items():
                    err[m_][h].append(v - y_)
                    by_day.setdefault((T, m_, h), []).append(v - y_)
                    by_room.setdefault((rte[k], m_, h), []).append(v - y_)
    print("\nWalk-forward, every day's models know only the days before it. Test days:", tests)
    print(f"{'rmse (F)':16s}" + "".join(f"{h:>9d} min" for h in ml.HORIZONS) + "   (n = %s)" % len(err["physics"][60]))
    for m_ in err:
        print(f"{m_:16s}" + "".join(f"{rmse(err[m_][h]):12.3f}" for h in ml.HORIZONS))
    print("\nphysics vs physics + trees, per test day (60 min | 120 min):")
    for T in tests:
        print(f"  {T}:  physics {rmse(by_day.get((T, 'physics', 60), [])):.2f} | {rmse(by_day.get((T, 'physics', 120), [])):.2f}   + trees {rmse(by_day.get((T, '+ trees', 60), [])):.2f} | {rmse(by_day.get((T, '+ trees', 120), [])):.2f}   + ridge {rmse(by_day.get((T, '+ ridge', 60), [])):.2f} | {rmse(by_day.get((T, '+ ridge', 120), [])):.2f}   nothing {rmse(by_day.get((T, 'nothing', 60), [])):.2f} | {rmse(by_day.get((T, 'nothing', 120), [])):.2f}")
    print("\n120 min, per room (physics | + trees | trees | nothing):")
    for rid in room_ids:
        print(f"  {rid:16s} {rmse(by_room.get((rid, 'physics', 120), [])):.2f} | {rmse(by_room.get((rid, '+ trees', 120), [])):.2f} | {rmse(by_room.get((rid, 'trees', 120), [])):.2f} | {rmse(by_room.get((rid, 'nothing', 120), [])):.2f}")
    json.dump({"tests": tests, "rmse": {m_: {str(h): rmse(v) for h, v in d.items()} for m_, d in err.items()},
               "n": {str(h): len(err["physics"][h]) for h in ml.HORIZONS}}, open(os.path.join(OUT, "results.json"), "w"))


if __name__ == "__main__":
    main()
