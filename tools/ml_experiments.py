"""Is the ML data-limited or converged, and would a different kind of model do better? Out-of-sample rows only.

    ~/.local/share/room-climate-watch/analysis-venv/bin/python tools/ml_experiments.py

Uses the rows tools/ml_train_local.py rebuilt (the physics model's out-of-sample predictions, one day at a time) under
~/.local/share/room-climate-watch/ml/bootstrap. Walk-forward: a test day is predicted by models that saw only earlier days.
  learning curve   error against the number of training days
  iterations       boosted trees against the number of boosting rounds
  capacity         the same trees scored on rows they were trained on (can the features explain the miss at all?)
  families         ridge (pooled, per room), boosted trees, extra trees, small neural net, and simple blends
"""
import glob, json, os, sys
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

WORK = os.path.expanduser("~/.local/share/room-climate-watch/ml")
HZ = (30, 60, 120)


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(WORK, "bootstrap", "*.json"))):
        rows += json.load(open(p))
    rooms = sorted({r[1] for r in rows})
    day = np.array([r[0][:10] for r in rows])
    hz = np.array([r[2] for r in rows])
    room = np.array([rooms.index(r[1]) for r in rows])
    X = np.array([r[4] for r in rows])
    y = np.array([r[3] for r in rows])
    return rows, rooms, day, hz, room, X, y


rows, rooms, DAY, HZA, ROOM, X, Y = load()
ODE = X[:, 0]
RES = Y - ODE
OH = np.eye(len(rooms))[ROOM]
XF = np.hstack([X, OH])
days = [d for d in sorted(set(DAY)) if (DAY == d).sum() >= int(os.environ.get("MIN_ROWS", 1000))]  # full days only (a few rows complete just after midnight)
print(f"{len(rows)} out-of-sample rows, days {days[0]}..{days[-1]}, rooms {len(rooms)}\n")


def rmse(a):
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def fit_predict(kind, tr, te, h, **kw):
    """Predicted CHANGE for rows te (a mask), trained on rows tr (a mask), at horizon h."""
    if kind == "physics":
        return ODE[te]
    if kind == "ridge":  # pooled, residual target (what the Green and Mac train now)
        m = Ridge(alpha=20.0).fit(StandardScaler().fit(XF[tr]).transform(XF[tr]), RES[tr])
        sc = StandardScaler().fit(XF[tr])
        return ODE[te] + Ridge(alpha=20.0).fit(sc.transform(XF[tr]), RES[tr]).predict(sc.transform(XF[te]))
    if kind == "ridge per room":
        out = np.zeros(te.sum())
        te_idx = np.where(te)[0]
        for k in range(len(rooms)):
            a, b = tr & (ROOM == k), te_idx[ROOM[te_idx] == k]
            if a.sum() < 30 or len(b) == 0:
                out[ROOM[te_idx] == k] = ODE[b]
                continue
            sc = StandardScaler().fit(X[a])
            out[ROOM[te_idx] == k] = ODE[b] + Ridge(alpha=50.0).fit(sc.transform(X[a]), RES[a]).predict(sc.transform(X[b]))
        return out
    if kind == "trees (residual)":
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=kw.get("iters", 120), learning_rate=0.05, l2_regularization=5.0,
                                          min_samples_leaf=30, random_state=0).fit(XF[tr], RES[tr])
        return ODE[te] + m.predict(XF[te])
    if kind == "trees (alone)":
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=kw.get("iters", 120), learning_rate=0.05, l2_regularization=5.0,
                                          min_samples_leaf=30, random_state=0).fit(XF[tr][:, 1:], Y[tr])
        return m.predict(XF[te][:, 1:])
    if kind == "extra trees (residual)":
        m = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=15, max_features=0.6, n_jobs=-1, random_state=0).fit(XF[tr], RES[tr])
        return ODE[te] + m.predict(XF[te])
    if kind == "neural net (residual)":
        sc = StandardScaler().fit(XF[tr])
        m = MLPRegressor((32, 16), alpha=1e-1, early_stopping=True, max_iter=400, random_state=0).fit(sc.transform(XF[tr]), RES[tr])
        return ODE[te] + m.predict(sc.transform(XF[te]))
    raise ValueError(kind)


def walk(kind, test_days, train_k=None, **kw):
    """RMSE per horizon over the test days, each predicted by models trained on the (last train_k) days before it."""
    err = {h: [] for h in HZ}
    for T in test_days:
        prior = [d for d in days if d < T]
        if train_k:
            prior = prior[-train_k:]
        if not prior:
            continue
        for h in HZ:
            tr = np.isin(DAY, prior) & (HZA == h)
            te = (DAY == T) & (HZA == h)
            if tr.sum() < 200 or te.sum() == 0:
                continue
            err[h].append(fit_predict(kind, tr, te, h, **kw) - Y[te])
    return {h: rmse(np.concatenate(v)) if v else float("nan") for h, v in err.items()}


def show(label, r, base=None):
    s = f"  {label:34s}" + "".join(f"{r[h]:8.3f}" for h in HZ)
    if base:
        s += "    " + " ".join(f"{100 * (r[h] / base[h] - 1):+5.1f}%" for h in HZ)
    print(s)


tests = days[3:]  # the first three days have too little before them
print(f"test days {tests}   (rmse F at 30/60/120 min; right: change vs physics)")
base = walk("physics", tests)
show("physics alone", base)

print("\n1. LEARNING CURVE: error vs number of training days (test days " + str(tests[-2:]) + ")")
t2 = tests[-2:]
b2 = walk("physics", t2)
show("physics alone", b2)
for k in (1, 2, 3, 4, 5):
    for kind in ("ridge", "trees (residual)", "trees (alone)"):
        show(f"{kind}, {k} day(s) of training", walk(kind, t2, train_k=k), b2)
    print()

print("2. BOOSTED TREES vs NUMBER OF ROUNDS (all earlier days, test days " + str(t2) + ")")
for it in (10, 30, 60, 120, 250, 500):
    show(f"trees (residual), {it} rounds", walk("trees (residual)", t2, iters=it), b2)

print("\n3. CAPACITY: scored on the rows the model was trained on (can the features explain the miss?)")
for T in t2[-1:]:
    for h in HZ:
        te = (DAY == T) & (HZA == h)
        tr = (DAY <= T) & (HZA == h)
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, l2_regularization=5.0, min_samples_leaf=30,
                                          random_state=0).fit(XF[tr], RES[tr])
        ins = rmse(ODE[te] + m.predict(XF[te]) - Y[te])
        oos = rmse(fit_predict("trees (residual)", (DAY < T) & (HZA == h), te, h) - Y[te])
        print(f"  {h:3d} min, day {T}: physics {rmse(ODE[te] - Y[te]):.3f}   trees trained on this day too (in-sample) {ins:.3f}   trees OUT-of-sample {oos:.3f}")

print("\n4. MODEL FAMILIES (all earlier days of training; test days " + str(tests) + ")")
for kind in ("physics", "ridge", "ridge per room", "trees (residual)", "trees (alone)", "extra trees (residual)", "neural net (residual)"):
    show(kind, walk(kind, tests), base)
