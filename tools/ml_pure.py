"""Training rows for the models that don't need the physics model's out-of-sample prediction ("pure" ML).

The correction models learn the physics model's misses, so each row needs a prediction the physics model made without
having seen that day: that costs about a day per room of history and drops the early days. A model that predicts the
change in temperature directly needs none of that: its rows only need the features (the unit outputs, compressor run
time, sun and so on, replayed from the minute log), so every day we have can train it. The latest physics model is used
here only to estimate those features (a unit's output rule, a room's neighbours), never as a prediction the model learns
from: the `ode` feature (index 0) is dropped by the pure models.

    rows(logs, model_dict, layout, days, workers) -> {day: [[iso, room, horizon, real change, features]]}   (cached)
"""
import json, os, sys
from datetime import timedelta
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: select.py must not shadow stdlib
import mlmodel as ml  # noqa: E402
import model as hm  # noqa: E402

G = {}


def _init(logs, model_dict, since):
    G["inp"], G["m"], G["since"] = ml.Inputs(logs), hm.HouseModel.from_dict(model_dict), since


def _day(day):
    inp, m, since = G["inp"], G["m"], G["since"]
    out = []
    for i in range(inp.n):
        if inp.rows[i]["t"][:10] != day or inp.ts[i].minute % 10:
            continue
        for rid, h, y, f in ml.feature_rows(m, inp, i):
            if since.get(rid) and inp.rows[i]["t"] < since[rid]:
                continue  # before the room's sensor was replaced: that isn't the room
            out.append([(inp.ts[i] + timedelta(minutes=h)).isoformat(), rid, h, round(y, 4), [round(x, 4) for x in f]])
    return day, out


def rows(logs, model_dict, layout, days, workers, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    since = {rid: spec["since"] for rid, spec in layout["rooms"].items() if spec.get("since")}
    paths = {d: os.path.join(cache_dir, f"{d}.json") for d in days}
    todo = [d for d in days if not os.path.exists(paths[d])]
    if todo:
        with Pool(min(workers, len(todo)), initializer=_init, initargs=(logs, model_dict, since)) as pool:
            for d, rws in pool.imap_unordered(_day, todo):
                json.dump(rws, open(paths[d], "w"))
                print(f"  {d}: {len(rws)} pure rows", flush=True)
    return {d: json.load(open(paths[d])) for d in days}
