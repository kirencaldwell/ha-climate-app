"""Run today's controller and the model-predictive one on the simulated house and compare.

    python tools/compare_controllers.py <room_climate config json> [--days 3] [--model fitted.json] [--seed 1]

The config (from room_climate/get or .storage/room_climate) supplies the real rooms, schedules, priorities and
settings. With --model, the MPC plans with that (e.g. fitted) model instead of the simulator's true one.
"""
import argparse, json, os, sys, time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
import coupled  # noqa: E402
import model as hm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--days", type=float, default=3)
    ap.add_argument("--model")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", default="2026-10-16T00:00:00-07:00")
    ap.add_argument("--mean", type=float, default=55.0, help="mean outdoor F")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg = cfg.get("data", cfg)
    truth = coupled.the_house()
    rooms, units = coupled.rooms_units_from_config(cfg, truth)
    settings = coupled.settings_from_config(cfg)
    start = datetime.fromisoformat(a.start)
    plan_model = hm.HouseModel.from_dict(json.load(open(a.model))) if a.model else truth
    results = []
    for ctl in (coupled.HeuristicController(rooms, units, settings),
                coupled.MPCController(rooms, units, plan_model, settings)):
        t0 = time.time()
        w = coupled.Weather(start, int(a.days) + 1, mean=a.mean, seed=a.seed)
        r = coupled.run(truth, rooms, units, ctl, start, a.days, w, settings, seed=a.seed)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
    keys = ["discomfort", "mode_switches", "compressor_starts", "commands", "energy_unit_h", "seconds"]
    print(f"{'':22s}" + "".join(f"{r['controller']:>14s}" for r in results))
    for k in keys:
        print(f"{k:22s}" + "".join(f"{r[k]:>14}" for r in results))
    for rid in results[0]["by_room"]:
        print(f"  {rid:20s}" + "".join(f"{r['by_room'][rid]:>8} / {r['hours_outside_band'][rid]:>4}h" for r in results))


if __name__ == "__main__":
    main()
