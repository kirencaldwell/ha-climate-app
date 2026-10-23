"""Fit the model on the real minute log on this machine (the same code the HA nightly job runs), with and without the
start-up pulse, and print Bedroom 3's pulse and the held-out scores.

    python3 tools/local_fit.py [--days 9] [--out reviews/local_fit.json]

Reads the log over SSH (as comfort_watch.py does) and the room layout from reviews/config_snapshot.json.
"""
import argparse, gzip, json, os, sys, time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import comfort_watch as w  # noqa: E402
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: select.py must not shadow stdlib
import core  # noqa: E402
import fit  # noqa: E402


def load_rows(days):
    now = datetime.now().astimezone()
    rows = []
    for k in range(days):
        day = (now - timedelta(days=k)).strftime("%Y-%m-%d")
        raw = w.ssh(f"cat {w.LOGDIR}/{day}.jsonl 2>/dev/null || cat {w.LOGDIR}/{day}.jsonl.gz 2>/dev/null")
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        rows += [json.loads(x) for x in raw.decode().splitlines() if x.startswith("{")]
    rows.sort(key=lambda r: r["t"])
    return rows


def layout_from_config(cfg):
    rooms = [core.Room(id=d["id"], name=d.get("name", d["id"]), unit=d.get("unit"), sensor=d.get("sensor"),
                       priority=int(d.get("priority", 3)), default_temp=float(d.get("default_temp", 68)), floor=int(d.get("floor", 1)))
             for d in cfg["rooms"]]
    units = [core.Unit(u["id"], u["entity"]) for u in cfg["units"]]
    since = {d["id"]: d["fit_since"] for d in cfg["rooms"] if d.get("fit_since")}
    return fit.layout_for(rooms, units, since)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=9)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "reviews", "local_fit.json"))
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "..", "reviews", "config_snapshot.json")))
    cfg = cfg.get("data", cfg)
    rows = load_rows(a.days)
    layout = layout_from_config(cfg)
    print(f"{len(rows)} minutes of log, rooms {sorted(layout['rooms'])}", flush=True)
    res = {}
    for name, kw in (("with pulse", {"pulse": True}), ("without pulse", {"pulse": False})):
        t0 = time.time()
        m, rep = fit.fit(rows, layout, **kw)
        res[name] = {"model": m.to_dict(), "holdout": rep["holdout"], "pulse": {r: rep[r].get("pulse") for r in m.rooms if isinstance(rep.get(r), dict)}}
        b3 = m.rooms.get("bedroom_3")
        print(f"[{name}] {time.time() - t0:.0f}s  usable={m.meta['usable']}  B3 pulse {b3.leak_pulse} F/h tau {b3.leak_pulse_tau_h * 60:.0f} min, leak {b3.leak}", flush=True)
        for r, v in sorted(rep["holdout"].items()):
            print(f"    {r:16s} held-out {v['model_rmse_F']} vs nothing-changes {v['no_change_rmse_F']} (n={v['n']})", flush=True)
        print("    pulse report:", {r: p for r, p in res[name]["pulse"].items() if p}, flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
