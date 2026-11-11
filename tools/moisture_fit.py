"""Fit the moisture model (custom_components/room_climate/moisture.py) on the real log and report how well it predicts.

    python3 tools/moisture_fit.py [--days 9]

The minute log does not record humidity before v0.28.0, so the humidity of the H5100 sensors comes from Home Assistant's own
recorder (reviews/moisture/rh_history.json and out_history.json, fetched over the REST history API: see the fetch snippet in the
session notes) and is merged into the minute-log rows by time. Results: reviews/moisture/model.json and a table on stdout.
"""
import argparse, bisect, gzip, json, os, sys
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))
import moisture as mo  # noqa: E402

MIRROR = os.path.expanduser("~/.local/share/room-climate-watch/ha/room_climate_data/")
OUT = os.path.join(HERE, "..", "reviews", "moisture")


def load_rows(days):
    rows = []
    now = datetime.now().astimezone()
    for k in range(days, -1, -1):
        day = (now - timedelta(days=k)).strftime("%Y-%m-%d")
        for p in (MIRROR + day + ".jsonl", MIRROR + day + ".jsonl.gz"):
            if os.path.exists(p):
                op = gzip.open if p.endswith(".gz") else open
                rows += [json.loads(l) for l in op(p, "rt") if l.startswith("{")]
    return sorted({r["t"]: r for r in rows}.values(), key=lambda r: r["t"])


def hold(series):
    ts = [datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() for t, _ in series]
    return ts, [v for _, v in series]


def at(h, t):
    ts, vs = h
    i = bisect.bisect_right(ts, t) - 1
    return None if i < 0 else vs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=9)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "..", "reviews", "config_now.json")))
    rh = {k: hold([(t, float(v)) for t, v in pts if v.replace(".", "", 1).isdigit()]) for k, pts in json.load(open(os.path.join(OUT, "rh_history.json"))).items() if pts}
    outw = json.load(open(os.path.join(OUT, "out_history.json")))
    orh = hold([(t, float(h)) for t, h, _ in outw if h is not None])
    room_rh = {}
    for r in cfg["rooms"]:
        s = r.get("sensor") or ""
        if s.endswith("_temperature") and s.replace("_temperature", "_humidity") in rh:
            room_rh[r["id"]] = rh[s.replace("_temperature", "_humidity")]
    rows = load_rows(a.days)
    t_first = min(h[0][0] for h in room_rh.values())
    rooms = sorted(room_rh)
    units = sorted({u["id"] for u in cfg["units"]})
    kept = []
    for row in rows:
        t = datetime.fromisoformat(row["t"]).timestamp()
        if t < t_first:
            continue
        for rid in rooms:
            v = (row.get("rooms") or {}).get(rid)
            if v is not None:
                v["rh"] = at(room_rh[rid], t)
        row["out_rh"] = at(orh, t)
        kept.append(row)
    print(f"{len(kept)} minutes with humidity, rooms {rooms}")
    S = mo.series(kept, rooms, units)
    n = len(S["t"])
    print(f"{n} ten-minute steps from {S['t'][0]:%m-%d %H:%M} to {S['t'][-1]:%m-%d %H:%M}")
    cut = int(n * 0.75)
    train = {k: ({r: v[:cut] for r, v in x.items()} if isinstance(x, dict) else x[:cut]) for k, x in S.items()}
    ridge = float(os.environ.get("RIDGE", "1e-4"))
    m = mo.fit(train, rooms, units, ridge=ridge)
    ev = mo.evaluate(m, S, cut)
    print("\nheld-out (last 25%): rmse in RH points, model vs nothing-changes (n origins)")
    print(f"{'room':16s}" + "".join(f"{h:>8d} min" + " " * 14 for h in (60, 120)))
    for r in rooms:
        print(f"{r:16s}" + "".join(f"  {ev[h][r][0]:5.2f} vs {ev[h][r][1]:5.2f} ({ev[h][r][2]})   " for h in (60, 120) if r in ev[h]))
    full = mo.fit(S, rooms, units, ridge=ridge)
    full.report = {"held_out": {str(h): {r: list(v) for r, v in d.items()} for h, d in ev.items()}}
    json.dump(full.to_dict(), open(os.path.join(OUT, "model.json"), "w"))
    print("\nfitted on all steps:")
    for r in rooms:
        p = full.rooms[r]
        print(f"  {r:16s} ventilation a={p['a']:.2f}/h  source s0={p['s0']*1000:+.2f} kPa/kh  daily amp={((p['ss']**2+p['sc']**2)**.5)*1000:.2f}  exchange {', '.join(f'{k}:{c:.2f}' for k, c in sorted(p['c'].items(), key=lambda x: -x[1])[:3])}  drying {p['d']}")


if __name__ == "__main__":
    main()
