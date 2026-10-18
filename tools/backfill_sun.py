"""Add estimated sunshine (NOAA HRRR via Open-Meteo: ghi, dni, dhi in W/m2) to minute-log rows that don't have it.

    python3 tools/backfill_sun.py <log dir> <out dir> --lat 47.68 --lon -122.38

Reads every day file in <log dir> (plain .jsonl or gzipped), writes the same files to <out dir> with each row's "sun"
gaining ghi/dni/dhi (interpolated from the 15-minute values, each the mean of the quarter hour before its time stamp).
Rows that already have them are left as they are. Open-Meteo keeps about three months of past HRRR values.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import urllib.request
from datetime import datetime

URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&models=gfs_hrrr&timeformat=unixtime"
       "&minutely_15=shortwave_radiation,direct_normal_irradiance,diffuse_radiation&past_days={days}&forecast_days=1")


def fetch(lat: float, lon: float, days: int) -> list:
    with urllib.request.urlopen(URL.format(lat=lat, lon=lon, days=days), timeout=60) as r:
        m = json.loads(r.read())["minutely_15"]
    return [(t - 450, g, n, f) for t, g, n, f in zip(m["time"], m["shortwave_radiation"],
                                                     m["direct_normal_irradiance"], m["diffuse_radiation"])
            if None not in (g, n, f)]


def at(pts: list, t: float):
    lo, hi = 0, len(pts) - 1
    if not pts or t < pts[0][0] or t > pts[-1][0]:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (ta, *a), (tb, *b) = pts[lo], pts[hi]
    w = 0.0 if tb == ta else (t - ta) / (tb - ta)
    return tuple(round(x + (y - x) * w) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("out")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.logs) if f.endswith((".jsonl", ".jsonl.gz")))
    first = datetime.strptime(files[0][:10], "%Y-%m-%d")
    pts = fetch(a.lat, a.lon, min(92, (datetime.now() - first).days + 2))
    os.makedirs(a.out, exist_ok=True)
    for name in files:
        op = gzip.open if name.endswith(".gz") else open
        added = total = 0
        lines = []
        with op(os.path.join(a.logs, name), "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    lines.append(line.rstrip("\n"))  # keep a damaged line as it was
                    continue
                total += 1
                sun = row.setdefault("sun", {})
                if sun.get("ghi") is None:
                    v = at(pts, datetime.fromisoformat(row["t"]).timestamp())
                    if v is not None:
                        sun["ghi"], sun["dni"], sun["dhi"] = v
                        added += 1
                lines.append(json.dumps(row, separators=(",", ":")))
        with op(os.path.join(a.out, name), "wt") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"{name}: {added} of {total} rows given sunshine")


if __name__ == "__main__":
    main()
