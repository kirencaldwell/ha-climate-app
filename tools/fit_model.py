"""Fit the thermal model to minute logs from the command line (HA refits nightly with the same code: fit.py).

    python3 tools/fit_model.py <log dir or .jsonl> [layout.json] -o model.json [--window 30] [--holdout 0.25]

Logs: the HA long-term log (/config/room_climate_data, plain or gzipped days) or simulator rows in the same shape.
layout.json (optional; see fit.fit) says which rooms to fit, their possible neighbours and units; without it every
room in the log is fitted against every other room and unit, and units' sensor rooms are detected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
import datalog  # noqa: E402
import fit  # noqa: E402


def load_rows(src: str) -> list:
    if os.path.isdir(src):
        return list(datalog.read_rows(src))
    with open(src) as f:
        return [json.loads(line) for line in f if line.strip()]


def layout_from_rows(rows: list) -> dict:
    rooms = sorted({r for row in rows for r, v in row.get("rooms", {}).items() if v.get("temp") is not None})
    units = sorted({u for row in rows for u in row.get("units", {})})
    zone = {u: sorted({r for row in rows for r, v in row.get("rooms", {}).items() if v.get("unit") == u and r in rooms})
            for u in units}
    return {"rooms": {r: {"units": units, "neighbors": [o for o in rooms if o != r]} for r in rooms},
            "units": {u: {"zone": zone[u]} for u in units}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("layout", nargs="?")
    ap.add_argument("-o", "--out", default="model.json")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--lam", type=float, default=1e-3)
    a = ap.parse_args()
    rows = load_rows(a.logs)
    layout = json.load(open(a.layout)) if a.layout else layout_from_rows(rows)
    m, rep = fit.fit(rows, layout, a.window, holdout=a.holdout, lam=a.lam)
    json.dump(m.to_dict(), open(a.out, "w"), indent=2)
    print(f"{len(rows)} rows -> {a.out}  usable: {m.meta['usable']}")
    for rid, r in m.rooms.items():
        print(f"  {rid:16s} loss {r.loss:.3f}/h  solar {r.solar:5.2f}  bias {r.bias:+.2f}  heat {r.heat}  cool {r.cool}"
              f"  coupling {r.coupling}  | {rep.get(rid)}")
    for rid, v in rep.items():
        if rid not in m.rooms and rid != "holdout":
            print(f"  {rid:16s} {v}")
    print("  units:", {u: (v.sensor_rooms, v.sensor_offset) for u, v in m.units.items()})
    print("  2 h ahead on held-out data:", json.dumps(rep["holdout"]))
    if not m.meta["usable"]:
        print("  NOT USABLE YET: doesn't beat 'nothing changes' for every room. More (and steadier) data needed.")


if __name__ == "__main__":
    main()
