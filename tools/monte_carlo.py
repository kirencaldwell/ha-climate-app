"""Monte Carlo: how do the planner's feedback pieces cope with model error of the size we actually have?

    python3 tools/monte_carlo.py CONFIG.json [--n 24] [--hours 15] [--scale 1] [--workers 9] [--out results.json]

CONFIG is a Room Climate config (room_climate/get, or .storage/room_climate): the real rooms, schedules and priorities.
Each draw samples a sim/disturbed.py Disturbance (a plant that differs from the planner's model by amounts seen in the
real logs), then runs every controller variant on that same plant, weather and start, so differences between variants
are paired. The run starts at 17:00 and covers the evening, the night and the morning: where the leak and the cold
kitchen showed up.

Variants: today (the everyday controller) | base (the planner as deployed 2026-10-21) | naive (+ drift estimated from
every window: the first, unstable version, kept as a negative control) | offs (+ drift learned only while nothing acts
on a room) | gain (+ per-unit effect gains) | gate (+ switch margin and confirmation) | all (offs + gain +
gate: what ships) | all+leak (also learns the leak scale: helped in sim, hurt on the real log) | oracle (planner given the true plant).

Stability: besides comfort, each run reports mode switches, "chatter" (a switch less than 60 min after the previous
one: the limit cycle seen on 2026-10-17..22) and the most switches in any 6 h window. --cycle replaces the random draw
with one built to provoke that cycle (a large leak that spikes as the compressor starts, a poorly heated kitchen, cold
nights), with small random variations.
"""
import argparse, json, math, os, random, statistics, sys, time
from datetime import datetime, timedelta
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow stdlib select
import coupled  # noqa: E402
import core  # noqa: E402
import disturbed  # noqa: E402
import model as hm  # noqa: E402
import observer  # noqa: E402
import optimizer  # noqa: E402

VARIANTS = ("today", "base", "naive", "offs", "gain", "gate", "all", "all+leak", "oracle")


class Tap:
    """Wraps a controller to record its house mode every tick."""

    def __init__(self, ctl):
        self.ctl, self.name, self.modes = ctl, ctl.name, []

    def tick(self, sim, now):
        m = self.ctl.tick(sim, now)
        self.modes.append((now, m))
        return m


def switch_stats(modes):
    """(switches, chatter, most switches in any 6 h) from [(time, mode)]."""
    ev, last = [], None
    for t, m in modes:
        if m in ("heat", "cool"):
            if last and m != last:
                ev.append(t)
            last = m
    chatter = sum(1 for a, b in zip(ev, ev[1:]) if (b - a).total_seconds() < 3600)
    worst = max([sum(1 for u in ev if t <= u < t + timedelta(hours=6)) for t in ev] or [0])
    return len(ev), chatter, worst


def controller(name, rooms, units, nominal, truth, settings):
    """`name` may carry observer settings for the gain-margin sweep: "all:noise=0.2/forget_h=3" (pairs separated by /)."""
    name, _, kv = name.partition(":")
    okw = {k: float(v) for k, v in (p.split("=") for p in kv.split("/") if p)}
    if name == "today":
        return coupled.HeuristicController(rooms, units, settings)
    if name.startswith("joint"):  # the joint run/fan planner, no outer loop (joint.py): joint | joint+obs | joint-oracle
        import joint
        import joint_ctl
        js = joint.JointSettings()
        for k, v in okw.items():  # "joint-oracle:unit_start_cost=1.0,population=80": planner settings for the tuning sweep
            setattr(js, k, type(getattr(js, k))(v))
        return joint_ctl.JointController(rooms, units, truth if name == "joint-oracle" else nominal, settings, js=js,
                                         observer=observer.Observer() if name == "joint+obs" else None, name=name)
    ps_off = optimizer.PlanSettings(switch_margin=0.0)
    ps_on = optimizer.PlanSettings()
    kw = {
        "base": dict(model=nominal, ps=ps_off, confirm_plans=1),
        "naive": dict(model=nominal, ps=ps_off, confirm_plans=1,
                      observer=observer.Observer(naive=True, leak_on=False, effect_on=False)),
        "offs": dict(model=nominal, ps=ps_off, confirm_plans=1,
                     observer=observer.Observer(leak_on=False, effect_on=False)),
        "gain": dict(model=nominal, ps=ps_off, confirm_plans=1, observer=observer.Observer(offsets_on=False)),
        "gate": dict(model=nominal, ps=ps_on, confirm_plans=3),
        "all": dict(model=nominal, ps=ps_on, confirm_plans=3, observer=observer.Observer(**okw)),
        "all+leak": dict(model=nominal, ps=ps_on, confirm_plans=3, observer=observer.Observer(leak_on=True)),
        "oracle": dict(model=truth, ps=ps_off, confirm_plans=1),
    }[name]
    m = kw.pop("model")
    return coupled.MPCController(rooms, units, m, settings, name=name, **kw)


def run_one(job):
    cfg, i, variant, seed, hours, scale, start, cycle = job[:8]
    trace = job[8] if len(job) > 8 else False
    rng = random.Random(seed * 1000 + i)
    d = disturbed.draw(rng, scale)
    if cycle:  # the limit-cycle provoker: a big, spiky leak and a poorly heated kitchen, with small variations
        d.leak_mult, d.leak_boost, d.leak_tau_min = rng.uniform(2.2, 3.0), rng.uniform(0.8, 1.5), rng.uniform(15, 30)
        d.kitchen_gain, d.b1_evening = rng.uniform(0.3, 0.5), rng.uniform(0.5, 1.5)
    nominal = coupled.the_house()
    truth = disturbed.perturbed_truth(nominal, d)
    rooms, units = coupled.rooms_units_from_config(cfg, nominal)
    settings = coupled.settings_from_config(cfg)
    settings.band_by_weight = os.environ.get("BAND_BY_WEIGHT", "1") == "1"  # A/B for the per-room dead zone
    settings.band_min_scale = float(os.environ.get("BAND_MIN", settings.band_min_scale))
    t0 = datetime.fromisoformat(start)
    w = coupled.Weather(t0, int(hours / 24) + 2, mean=rng.uniform(42, 48) if cycle else rng.uniform(44, 54),
                        seed=seed * 1000 + i)
    sim = disturbed.DisturbedSim(truth, t0, w, d, seed=seed * 1000 + i,
                                 temps={r: 66.5 + rng.uniform(-0.8, 0.8) for r in truth.rooms})
    ctl = Tap(controller(variant, rooms, units, nominal, truth, settings))
    log = []
    t = time.time()
    res = coupled.run(truth, rooms, units, ctl, t0, hours / 24, w, settings, log=log, sim=sim)
    night = [r for r in log if r["t"][11:13] >= "21" or r["t"][11:13] < "07"]
    def over(rid, rows):
        return [v["temp"] - v["target"] for r in rows for k, v in r["rooms"].items() if k == rid]
    b3 = over("bedroom_3", night) or over("bedroom_3", log)
    main = [max(0.0, v["target"] - v["temp"]) for r in log for k, v in r["rooms"].items()
            if k in ("kitchen", "living_room", "dining_area")]
    sw, chatter, worst6 = switch_stats(ctl.modes)
    def rms(rid):
        e = [v["temp"] - v["target"] for r in log for k, v in r["rooms"].items() if k == rid]
        return round((sum(x * x for x in e) / len(e)) ** 0.5, 3) if e else None
    wnum = wden = 0.0
    for r in log:
        for k, v in r["rooms"].items():
            w_ = core.room_priority(next(x for x in rooms if x.id == k), datetime.fromisoformat(r["t"]))
            wnum += w_ * (v["temp"] - v["target"]) ** 2
            wden += w_
    out = {"i": i, "variant": variant, "seconds": round(time.time() - t, 1), "discomfort": res["discomfort"],
           "switches": sw, "chatter": chatter, "worst6h": worst6, "starts": res["compressor_starts"], "energy": res["energy_unit_h"],
           "rms_b3": rms("bedroom_3"), "rms_primary": rms("primary_bedroom"), "house_rms": round((wnum / wden) ** 0.5, 3),
           "b3_night_max": round(max(b3), 2), "b3_night_mean_abs": round(sum(abs(x) for x in b3) / len(b3), 2),
           "main_cold_mean": round(sum(main) / len(main), 2), "worst": res["worst_F"],
           "hours_out": res["hours_outside_band"], "dist": {k: (v if not isinstance(v, dict) else v) for k, v in d.__dict__.items()}}
    if trace:  # a downsampled record of the run, for the report's time-series figures
        keep = [k for k in range(0, len(log), 5)]
        out["trace"] = {
            "t": [log[k]["t"] for k in keep],
            "mode": [ctl.modes[k][1] for k in keep if k < len(ctl.modes)],
            "rooms": {rid: [log[k]["rooms"][rid]["temp"] for k in keep] for rid in log[0]["rooms"]},
            "target": {rid: [log[k]["rooms"][rid]["target"] for k in keep] for rid in log[0]["rooms"]},
            "running": {u: [1 if log[k]["units"][u]["action"] in ("heating", "cooling") else 0 for k in keep]
                        for u in log[0]["units"]}}
    obs = getattr(ctl.ctl, "observer", None)
    if obs:
        out["observer"] = obs.to_state()
    if hasattr(ctl.ctl, "plans"):
        out["plans"], out["evals_per_plan"] = ctl.ctl.plans, round(ctl.ctl.evals / max(1, ctl.ctl.plans))
    return out


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(results, variants):
    by = {v: {r["i"]: r for r in results if r["variant"] == v} for v in variants}
    print(f"{'variant':8s} {'n':>3s} {'discomfort':>11s} {'B3 night max':>13s} {'B3 |miss|':>10s} {'main cold':>10s} {'starts':>7s} | "
          f"{'switches':>9s} {'p90':>4s} {'max':>4s} {'chatter':>8s} {'worst 6h':>9s} {'cycling':>8s}")
    for v in variants:
        rs = list(by[v].values())
        if not rs:
            continue
        f = lambda k: statistics.mean(r[k] for r in rs)
        sw = sorted(r["switches"] for r in rs)
        cyc = sum(1 for r in rs if r["worst6h"] >= 4)
        print(f"{v:8s} {len(rs):3d} {f('discomfort'):11.1f} {f('b3_night_max'):13.2f} {f('b3_night_mean_abs'):10.2f} "
              f"{f('main_cold_mean'):10.2f} {f('starts'):7.1f} | {f('switches'):9.1f} {sw[int(.9 * (len(sw) - 1))]:4d} {sw[-1]:4d} "
              f"{f('chatter'):8.2f} {f('worst6h'):9.2f} {cyc:3d}/{len(rs):<3d}")
    print("  (cycling: runs with 4+ mode switches inside one 6 h window)")
    if "base" in by:
        print("\npaired against base (same plant, weather, start):  mean change [10th, 90th percentile] and share of draws better")
        for v in variants:
            if v == "base" or not by[v]:
                continue
            common = sorted(set(by[v]) & set(by["base"]))
            for key, label, lower_better in (("discomfort", "discomfort", True), ("b3_night_max", "B3 night peak", True),
                                             ("switches", "mode switches", True), ("main_cold_mean", "main floor cold", True)):
                dl = [by[v][i][key] - by["base"][i][key] for i in common]
                better = sum(1 for x in dl if x < -1e-9) / len(dl)
                print(f"  {v:7s} {label:15s} {statistics.mean(dl):+8.2f}  [{pct(dl, .1):+7.2f}, {pct(dl, .9):+7.2f}]  better in {better:4.0%}")
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--hours", type=float, default=15)
    ap.add_argument("--scale", type=float, default=1.0, help="stretch every disturbance (0: perfect model)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--start", default="2026-10-19T17:00:00-07:00")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--cycle", action="store_true", help="draws built to provoke the heat/cool limit cycle")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg = cfg.get("data", cfg)
    variants = a.variants.split(",")
    jobs = [(cfg, i, v, a.seed, a.hours, a.scale, a.start, a.cycle) for i in range(a.n) for v in variants]
    t0 = time.time()
    with Pool(a.workers) as pool:
        results = []
        for k, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            results.append(r)
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} runs, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"))
    summarize(results, variants)


if __name__ == "__main__":
    main()
