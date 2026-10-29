"""Joint run/fan planner: choose, for the next two hours, WHEN each unit runs and at what fan speed.

Why this and not setpoint targets. The units are inverters whose output hardly depends on the setpoint gap (fitted k1 is
0.003 for Bedroom 3's minisplit, 0.02 for the Primary's): once the gap is big enough to start them they run at about the
same level, and the setpoint mostly decides when they stop (about 1 F past it). What changes how hard a minisplit works is
its fan. So the real decisions are run / stop per time block, and fan. The setpoint is then derived: the unit's own
reading plus a push big enough to make it start (or stop) and hold for the block. There is no feedback loop on the
setpoint; fast disturbances are handled by replanning (the caller replans every 15 min, and at once when the readings
leave the prediction).

Optimization. Decisions are categorical per unit and block (off, or run at a fan speed), blocks of 15 min for the first
hour then 30 min (2 h in all). The cost is simulated by the thermal model (model.py) with the units' own on/off
hysteresis, lag and the shared-compressor leak, so the units' coupling through the leak and the discontinuities are
inside the cost. Joint search by the cross-entropy method (sample patterns for all units at once, keep the best few,
re-fit the sampling distribution, repeat), seeded with the previous plan shifted in time, then a one-variable polish.
It is stochastic but seeded, so it is repeatable.

Cost = what optimizer.py charges (priority-weighted squared distance outside each room's comfort band over the
horizon, hard floor/ceiling, hot misses count extra, a small pull toward target, the end of the horizon counted double)
+ a charge for every unit start and every compressor start (cycling) + a little for energy and for changing a unit's
pattern + a charge for switching heat <-> cool (with a margin, so a near tie doesn't flip the house).

Pure Python.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

try:
    from . import core
    from . import model as hm
    from . import optimizer
except ImportError:  # offline (sim, tests, tools)
    import core  # type: ignore
    import model as hm  # type: ignore
    import optimizer  # type: ignore


@dataclass
class JointSettings:
    block_min: tuple = (15, 15, 15, 15, 30, 30)
    step_min: int = 5  # model step; the units' on/off decisions are made at this rate
    push_run: float = 3.0  # F past the unit's reading: starts it and keeps it running through the block
    push_stop: float = 2.0  # F behind the reading: makes it stop (it idles ~1 F past its setpoint)
    population: int = 60
    elite: int = 8
    iters: int = 5
    smooth: float = 0.6  # weight of the new elite distribution against the old one
    unit_start_cost: float = 0.6  # per indoor-unit start (tuned in simulation: tools/mc_suite.py, 'tune')
    compressor_start_cost: float = 1.6  # per start of the shared compressor (every unit idle -> one running)
    change_cost: float = 0.02  # per block-to-block change of a unit's run/fan pattern
    energy_cost: float = 0.02  # per hour of full output
    aim_weight: float = 0.1
    terminal_h: float = 0.5
    switch_cost: float = 3.0
    switch_margin: float = 0.15
    sleep_option: bool = True  # a unit that isn't needed may be switched fully OFF for a block, not just idle in the house mode:
    #                            the 10-27/28 experiment showed an idle-in-heat unit passes the other compressor's heat to its room
    #                            (Bedroom 3 +2 F in an hour of the Primary's run; fully off: +0.5, about the sun)
    max_fans: int = 3  # run options per fan-controlled unit besides "off": auto plus the lowest and a middle fan
    seed: int = 1


@dataclass
class JointPlan:
    """Same fields as optimizer.Plan (so callers can treat them alike) plus the pattern and fans."""
    mode: str
    setpoints: dict
    cost: float
    by_mode: dict = field(default_factory=dict)
    predicted: dict = field(default_factory=dict)  # room id -> [F at each model step]
    reason: str = ""
    later: dict = field(default_factory=dict)
    comfort: float = 0.0
    first: dict = field(default_factory=dict)
    fans: dict = field(default_factory=dict)  # unit id -> fan for the first block (None: leave as is)
    pattern: dict = field(default_factory=dict)  # unit id -> ["off" | "run:<fan>"] per block
    step_min: int = 5
    evals: int = 0
    sleep: dict = field(default_factory=dict)  # unit id -> True when its first block is "fully off" (mode off, no leak)


class JointPlanner:
    def __init__(self, js: JointSettings = None):
        self.js = js or JointSettings()
        self.prev = None  # (time, mode, {unit: [option label per block]}) of the last plan, for warm starts

    # -- helpers --
    def _options(self, m, u, fan_units):
        """Per unit: [(run?, fan)]; the first is "off"."""
        fans = [None]
        if u.id in fan_units and u.id in m.units:
            sc = m.units[u.id].fan_scale
            rest = sorted((f for f in sc if f != "auto"), key=lambda f: sc[f])  # weakest first
            pick = [rest[0]] + ([rest[len(rest) // 2]] if len(rest) > 2 else []) if rest else []
            fans = ["auto"] + [f for f in pick if f][:max(0, self.js.max_fans - 1)]
        return [(False, None)] + [(True, f) for f in fans] + ([(False, "sleep")] if self.js.sleep_option else [])

    def plan(self, m: hm.HouseModel, rooms: list, units: list, temps: dict, now: datetime, house: core.HouseState,
             running: dict, outdoor, sun_at, settings: core.Settings = core.Settings(), ignore=(), fixed=None,
             lagged=None, fans=None, masses=None, offsets=None, leak_scale=None, effect_scale=None,
             unit_temps=None, comp=None) -> JointPlan:
        js = self.js
        fixed = fixed or {}
        blocks, k = [], 0
        for bm in js.block_min:
            n = max(1, bm // js.step_min)
            blocks.append((k, k + n))
            k += n
        n_steps = k
        dt_h = js.step_min / 60
        times = [now + timedelta(minutes=js.step_min * (i + 1)) for i in range(n_steps)]
        outdoor_seq = [outdoor(t) if callable(outdoor) else outdoor for t in times]
        sun_seq = [hm.as_sun(sun_at(t) if callable(sun_at) else sun_at) for t in times]
        scored = [r for r in rooms if r.unit is not None and r.id in m.rooms and temps.get(r.id) is not None
                  and r.id not in ignore]
        tw = {r.id: [(core.target_temp(r, t), core.room_priority(r, t)) for t in times] for r in scored}
        end_w = [2.0 if (times[-1] - t).total_seconds() < js.terminal_h * 3600 else 1.0 for t in times]
        known = [u for u in units if u.id in m.units and u.id not in fixed]
        fan_units = {x.strip() for x in settings.fan_units.split(",")}
        opts = {u.id: self._options(m, u, fan_units) for u in known}
        label = {u.id: [("sleep" if f == "sleep" else "off" if not r else f"run:{f or '-'}") for r, f in opts[u.id]] for u in known}
        nb = len(blocks)
        zone = {u.id: [r for r in scored if r.unit == u.id] for u in known}
        evals = [0]
        cache = {}

        def sp_for(u, mode, reading, run):
            if reading is None:
                return None
            s = 1.0 if mode == "heat" else -1.0
            v = reading + s * (js.push_run if run else -js.push_stop)
            return core.to_grid(max(u.min_setpoint, min(u.max_setpoint, v)))

        def evaluate(mode, pattern):
            key = (mode, tuple((uid, tuple(p)) for uid, p in sorted(pattern.items())))
            if key in cache:
                return cache[key]
            evals[0] += 1
            run_now = {uid: running.get(uid, False) and (uid in fixed or mode == house.mode) for uid in m.units}
            modes_by_unit = {uid: (fixed[uid][0] if uid in fixed else mode) for uid in m.units}
            cur, lag, mass = dict(temps), dict(lagged) if lagged is not None else None, dict(masses or {})
            comp_state = dict(comp) if comp is not None else None
            run = dict(run_now)
            traj, outs, first_sp, first_fan = [], [], {}, {}
            for bi, (a, b) in enumerate(blocks):
                sps, fan_now, bmodes = {}, dict(fans or {}), dict(modes_by_unit)
                for u in known:
                    runb, fan = opts[u.id][pattern[u.id][bi]]
                    if fan == "sleep":  # fully off for this block: no setpoint, no leak
                        bmodes[u.id], sps[u.id] = "off", None
                        if bi == 0:
                            first_sp[u.id], first_fan[u.id] = None, "sleep"
                        continue
                    rd = unit_temps.get(u.id) if (bi == 0 and unit_temps and unit_temps.get(u.id) is not None) \
                        else hm.unit_reading(m, u.id, cur)
                    sps[u.id] = sp_for(u, mode, rd, runb)
                    if runb and fan:
                        fan_now[u.id] = fan
                    if bi == 0:
                        first_sp[u.id], first_fan[u.id] = sps[u.id], (fan if runb else None)
                sps.update({uid: f[1] for uid, f in fixed.items()})
                tr, ou, run = hm.rollout(m, cur, bmodes, sps, run, outdoor_seq[a:b], sun_seq[a:b], dt_h, 1, lag,
                                         None, fans=fan_now, masses=mass, offsets=offsets, leak_scale=leak_scale,
                                         effect_scale=effect_scale, comp=comp_state)
                traj += tr
                outs += ou
                cur = tr[-1]
            c = 0.0
            for i, tk in enumerate(traj):
                for r in scored:
                    tgt, w = tw[r.id][i]
                    t = tk[r.id]
                    aim = js.aim_weight * w * (t - (tgt - settings.aim_offset)) ** 2
                    c += (optimizer.band_cost(t, tgt, w, settings) + aim) * dt_h * end_w[i]
                c += js.energy_cost * sum(abs(y) for y in outs[i].values()) * dt_h
            on = {uid: run_now.get(uid, False) for uid in m.units}
            comp_on = any(on.values())
            for i in range(n_steps):
                now_on = {uid: abs(outs[i].get(uid, 0.0)) > 1e-9 for uid in m.units}
                for uid in m.units:
                    if now_on[uid] and not on[uid]:
                        c += js.unit_start_cost
                if any(now_on.values()) and not comp_on:
                    c += js.compressor_start_cost
                on, comp_on = now_on, any(now_on.values())
            for u in known:
                p = pattern[u.id]
                c += js.change_cost * sum(1 for x, y in zip(p, p[1:]) if x != y)
            cache[key] = (c, traj, first_sp, first_fan)
            return cache[key]

        # -- modes to compare (same rules as optimizer.plan) --
        forced = next((f[0] for f in fixed.values() if f[0] in ("heat", "cool")), None)
        if forced:
            modes = [forced]
        elif house.mode in ("heat", "cool"):
            modes = [house.mode]
            other = "cool" if house.mode == "heat" else "heat"
            held = core.elapsed(now, house.since) if house.since else 1e9
            if held >= settings.min_mode_s and (settings.allow_cool if other == "cool" else settings.allow_heat):
                modes.append(other)
        else:
            modes = ["off"] + [md for md, ok in (("heat", settings.allow_heat), ("cool", settings.allow_cool)) if ok]

        rng = random.Random(js.seed * 1000003 + int(now.timestamp() // 60))
        results, by_mode = {}, {}
        for mode in modes:
            if mode == "off":
                pat = {u.id: [0] * nb for u in known}
                c, traj, fsp, ffan = evaluate(mode, pat)
                results[mode] = (c, pat, traj, fsp, ffan)
            else:
                results[mode] = self._search(mode, known, opts, nb, evaluate, zone, tw, rng, house, now, label, temps)
            c, pat, traj, fsp, ffan = results[mode]
            if house.mode in ("heat", "cool") and mode != house.mode:
                c += js.switch_cost
            by_mode[mode] = {"cost": round(c, 3), "setpoints": dict(fsp), "later": {}, "first": dict(fsp)}
            results[mode] = (c, pat, traj, fsp, ffan)
        best = min(modes, key=lambda md: results[md][0])
        if (best != house.mode and house.mode in results and not forced
                and results[best][0] > results[house.mode][0] * (1.0 - js.switch_margin)):
            best = house.mode  # not enough better to be worth switching
        c, pat, traj, fsp, ffan = results[best]
        comfort = 0.0
        for i, tk in enumerate(traj):
            for r in scored:
                tgt, w = tw[r.id][i]
                comfort += optimizer.band_cost(tk[r.id], tgt, w, settings)
        comfort /= max(1, len(traj))
        pattern = {uid: [label[uid][i] for i in p] for uid, p in pat.items()}
        self.prev = (now, best, pattern)
        others = ", ".join(f"{md} {v['cost']:.2f}" for md, v in by_mode.items() if md != best)
        return JointPlan(best, dict(fsp), c, by_mode, {r.id: [round(t[r.id], 2) for t in traj] for r in scored},
                         f"{best}: cost {c:.2f}" + (f" (vs {others})" if others else ""), {}, round(comfort, 3), dict(fsp),
                         {u: (None if f == "sleep" else f) for u, f in ffan.items()}, pattern, js.step_min, evals[0],
                         {u: f == "sleep" for u, f in ffan.items() if f == "sleep"})

    def _search(self, mode, known, opts, nb, evaluate, zone, tw, rng, house, now, label, temps):
        js = self.js
        uids = [u.id for u in known]
        # sampling distribution: probability of each option per unit and block
        prob = {uid: [[1.0 / len(opts[uid])] * len(opts[uid]) for _ in range(nb)] for uid in uids}
        seeds = [{uid: [0] * nb for uid in uids}]
        # greedy seed: a unit runs while its rooms sit below (heat) / above (cool) their targets
        g = {}
        for u in known:
            rs = [r for r in zone[u.id] if temps.get(r.id) is not None]
            gap = 0.0
            if rs:
                gap = sum((tw[r.id][0][0] - temps[r.id]) * tw[r.id][0][1] for r in rs) / sum(tw[r.id][0][1] for r in rs)
            need = gap > 0.7 if mode == "heat" else gap < -0.7
            g[u.id] = [1 if need else 0] * nb  # option 1 is "run" at the first fan option (auto)
        seeds.append(g)
        # previous plan, shifted by the time elapsed
        if self.prev and self.prev[1] == mode:
            t0, _, pat = self.prev
            shift = int(((now - t0).total_seconds() / 60) // js.block_min[0])
            warm = {}
            for uid in uids:
                lab = pat.get(uid)
                if not lab:
                    continue
                idx = [min(len(lab) - 1, i + shift) for i in range(nb)]
                warm[uid] = [label[uid].index(lab[i]) if lab[i] in label[uid] else 0 for i in idx]
            if len(warm) == len(uids):
                seeds.append(warm)
                for uid in uids:
                    for bi in range(nb):
                        w = warm[uid][bi]
                        prob[uid][bi] = [0.25 / max(1, len(opts[uid]) - 1)] * len(opts[uid])
                        prob[uid][bi][w] = 0.75
        best = None
        for s in seeds:
            c, traj, fsp, ffan = evaluate(mode, s)
            if best is None or c < best[0]:
                best = (c, s, traj, fsp, ffan)
        for _ in range(js.iters):
            pop = []
            for _ in range(js.population):
                cand = {uid: [self._draw(rng, prob[uid][bi]) for bi in range(nb)] for uid in uids}
                c, traj, fsp, ffan = evaluate(mode, cand)
                pop.append((c, cand, traj, fsp, ffan))
                if c < best[0]:
                    best = (c, cand, traj, fsp, ffan)
            pop.sort(key=lambda x: x[0])
            elite = pop[:js.elite]
            for uid in uids:
                for bi in range(nb):
                    cnt = [0.0] * len(opts[uid])
                    for _, cand, *_ in elite:
                        cnt[cand[uid][bi]] += 1.0 / len(elite)
                    prob[uid][bi] = [js.smooth * q + (1 - js.smooth) * p for q, p in zip(cnt, prob[uid][bi])]
                    s = sum(prob[uid][bi])
                    prob[uid][bi] = [max(0.02, p / s) for p in prob[uid][bi]]
        # polish: one variable at a time from the best
        improved = True
        passes = 0
        while improved and passes < 2:
            improved, passes = False, passes + 1
            for uid in uids:
                for bi in range(nb):
                    for o in range(len(opts[uid])):
                        if o == best[1][uid][bi]:
                            continue
                        cand = {u: list(p) for u, p in best[1].items()}
                        cand[uid][bi] = o
                        c, traj, fsp, ffan = evaluate(mode, cand)
                        if c < best[0] - 1e-9:
                            best, improved = (c, cand, traj, fsp, ffan), True
        c, pat, traj, fsp, ffan = best
        return c, pat, traj, fsp, ffan

    @staticmethod
    def _draw(rng, p):
        x, acc = rng.random(), 0.0
        for i, q in enumerate(p):
            acc += q
            if x <= acc:
                return i
        return len(p) - 1
