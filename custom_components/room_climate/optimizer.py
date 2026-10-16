"""Model-predictive choice of house mode and unit setpoints.

For each mode the house could be in (one mode for every unit: they share a compressor), search unit setpoints and
simulate the next couple of hours with the thermal model (model.py). The cost is what the current controller already
minimizes, summed over the horizon instead of judged on the spot:

    sum over rooms and steps of  priority x (distance outside the comfort band)^2   [+ the hard floor/ceiling terms]

plus a charge for switching heat <-> cool, for moving setpoints, and a little for energy. Priorities (and their
schedules) are the weights, so "bedroom matters at night, first floor during the day" is part of the objective.

Pure Python. The search is coordinate descent over a small grid of setpoints around each unit's zone target, which
is plenty for three units and keeps a plan to well under a second on an HA Green.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

try:
    from . import core
    from . import model as hm
except ImportError:  # offline (sim, tests, tools) where core.py/model.py are top-level modules
    import core  # type: ignore
    import model as hm  # type: ignore


@dataclass
class PlanSettings:
    horizon_h: float = 2.0
    step_min: int = 10
    substeps: int = 2  # model steps per plan step (unit on/off decisions need a finer step than 10 min)
    offsets: tuple = (-3, -2, -1, 0, 1, 2, 3)  # setpoint candidates around each unit's zone target (F)
    passes: int = 2
    switch_cost: float = 3.0  # charged once for a heat <-> cool switch (weighted F^2 * h); tuned on the simulated house
    change_cost: float = 0.05  # per F^2 of setpoint change on a unit
    energy_cost: float = 0.02  # per hour of full output on a unit
    terminal_h: float = 0.5  # the end of the horizon counts double, so a plan has to settle, not just pass through
    first_block_min: int = 30  # setpoints are planned in two blocks: this long now (what gets sent), then the rest,
    #                            which may differ (e.g. after a schedule change), so the near term isn't sacrificed


@dataclass
class Plan:
    mode: str
    setpoints: dict  # unit id -> F, to send now
    cost: float
    by_mode: dict = field(default_factory=dict)  # mode -> {"cost", "setpoints", "later"}
    predicted: dict = field(default_factory=dict)  # room id -> [F at each step] for the chosen plan
    reason: str = ""
    later: dict = field(default_factory=dict)  # unit id -> F planned for after the first block


def band_cost(temp: float, target: float, weight: float, settings: core.Settings) -> float:
    """Same cold + hot cost as core.room_costs for a known temperature (no prediction, no free cooling)."""
    cold = weight * max(0.0, (target - settings.band_below) - temp) ** 2 + settings.hard_weight * max(0.0, settings.floor - temp) ** 2
    hot = weight * max(0.0, temp - (target + settings.band_above)) ** 2 + settings.hard_weight * max(0.0, temp - settings.ceiling) ** 2
    return cold + hot


def _pad(seq, n):
    seq = list(seq) or [None]
    return (seq + [seq[-1]] * n)[:n]


def plan(m: hm.HouseModel, rooms: list, units: list, temps: dict, now: datetime, house: core.HouseState,
         setpoints_now: dict, running: dict, outdoor_seq, sun_seq, settings: core.Settings = core.Settings(),
         ps: PlanSettings = PlanSettings()) -> Plan:
    """Best house mode and setpoints for the next `horizon_h`. `temps` are current room temperatures (rooms the model
    knows), `running` which units are actually running now, `outdoor_seq`/`sun_seq` the expected weather per step."""
    n = max(1, int(round(ps.horizon_h * 60 / ps.step_min)))
    dt_h = ps.step_min / 60
    times = [now + timedelta(minutes=ps.step_min * (k + 1)) for k in range(n)]
    outdoor, sun = _pad(outdoor_seq, n), [s or 0.0 for s in _pad(sun_seq, n)]
    scored = [r for r in rooms if r.unit is not None and r.id in m.rooms and temps.get(r.id) is not None]
    # Targets and weights over the horizon don't depend on the plan: work them out once.
    tw = {r.id: [(core.target_temp(r, t), core.room_priority(r, t)) for t in times] for r in scored}
    end_weight = [2.0 if (times[-1] - t).total_seconds() < ps.terminal_h * 3600 else 1.0 for t in times]
    known_units = [u for u in units if u.id in m.units]

    b1 = min(n, max(1, int(round(ps.first_block_min / ps.step_min))))
    blocks = [(0, b1)] + ([(b1, n)] if b1 < n else [])

    def base_setpoint(u, k) -> float:
        """Priority-weighted target of the unit's rooms at step k, as a setpoint on the unit's own reading."""
        mine = [r for r in scored if r.unit == u.id]
        if not mine:
            return float(round(setpoints_now.get(u.id) or 68.0))
        wsum = sum(tw[r.id][k][1] for r in mine)
        zone = sum(tw[r.id][k][0] * tw[r.id][k][1] for r in mine) / wsum
        return float(round(zone + m.units[u.id].sensor_offset))

    def clamp(u, v):
        return float(max(u.min_setpoint, min(u.max_setpoint, v)))

    cache = {}

    def evaluate(mode, sps):
        """sps: unit id -> (setpoint per block)."""
        key = (mode, tuple(sorted(sps.items())))
        if key in cache:
            return cache[key]
        run = {uid: (running.get(uid, False) and mode == house.mode) for uid in m.units}
        traj, outs, cur = [], [], dict(temps)
        for bi, (a, b) in enumerate(blocks):
            tr, ou, run = hm.rollout(m, cur, mode, {uid: v[bi] for uid, v in sps.items()}, run, outdoor[a:b], sun[a:b],
                                     dt_h, ps.substeps)
            traj += tr
            outs += ou
            cur = tr[-1]
        c = 0.0
        for k, tk in enumerate(traj):
            for r in scored:
                tgt, w = tw[r.id][k]
                c += band_cost(tk[r.id], tgt, w, settings) * dt_h * end_weight[k]
            c += ps.energy_cost * sum(abs(y) for y in outs[k].values()) * dt_h
        for uid, v in sps.items():
            prev = setpoints_now.get(uid)
            if mode == house.mode and prev is not None:
                c += ps.change_cost * (v[0] - prev) ** 2
            for x, y in zip(v, v[1:]):
                c += ps.change_cost * (y - x) ** 2
        cache[key] = (c, traj)
        return cache[key]

    modes = []
    if house.mode in ("heat", "cool"):
        modes.append(house.mode)
        other = "cool" if house.mode == "heat" else "heat"
        held = core.elapsed(now, house.since) if house.since else 1e9
        if held >= settings.min_mode_s and (settings.allow_cool if other == "cool" else settings.allow_heat):
            modes.append(other)
    else:
        modes = ["off"] + [md for md, ok in (("heat", settings.allow_heat), ("cool", settings.allow_cool)) if ok]

    by_mode, best = {}, None
    for mode in modes:
        if mode == "off":
            sps = {}
            c, traj = evaluate(mode, sps)
        else:
            base = {u.id: tuple(clamp(u, base_setpoint(u, a)) for a, _ in blocks) for u in known_units}
            sps = dict(base)
            c, traj = evaluate(mode, sps)
            for _ in range(ps.passes):
                improved = False
                for u in known_units:
                    for bi in range(len(blocks)):
                        for off in ps.offsets:
                            v = list(sps[u.id])
                            v[bi] = clamp(u, base[u.id][bi] + off)
                            cand = dict(sps)
                            cand[u.id] = tuple(v)
                            if cand[u.id] == sps[u.id]:
                                continue
                            cc, tt = evaluate(mode, cand)
                            if cc < c - 1e-9:
                                c, traj, sps, improved = cc, tt, cand, True
                if not improved:
                    break
            if house.mode in ("heat", "cool") and mode != house.mode:
                c += ps.switch_cost
        by_mode[mode] = {"cost": round(c, 3), "setpoints": {uid: v[0] for uid, v in sps.items()},
                         "later": {uid: v[-1] for uid, v in sps.items()}}
        if best is None or c < best[1]:
            best = (mode, c, sps, traj)

    mode, c, sps, traj = best
    others = ", ".join(f"{md} {v['cost']:.2f}" for md, v in by_mode.items() if md != mode)
    return Plan(mode, by_mode[mode]["setpoints"], c, by_mode, {r.id: [round(t[r.id], 2) for t in traj] for r in scored},
                f"{mode}: cost {c:.2f}" + (f" (vs {others})" if others else ""), by_mode[mode]["later"])
