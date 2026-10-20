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
    aim_weight: float = 0.1  # small pull toward target - aim_offset inside the band (otherwise energy alone would
    #                          park rooms at the band's warm edge in summer; the household prefers cool)
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
    comfort: float = 0.0  # predicted comfort score of the chosen plan: band cost per hour, averaged over the horizon
    first: dict = field(default_factory=dict)  # with the outer loop: unit id -> the setpoint it would send first


def band_cost(temp: float, target: float, weight: float, settings: core.Settings) -> float:
    """Same cold + hot cost as core.room_costs for a known temperature (no prediction, no free cooling)."""
    cold = weight * max(0.0, (target - settings.band_below) - temp) ** 2 + settings.hard_weight * max(0.0, settings.floor - temp) ** 2
    hot = settings.warm_weight * weight * max(0.0, temp - (target + settings.band_above)) ** 2 + settings.hard_weight * max(0.0, temp - settings.ceiling) ** 2
    return cold + hot


def _pad(seq, n):
    seq = list(seq) or [None]
    return (seq + [seq[-1]] * n)[:n]


def plan(m: hm.HouseModel, rooms: list, units: list, temps: dict, now: datetime, house: core.HouseState,
         setpoints_now: dict, running: dict, outdoor_seq, sun_seq, settings: core.Settings = core.Settings(),
         ps: PlanSettings = PlanSettings(), ignore=(), fixed=None, drives=None, lagged=None, fans=None,
         masses=None) -> Plan:
    """Best house mode and setpoints for the next `horizon_h`. `temps` are current room temperatures (rooms the model
    knows), `running` which units are actually running now, `outdoor_seq`/`sun_seq` the expected weather per step.
    Rooms in `ignore` are simulated (their unit may regulate on them) but don't count in the cost: e.g. a room that
    reads its own unit's sensor, which measures the unit's airflow rather than the room. Units in `fixed`
    (unit id -> (mode, setpoint)) run on their own thermostats: simulated as they are, never planned, and one in heat or
    cool sets the mode for the rest (they share a compressor).

    With the outer loop on (settings.drive), what's planned per unit is the *target* for its rooms, and the simulation
    runs core.drive_law every substep to turn it into setpoints, as the integration will: `setpoints_now` are then
    the targets planned last time, `drives` the units' outer-loop states now (unit id -> core.DriveState). `lagged`:
    the units' effective outputs now (model.step). `fans`: unit id -> fan mode now; with the outer loop, the
    simulation picks fan speeds as it does (core.fan_for)."""
    fixed = fixed or {}
    drive = settings.drive
    n = max(1, int(round(ps.horizon_h * 60 / ps.step_min)))
    dt_h = ps.step_min / 60
    times = [now + timedelta(minutes=ps.step_min * (k + 1)) for k in range(n)]
    outdoor, sun = _pad(outdoor_seq, n), [hm.as_sun(s) for s in _pad(sun_seq, n)]
    scored = [r for r in rooms if r.unit is not None and r.id in m.rooms and temps.get(r.id) is not None
              and r.id not in ignore]
    # Targets and weights over the horizon don't depend on the plan: work them out once.
    tw = {r.id: [(core.target_temp(r, t), core.room_priority(r, t)) for t in times] for r in scored}
    end_weight = [2.0 if (times[-1] - t).total_seconds() < ps.terminal_h * 3600 else 1.0 for t in times]
    known_units = [u for u in units if u.id in m.units and u.id not in fixed]

    b1 = min(n, max(1, int(round(ps.first_block_min / ps.step_min))))
    blocks = [(0, b1)] + ([(b1, n)] if b1 < n else [])

    def zone_target(mine, k) -> float:
        wsum = sum(tw[r.id][k][1] for r in mine)
        return sum(tw[r.id][k][0] * tw[r.id][k][1] for r in mine) / wsum - settings.aim_offset

    def base_setpoint(u, a, b) -> float:
        """Priority-weighted target of the unit's rooms for the block of steps a..b, as a setpoint on the unit's own
        reading (or, with the outer loop, as a target on the rooms' scale: the block's mean, so a schedule change
        inside a long block is within the search's reach)."""
        mine = [r for r in scored if r.unit == u.id]
        if not mine:
            return float(round(setpoints_now.get(u.id) or 68.0))
        if drive:
            zone = sum(zone_target(mine, k) for k in range(a, b)) / max(1, b - a)
            return round(zone * 2) / 2
        return core.round_cool(zone_target(mine, a) + m.units[u.id].sensor_offset)

    zone_rooms = {u.id: [r for r in scored if r.unit == u.id] for u in known_units}
    kp, cap = {}, {}
    for u in known_units:
        eff = [m.rooms[r.id].heat.get(u.id, 0.0) for r in zone_rooms[u.id]]
        e = hm.rule_effect(sum(eff) / len(eff) if eff else None, m.units[u.id])
        kp[u.id] = core.drive_gain(e, settings)
        cap[u.id] = core.max_push_for(u.id, e, settings)

    fan_units = {x.strip() for x in settings.fan_units.split(",")}

    def controller(mode, targets, states, clock, first, fan_now):
        """model.rollout's control hook: the outer loop on the simulated rooms."""
        def control(temps, rates, dt_h):
            clock[0] += timedelta(hours=dt_h)
            out = {}
            for u in known_units:
                rs = [r for r in zone_rooms[u.id] if temps.get(r.id) is not None]
                if not rs or targets.get(u.id) is None:
                    out[u.id] = targets.get(u.id)  # no room to steer by: the target is the setpoint, as without the loop
                    continue
                k = min(n - 1, int((clock[0] - now).total_seconds() // (ps.step_min * 60)))
                w = [tw[r.id][k][1] for r in rs]
                room = sum(temps[r.id] * x for r, x in zip(rs, w)) / sum(w)
                rate = None if rates is None else sum(rates.get(r.id, 0.0) * x for r, x in zip(rs, w)) / sum(w)
                rd = hm.unit_reading(m, u.id, temps)
                off = 0.0 if rd is None else rd - room
                lag = m.units[u.id].lag_h * 60
                push, states[u.id] = core.drive_law(states.get(u.id), mode, room, targets[u.id], rate, kp[u.id],
                                                    clock[0], settings, lag if lag > 0 else None, cap[u.id])
                sgn = 1.0 if mode == "heat" else -1.0
                out[u.id] = max(u.min_setpoint, min(u.max_setpoint, room + off + sgn * push))
                if u.id in fan_units:
                    f = core.fan_for(push, states[u.id].miss, states[u.id].on, settings, cap[u.id])
                    if f is not None:
                        fan_now[u.id] = f
                if u.id not in first:
                    first[u.id] = core.to_grid(out[u.id])
            return out
        return control

    def clamp(u, v):
        return float(max(u.min_setpoint, min(u.max_setpoint, v)))

    cache = {}

    def evaluate(mode, sps):
        """sps: unit id -> (setpoint per block)."""
        key = (mode, tuple(sorted(sps.items())))
        if key in cache:
            return cache[key]
        run = {uid: running.get(uid, False) and (uid in fixed or mode == house.mode) for uid in m.units}
        modes_by_unit = {uid: (fixed[uid][0] if uid in fixed else mode) for uid in m.units}
        traj, outs, cur = [], [], dict(temps)
        lag = dict(lagged) if lagged is not None else None
        mass = dict(masses) if masses is not None else {}
        states = {uid: st for uid, st in (drives or {}).items() if mode == house.mode} if drive else {}
        clock, first = [now], {}
        fan_now = dict(fans or {})
        for bi, (a, b) in enumerate(blocks):
            sp_b = {uid: v[bi] for uid, v in sps.items()}
            ctl = None
            if drive and mode in ("heat", "cool"):
                inner = controller(mode, sp_b, states, clock, first, fan_now)
                fixed_sp = {uid: f[1] for uid, f in fixed.items()}
                ctl = lambda tt, rr, hh, inner=inner, fixed_sp=fixed_sp: {**inner(tt, rr, hh), **fixed_sp}
            sp_b.update({uid: f[1] for uid, f in fixed.items()})
            tr, ou, run = hm.rollout(m, cur, modes_by_unit, sp_b, run, outdoor[a:b], sun[a:b], dt_h, ps.substeps,
                                     lag, ctl, fans=fan_now, masses=mass)
            traj += tr
            outs += ou
            cur = tr[-1]
        c = 0.0
        for k, tk in enumerate(traj):
            for r in scored:
                tgt, w = tw[r.id][k]
                t = tk[r.id]
                aim = ps.aim_weight * w * (t - (tgt - settings.aim_offset)) ** 2
                c += (band_cost(t, tgt, w, settings) + aim) * dt_h * end_weight[k]
            c += ps.energy_cost * sum(abs(y) for y in outs[k].values()) * dt_h
        for uid, v in sps.items():
            prev = setpoints_now.get(uid)
            if mode == house.mode and prev is not None:
                c += ps.change_cost * (v[0] - prev) ** 2
            for x, y in zip(v, v[1:]):
                c += ps.change_cost * (y - x) ** 2
        cache[key] = (c, traj, first)
        return cache[key]

    modes = []
    forced = next((f[0] for f in fixed.values() if f[0] in ("heat", "cool")), None)
    if forced:
        modes = [forced]  # a unit on its own thermostat in heat/cool sets the mode for the shared compressor
    elif house.mode in ("heat", "cool"):
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
            c, traj, first = evaluate(mode, sps)
        else:
            base = {u.id: tuple(clamp(u, base_setpoint(u, a, b)) for a, b in blocks) for u in known_units}
            sps = dict(base)
            c, traj, first = evaluate(mode, sps)
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
                            cc, tt, ff = evaluate(mode, cand)
                            if cc < c - 1e-9:
                                c, traj, sps, first, improved = cc, tt, cand, ff, True
                if not improved:
                    break
            if house.mode in ("heat", "cool") and mode != house.mode:
                c += ps.switch_cost
        by_mode[mode] = {"cost": round(c, 3), "setpoints": {uid: v[0] for uid, v in sps.items()},
                         "later": {uid: v[-1] for uid, v in sps.items()}, "first": dict(first)}
        if best is None or c < best[1]:
            best = (mode, c, sps, traj)

    mode, c, sps, traj = best
    comfort = 0.0
    for k, tk in enumerate(traj):
        for r in scored:
            tgt, w = tw[r.id][k]
            comfort += band_cost(tk[r.id], tgt, w, settings)
    comfort /= max(1, len(traj))  # per step = per hour, as an average rate
    others = ", ".join(f"{md} {v['cost']:.2f}" for md, v in by_mode.items() if md != mode)
    return Plan(mode, by_mode[mode]["setpoints"], c, by_mode, {r.id: [round(t[r.id], 2) for t in traj] for r in scored},
                f"{mode}: cost {c:.2f}" + (f" (vs {others})" if others else ""), by_mode[mode]["later"],
                round(comfort, 3), by_mode[mode]["first"])
