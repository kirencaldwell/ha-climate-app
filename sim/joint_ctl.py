"""The joint run/fan planner (joint.py) as a controller on the simulated house: no outer loop on the setpoint, a plan
every 15 minutes and an immediate replan when the readings leave the prediction."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_components", "room_climate"))
import coupled  # noqa: E402
import core  # noqa: E402
import joint  # noqa: E402


class JointController:
    def __init__(self, rooms, units, model, settings=core.Settings(), js=None, observer=None, every_min=15,
                 trigger_f=0.7, min_gap_min=3, name="joint"):
        self.rooms, self.units, self.m, self.s = rooms, units, model, settings
        self.planner = joint.JointPlanner(js)
        self.observer, self.every, self.trigger, self.min_gap, self.name = observer, every_min, trigger_f, min_gap_min, name
        self.house = core.HouseState()
        self.next_at = self.last_plan_at = self.plan = self.plan_at = None
        self.deviant = 0
        self.plans = 0
        self.evals = 0

    def _observe(self, sim, now):
        if self.observer is None:
            return
        temps = {r.id: sim.reading(r.id) for r in self.rooms if r.id in sim.temps and r.id in self.m.rooms}
        actual = {u.id: {"mode": sim.mode[u.id], "setpoint": sim.sp[u.id], "running": sim.action(u.id) in ("heating", "cooling"),
                         "fan": sim.fan[u.id]} for u in self.units}
        self.observer.update(self.m, now, temps, actual, sim.weather.outdoor(now), sim.weather.sun(now), dict(sim.effect),
                             None, {"h": sim.comp_min / 60})

    def _off_plan(self, sim, now):
        return self.plan is not None and self.plan_at is not None and now < self.plan_at + timedelta(minutes=self.min_gap)

    def _deviation(self, sim, now) -> float:
        if self.plan is None:
            return 0.0
        k = (now - self.plan_at).total_seconds() / 60 / self.plan.step_min - 1  # index of the step at or before now
        worst = 0.0
        for rid, traj in self.plan.predicted.items():
            if rid not in sim.temps or not traj:
                continue
            if k < 0:
                continue
            i = min(len(traj) - 1, int(k))
            j = min(len(traj) - 1, i + 1)
            f = max(0.0, min(1.0, k - i))
            pred = traj[i] + (traj[j] - traj[i]) * f
            worst = max(worst, abs(sim.reading(rid) - pred))
        return worst

    def tick(self, sim, now: datetime):
        self._observe(sim, now)
        replan = self.next_at is None or now >= self.next_at
        if not replan and self.plan is not None and not self._off_plan(sim, now):
            self.deviant = self.deviant + 1 if self._deviation(sim, now) > self.trigger else 0
            replan = self.deviant >= 2
        if not replan:
            return self.house.mode
        self.deviant = 0
        self.next_at = now + timedelta(minutes=self.every)
        fw = getattr(sim.weather, "w", sim.weather)  # the forecast: without the plant's own weather error
        temps = {r.id: sim.reading(r.id) for r in self.rooms if r.id in sim.temps and r.id in self.m.rooms}
        running = {u.id: sim.action(u.id) in ("heating", "cooling") for u in self.units}
        corr = self.observer.corrections() if self.observer else {}
        p = self.planner.plan(self.m, self.rooms, self.units, temps, now, self.house, running, fw.outdoor, fw.sun, self.s,
                              lagged=dict(sim.effect), fans=dict(sim.fan), comp={"h": sim.comp_min / 60},
                              unit_temps={u.id: sim.unit_temp(u.id) for u in self.units}, **corr)
        self.plan, self.plan_at = p, now
        self.plans += 1
        self.evals += p.evals
        setpoints, fans = p.setpoints, p.fans
        if self.house.mode not in ("heat", "cool"):
            want = p.mode if p.mode != "off" else "heat"
            benefit = 0.0 if p.mode == "off" else p.by_mode["off"]["cost"] - p.cost
            self.house, _ = core.choose_mode(benefit if want == "heat" else 0.0, benefit if want == "cool" else 0.0,
                                             self.house, now, self.s)
            if self.house.mode not in ("heat", "cool"):
                return self.house.mode
            setpoints, fans = p.by_mode[self.house.mode]["setpoints"], {}
        elif p.mode != self.house.mode:
            self.house = core.HouseState(p.mode, now)
        for u in self.units:
            sim.command(u.id, self.house.mode, setpoints.get(u.id))
            if fans.get(u.id):
                sim.set_fan(u.id, fans[u.id])
        return self.house.mode
