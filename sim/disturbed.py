"""A simulated house that differs from the planner's model the way the real one does, for testing the planner's
feedback (observer.py, the mode hysteresis) and not just its plans.

The planner is given the nominal house (coupled.the_house()), as a fitted model would be; the *plant* it runs on is that
house perturbed by a `Disturbance`: a gain the model has wrong, a leak it under-predicts, a disturbance it never sees.
`draw` samples disturbances with magnitudes taken from what the real logs showed (Oct 14-22), so a Monte Carlo over
draws says how a controller copes with model error of about the size we actually have:

  kitchen_gain   the kitchen warms at ~1/3 the rate of the dining and living rooms on the same air handler
                 (1.1 vs 2.9 F/h, 4 days of logs) while the model gives it a neighbour-like gain: truth/model ~0.25-1
  leak_mult      Bedroom 3 rose 3.8 F in 20 min when the compressor ran at 2.2 kW; the fitted leak predicts ~1.8: 1-3x
  leak_boost     the leak is biggest in the first minutes of a heat run: up to +150% decaying over 10-30 min
  solar_err      the east/south/overhead sun split flips between fits: +-30% on every sun term
  unit_gain      each unit's heating/cooling effect, +-15%
  loss_err       envelope loss, +-20%
  bias           unmodelled steady gains or losses per room, +-0.25 F/h
  b1_evening     Bedroom 1 climbs to ~70 F around 17:00 with its unit idle (a person, a computer): up to 2 F/h, 15:30-19:00
  ou_sigma       occupancy, doors, cooking: a wandering disturbance of 0.1-0.5 F/h (1 h memory) in every room
  outdoor_bias   the plant's weather differs from the forecast the planner uses, +-2 F
"""
from __future__ import annotations

import math
import os
import random
import sys
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_components", "room_climate"))
import coupled  # noqa: E402
import model as hm  # noqa: E402


@dataclass
class Disturbance:
    kitchen_gain: float = 1.0
    leak_mult: float = 1.0
    leak_boost: float = 0.0
    leak_tau_min: float = 20.0
    solar_err: float = 1.0
    unit_gain: dict = field(default_factory=dict)
    loss_err: float = 1.0
    bias: dict = field(default_factory=dict)
    b1_evening: float = 0.0
    ou_sigma: float = 0.0
    ou_tau_min: float = 60.0
    outdoor_bias: float = 0.0


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def draw(rng: random.Random, scale: float = 1.0) -> Disturbance:
    """One sample. `scale` stretches every deviation from nothing-wrong (0: a perfect model, 2: twice as wrong)."""
    def dev(x):  # x's deviation from 1, scaled
        return 1.0 + (x - 1.0) * scale
    units = ("air_handler", "minisplit_1", "minisplit_2")
    rooms = ("kitchen", "living_room", "dining_area", "bedroom_1", "bedroom_2", "primary_bedroom", "bedroom_3")
    return Disturbance(
        kitchen_gain=_clip(dev(math.exp(rng.gauss(math.log(0.45), 0.35))), 0.15, 1.2),
        leak_mult=_clip(dev(rng.uniform(1.0, 3.0)), 0.5, 5.0),
        leak_boost=rng.uniform(0.0, 1.5) * scale,
        leak_tau_min=rng.uniform(10, 30),
        solar_err=_clip(dev(rng.gauss(1.0, 0.3)), 0.3, 2.0),
        unit_gain={u: _clip(dev(rng.gauss(1.0, 0.15)), 0.5, 1.5) for u in units},
        loss_err=_clip(dev(rng.gauss(1.0, 0.2)), 0.5, 1.6),
        bias={r: rng.gauss(0.0, 0.25) * scale for r in rooms},
        b1_evening=rng.uniform(0.0, 2.0) * scale,
        ou_sigma=rng.uniform(0.1, 0.5) * scale,
        outdoor_bias=rng.gauss(0.0, 2.0) * scale)


def perturbed_truth(base: hm.HouseModel, d: Disturbance) -> hm.HouseModel:
    t = hm.HouseModel.from_dict(base.to_dict())
    for rid, r in t.rooms.items():
        r.loss *= d.loss_err
        r.solar *= d.solar_err
        r.sun_dir = {k: v * d.solar_err for k, v in r.sun_dir.items()}
        r.bias += d.bias.get(rid, 0.0)
        kg = d.kitchen_gain if rid == "kitchen" else 1.0
        r.heat = {u: v * kg * d.unit_gain.get(u, 1.0) for u, v in r.heat.items()}
        r.cool = {u: v * kg * d.unit_gain.get(u, 1.0) for u, v in r.cool.items()}
        r.leak = {u: v * d.leak_mult for u, v in r.leak.items()}
    return t


class BiasedWeather:
    """The plant's weather: the planner's forecast plus a constant error."""

    def __init__(self, weather: coupled.Weather, bias: float):
        self.w, self.bias = weather, bias

    def outdoor(self, t):
        return self.w.outdoor(t) + self.bias

    def cloud(self, t):
        return self.w.cloud(t)

    def sun(self, t):
        return self.w.sun(t)


class DisturbedSim(coupled.CoupledSim):
    """CoupledSim plus what the model doesn't know: a leak that spikes as the compressor starts, Bedroom 1's evening
    gain, and a wandering disturbance in every room."""

    def __init__(self, truth, start, weather, d: Disturbance, seed=0, **kw):
        super().__init__(truth, start, BiasedWeather(weather, d.outdoor_bias), seed=seed, **kw)
        self.d = d
        self.rng2 = random.Random(seed * 7919 + 13)
        self.ou = {r: 0.0 for r in truth.rooms}
        self.comp_min = 0  # minutes the compressor has been running (0 when every unit is idle)
        self.day_factor = {}

    def _b1(self, now: datetime) -> float:
        h = now.hour + now.minute / 60
        if not (15.5 <= h < 19.0) or self.d.b1_evening <= 0:
            return 0.0
        key = now.date()
        if key not in self.day_factor:
            self.day_factor[key] = random.Random(zlib.crc32(str(key).encode())).uniform(0.5, 1.2)
        return self.d.b1_evening * self.day_factor[key]

    def advance(self, minutes: int = 1):
        d = self.d
        for _ in range(minutes):
            for uid, u in self.m.units.items():
                was = self.running[uid]
                y, self.running[uid] = hm.unit_output(u, self.mode[uid], self.sp[uid],
                                                      hm.unit_reading(self.m, uid, self.temps), was)
                y *= hm.fan_factor(u, self.fan[uid])
                self.starts += (not was) and self.running[uid]
                self.output[uid] = y
                self.effect[uid] = hm.lag(u, self.effect[uid], y, 1 / 60)
                self.energy += abs(y) / 60
            self.comp_min = self.comp_min + 1 if any(abs(y) > 0 for y in self.output.values()) else 0
            boost = 1.0 + d.leak_boost * math.exp(-self.comp_min / max(1.0, d.leak_tau_min)) if self.comp_min else 1.0
            leaks = {u: v * boost for u, v in hm.leak_inputs(self.mode, self.running, self.output, self.m.units).items()}
            rate = hm.derivative(self.m, self.temps, self.weather.outdoor(self.now), self.weather.sun(self.now),
                                 self.effect, leaks)
            for rid, v in rate.items():
                self.ou[rid] += (-self.ou[rid] / d.ou_tau_min
                                 + d.ou_sigma * math.sqrt(2.0 / d.ou_tau_min) * self.rng2.gauss(0, 1))
                extra = self.ou[rid] + (self._b1(self.now) if rid == "bedroom_1" else 0.0)
                self.temps[rid] += (v + extra) / 60
            self.now += timedelta(minutes=1)
