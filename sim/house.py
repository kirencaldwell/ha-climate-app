"""Tiny RC thermal simulator for exercising the controller without hardware."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))
import core  # noqa: E402


@dataclass
class SimRoom:
    temp: float
    ua: float = 0.002  # loss to outdoors, 1/min (~8h time constant)
    cap_gain: float = 1.0  # F/min of full-power effect at unit output 1.0


class SimHouse:
    """Rooms sharing a unit split its output evenly. Unit sensor reads +`bias` F off the room average."""

    def __init__(self, rooms, units, outdoor, start, unit_bias=3.0):
        self.rooms: dict[str, core.Room] = {r.id: r for r in rooms}
        self.units: dict[str, core.Unit] = {u.id: u for u in units}
        self.sim = {r.id: SimRoom(temp=68.0) for r in rooms}
        self.state = {u.id: core.UnitState() for u in units}
        self.cmd = {u.id: core.Command("off", None) for u in units}
        self.outdoor = outdoor  # callable(datetime) -> F
        self.now = start
        self.bias = unit_bias
        self.settings = core.Settings()
        self.house = core.HouseState()
        self.hist = {}
        self.log = []

    def unit_temp(self, uid):
        rs = [self.sim[r.id].temp for r in self.rooms.values() if r.unit == uid]
        return sum(rs) / len(rs) + self.bias

    def step(self, minutes=1):
        for _ in range(minutes):
            self.now += timedelta(minutes=1)
            out = self.outdoor(self.now)
            for uid, unit in self.units.items():
                rooms = [r for r in self.rooms.values() if r.unit == uid]
                power = 0.0
                c = self.cmd[uid]
                ut = self.unit_temp(uid)
                if c.mode == "heat" and ut < c.setpoint:
                    power = 1.0
                elif c.mode == "cool" and ut > c.setpoint:
                    power = -1.0
                for r in rooms:
                    s = self.sim[r.id]
                    s.temp += s.ua * (out - s.temp) + power * 0.35 / len(rooms) * s.cap_gain
            if self.now.minute % 1 == 0:
                self.control()

    def control(self):
        rooms = list(self.rooms.values())
        readings = {r.id: (self.sim[r.id].temp, self.now) for r in rooms}
        predicted = {}
        for r in rooms:
            h = self.hist.setdefault(r.id, [])
            h.append((self.now, self.sim[r.id].temp))
            del h[:max(0, len(h) - 40)]
            predicted[r.id] = core.predict(h, self.now, self.settings)
        cmds, self.house, _ = core.control_step(
            list(self.units.values()), rooms, readings, predicted, {u: self.unit_temp(u) for u in self.units},
            self.state, self.house, self.now, self.settings)
        for uid, cmd in cmds.items():
            old, self.state[uid] = self.state[uid], core.apply(self.state[uid], cmd, self.now)
            if core.regime_changed(old, self.state[uid]):
                for r in rooms:
                    if r.unit == uid and self.hist.get(r.id):
                        self.hist[r.id] = self.hist[r.id][-1:]
            if cmd.setpoint is not None or cmd.mode != self.cmd[uid].mode:
                self.cmd[uid] = cmd
            self.log.append((self.now, uid, cmd.mode))
