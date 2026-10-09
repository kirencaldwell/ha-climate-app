"""Small RC thermal simulator for exercising the controller without hardware."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate")
sys.path.insert(0, _root)
import core  # noqa: E402
import predict  # noqa: E402


@dataclass
class SimRoom:
    temp: float
    ua: float = 0.002      # loss to outdoors, 1/min (~8 h time constant)
    cap_gain: float = 1.0  # scales the unit's effect on this room
    sun_gain: float = 0.0  # F/min at full sun (windows given by the core.Room)


class SimHouse:
    """Rooms sharing a unit split its output evenly. Unit sensor reads +`bias` F off the room average.

    With `predictive=True` each room learns a thermal model online from its own sensor history and the
    controller gets idle-drift predictions (using the true outdoor curve as a perfect forecast)."""

    def __init__(self, rooms, units, outdoor, start, unit_bias=3.0, shared_mode=False,
                 power=0.35, predictive=False, site=None, forecast_error=0.0):
        self.rooms: dict[str, core.Room] = {r.id: r for r in rooms}
        self.units: dict[str, core.Unit] = {u.id: u for u in units}
        self.sim = {r.id: SimRoom(temp=68.0) for r in rooms}
        self.state = {u.id: core.UnitState() for u in units}
        self.cmd = {u.id: core.Command("off", None) for u in units}
        self.running = {u.id: 0 for u in units}  # +1 heating, -1 cooling, 0 idle (what really happened last minute)
        self.outdoor = outdoor  # callable(datetime) -> F
        self.now = start
        self.bias = unit_bias
        self.power = power
        self.settings = core.Settings()
        self.log = []
        self.shared_mode = shared_mode
        self.sys = core.SystemState()
        self.predictive = predictive
        self.site = site or predict.Site()
        self.forecast_error = forecast_error  # constant F added to the forecast the controller sees
        self.models = {r.id: predict.RoomModel() for r in rooms}

    def unit_temp(self, uid):
        rs = [self.sim[r.id].temp for r in self.rooms.values() if r.unit == uid]
        return sum(rs) / len(rs) + self.bias

    def sun(self, rid):
        return predict.sun_factor(self.rooms[rid].windows, self.site, self.now)

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
                self.running[uid] = int(power)
                for r in rooms:
                    s = self.sim[r.id]
                    s.temp += (s.ua * (out - s.temp) + power * self.power / len(rooms) * s.cap_gain
                               + s.sun_gain * self.sun(r.id))
            if self.predictive:
                for rid, room in self.rooms.items():
                    self.models[rid].update(self.now, self.sim[rid].temp, out, self.running[room.unit], self.sun(rid))
            self.control()

    def predictions(self):
        lead = self.settings.precompensate_min
        fc = lambda t: self.outdoor(t) + self.forecast_error  # noqa: E731
        preds = {}
        for rid, room in self.rooms.items():
            m = self.models[rid]
            if m.ready:
                preds[rid] = predict.predict_temp(m, self.sim[rid].temp, self.now, lead, fc, room.windows, self.site)
        return preds

    def control(self):
        preds = self.predictions() if self.predictive else None
        if self.shared_mode:
            readings = {r.id: (self.sim[r.id].temp, self.now) for r in self.rooms.values()}
            temps = {u: self.unit_temp(u) for u in self.units}
            cmds, self.sys = core.decide_system(list(self.units.values()), list(self.rooms.values()),
                                                readings, temps, self.sys, self.now, self.settings, preds)
            for uid, c in cmds.items():
                self.cmd[uid] = c
                self.log.append((self.now, uid, c.mode))
            return
        for uid, unit in self.units.items():
            rooms = [r for r in self.rooms.values() if r.unit == uid]
            readings = {r.id: (self.sim[r.id].temp, self.now) for r in rooms}
            cmd = core.decide(unit, rooms, readings, self.unit_temp(uid), self.state[uid], self.now, self.settings)
            self.state[uid] = core.apply(self.state[uid], cmd, self.now)
            if cmd.setpoint is not None or cmd.mode != self.cmd[uid].mode:
                self.cmd[uid] = cmd
            self.log.append((self.now, uid, cmd.mode))
