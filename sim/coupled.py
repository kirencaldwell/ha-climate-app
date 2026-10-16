"""A coupled multi-room house for developing the model-based controller offline.

The "true" house is a model.HouseModel with known parameters (first floor coupled up the stairs to the bedrooms,
sun, outdoor loss), integrated minute by minute. Sensors are quantized like the real ones (H5100s in 0.18 F steps,
unit sensors in 0.5 C), units store setpoints in 0.5 C steps and run their own inverter loop, and the weather has
day-to-day variety and a real sun path for the house's latitude.

Controllers plug in with tick(sim, now): HeuristicController is today's controller (core.control_step), mirroring
the HA integration's tick; MPCController plans with optimizer.plan. run() scores them on the *true* temperatures.
Every minute can also be logged in the same shape as /config/room_climate_data rows, so tools/fit_model.py fits
simulated and real data the same way.
"""
from __future__ import annotations

import math
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
import core  # noqa: E402
import model as hm  # noqa: E402
import optimizer  # noqa: E402

LAT, LON = 47.68, -122.38  # the house (from its NWS weather entity)


def sun_elevation(t: datetime, lat=LAT, lon=LON, utc_offset_h=-7.0) -> float:
    doy = t.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    solar_hour = t.hour + t.minute / 60 - utc_offset_h + lon / 15  # local solar time (ignores the equation of time)
    ha = math.radians(15 * (solar_hour - 12))
    la = math.radians(lat)
    s = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


class Weather:
    """Outdoor temperature (min ~4am, max ~3pm), cloud cover per day, sun from the real sun path."""

    def __init__(self, start: datetime, days: int, mean=55.0, swing=8.0, seed=0):
        rng = random.Random(seed)
        self.start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        self.days = [(mean + rng.uniform(-5, 5), swing * rng.uniform(0.6, 1.3), rng.choice([0, 10, 30, 60, 90, 100]))
                     for _ in range(days + 3)]

    def _day(self, t):
        return self.days[min(len(self.days) - 1, max(0, (t - self.start).days))]

    def outdoor(self, t: datetime) -> float:
        mean, swing, _ = self._day(t)
        h = t.hour + t.minute / 60
        return mean + swing * math.sin(2 * math.pi * (h - 9.5) / 24)

    def cloud(self, t: datetime) -> float:
        return float(self._day(t)[2])

    def sun(self, t: datetime) -> float:
        return hm.solar_index(sun_elevation(t), self.cloud(t))


def the_house() -> hm.HouseModel:
    """A plausible stand-in for the real house: first floor (air handler) open plan, bedrooms 1-2 off it, two
    minisplit bedrooms upstairs that the first floor warms up the stairs (stack effect: stronger up than down)."""
    R = hm.RoomModel
    first = ["kitchen", "living_room", "dining_area"]
    rooms = {
        "kitchen": R(loss=0.035, solar=0.8, bias=0.5, heat={"air_handler": 5.0}, cool={"air_handler": 4.2},
                     coupling={"living_room": 0.8, "dining_area": 0.8}),
        "living_room": R(loss=0.035, solar=1.2, bias=0.3, heat={"air_handler": 5.0}, cool={"air_handler": 4.2},
                         coupling={"kitchen": 0.8, "dining_area": 0.8, "bedroom_1": 0.15, "bedroom_2": 0.15,
                                   "primary_bedroom": 0.04, "bedroom_3": 0.04}),
        "dining_area": R(loss=0.035, solar=0.6, bias=0.1, heat={"air_handler": 4.5}, cool={"air_handler": 3.8},
                         coupling={"kitchen": 0.8, "living_room": 0.8}),
        "bedroom_1": R(loss=0.045, solar=0.8, bias=0.1, heat={"air_handler": 4.0}, cool={"air_handler": 3.4},
                       coupling={"living_room": 0.15, "bedroom_2": 0.1}),
        "bedroom_2": R(loss=0.045, solar=0.8, bias=0.1, heat={"air_handler": 4.0}, cool={"air_handler": 3.4},
                       coupling={"living_room": 0.15, "bedroom_1": 0.1}),
        "primary_bedroom": R(loss=0.055, solar=1.6, bias=0.15, heat={"minisplit_1": 7.0}, cool={"minisplit_1": 6.0},
                             coupling={"living_room": 0.12, "bedroom_3": 0.15}),
        "bedroom_3": R(loss=0.06, solar=2.5, bias=0.1, heat={"minisplit_2": 7.0}, cool={"minisplit_2": 6.0},
                       coupling={"living_room": 0.12, "primary_bedroom": 0.15}),
    }
    units = {"air_handler": hm.UnitModel(sensor_rooms=["kitchen"]),
             "minisplit_1": hm.UnitModel(sensor_rooms=["primary_bedroom"]),
             "minisplit_2": hm.UnitModel(sensor_rooms=["bedroom_3"])}
    return hm.HouseModel(rooms, units, {"source": "sim truth", "first_floor": first})


def _q(v: float, step: float) -> float:
    return round(round(v / step) * step, 2)


class CoupledSim:
    def __init__(self, truth: hm.HouseModel, start: datetime, weather: Weather, temps=None, seed=0,
                 room_step=0.18, unit_step=0.9, noise=0.03):
        self.m, self.now, self.weather = truth, start, weather
        self.temps = dict(temps or {r: 66.0 for r in truth.rooms})
        self.mode = {u: "off" for u in truth.units}
        self.sp = {u: None for u in truth.units}
        self.running = {u: False for u in truth.units}
        self.output = {u: 0.0 for u in truth.units}
        self.room_step, self.unit_step, self.noise = room_step, unit_step, noise
        self.rng = random.Random(seed)
        self.starts = 0
        self.energy = 0.0  # full-output unit-hours
        self.commands = 0

    # --- what the controller can do and see ---
    def command(self, uid: str, mode: str, setpoint_f=None):
        if mode != self.mode[uid]:
            self.mode[uid] = mode
            self.commands += 1
        if setpoint_f is not None:
            stored = core.stored_setpoint_c(setpoint_f) * 1.8 + 32  # 0.5 C steps, like the real units
            if self.sp[uid] is None or abs(stored - self.sp[uid]) > 0.01:
                self.sp[uid] = stored
                self.commands += 1

    def reading(self, rid: str) -> float:
        return _q(self.temps[rid] + self.rng.gauss(0, self.noise), self.room_step)

    def unit_temp(self, uid: str):
        r = hm.unit_reading(self.m, uid, self.temps)
        return None if r is None else _q(r, self.unit_step)

    def action(self, uid: str) -> str:
        if self.mode[uid] not in ("heat", "cool"):
            return "off"
        y = self.output[uid]
        return "heating" if y > 0 else "cooling" if y < 0 else "idle"

    # --- physics ---
    def advance(self, minutes: int = 1):
        for _ in range(minutes):
            for uid, u in self.m.units.items():
                was = self.running[uid]
                y, self.running[uid] = hm.unit_output(u, self.mode[uid], self.sp[uid], hm.unit_reading(self.m, uid, self.temps), was)
                self.starts += (not was) and self.running[uid]
                self.output[uid] = y
                self.energy += abs(y) / 60
            d = hm.derivative(self.m, self.temps, self.weather.outdoor(self.now), self.weather.sun(self.now), self.output)
            for rid, v in d.items():
                self.temps[rid] += v / 60
            self.now += timedelta(minutes=1)

    def row(self, rooms) -> dict:
        """One minute in the shape of the HA long-term log."""
        t = self.now
        return {"t": t.isoformat(), "outdoor": round(self.weather.outdoor(t), 2),
                "weather": {"cloud": self.weather.cloud(t)}, "sun": {"elev": round(sun_elevation(t), 2)},
                "rooms": {r.id: {"temp": self.reading(r.id), "target": core.target_temp(r, t), "prio": core.room_priority(r, t),
                                 "unit": r.unit} for r in rooms if r.id in self.temps},
                "units": {u: {"state": self.mode[u], "sp": None if self.sp[u] is None else round(self.sp[u], 1),
                              "action": self.action(u), "temp": self.unit_temp(u)} for u in self.m.units}}


class HeuristicController:
    """Today's controller, as the HA integration runs it each tick."""

    name = "heuristic"

    def __init__(self, rooms, units, settings=core.Settings()):
        self.rooms, self.units, self.s = rooms, units, settings
        self.house, self.states, self.hist, self.sent, self.prev_busy, self.last_good = core.HouseState(), {}, {}, {}, set(), {}

    def tick(self, sim: CoupledSim, now: datetime):
        readings, predicted = {}, {}
        for r in self.rooms:
            if r.id not in sim.temps:
                continue
            got, self.last_good[r.id] = core.hold_reading((sim.reading(r.id), now), self.last_good.get(r.id), now, self.s)
            readings[r.id] = got
            h = self.hist.setdefault(r.id, [])
            h.append((now, got[0]))
            del h[:max(0, next((k for k, (t, _) in enumerate(h) if core.elapsed(now, t) <= self.s.history_s), len(h)))]
            predicted[r.id] = core.predict(h, now, self.s)
        busy = {u.id for u in self.units if sim.action(u.id) in ("heating", "cooling")}
        for u in self.units:
            if (u.id in busy) != (u.id in self.prev_busy):
                for r in self.rooms:
                    if r.unit == u.id and self.hist.get(r.id):
                        self.hist[r.id] = self.hist[r.id][-1:]
                        predicted[r.id] = None
        self.prev_busy = busy
        cmds, self.house, info = core.control_step(self.units, self.rooms, readings, predicted,
                                                   {u.id: sim.unit_temp(u.id) for u in self.units}, self.states, self.house,
                                                   now, self.s, {}, busy, sim.weather.outdoor(now))
        for u in self.units:
            cmd, changed = core.stabilize(cmds[u.id], self.sent.get(u.id), now, self.s)
            if changed and cmd.setpoint is not None:
                self.sent[u.id] = (cmd, now)
            state = self.states.setdefault(u.id, core.UnitState())
            if not (cmd.setpoint is None and cmd.mode == state.mode):  # same rule as Manager._send
                sim.command(u.id, cmd.mode, cmd.setpoint)
            self.states[u.id] = core.apply(state, cmd, now)
            if core.regime_changed(state, self.states[u.id]):
                for r in self.rooms:
                    if r.unit == u.id and self.hist.get(r.id):
                        self.hist[r.id] = self.hist[r.id][-1:]
        return self.house.mode


class MPCController:
    """Plans every `every_min` with optimizer.plan on `model` (the truth, or one fitted from logs). Setpoints still go
    through core.stabilize, and leaving "off" still needs core's confirmation, so the safety rules are unchanged."""

    name = "mpc"

    def __init__(self, rooms, units, model: hm.HouseModel, settings=core.Settings(), ps=optimizer.PlanSettings(),
                 every_min=10):
        self.rooms, self.units, self.m, self.s, self.ps, self.every = rooms, units, model, settings, ps, every_min
        self.house, self.sent, self.next_at, self.last_plan = core.HouseState(), {}, None, None

    def tick(self, sim: CoupledSim, now: datetime):
        if self.next_at is not None and now < self.next_at:
            return self.house.mode
        self.next_at = now + timedelta(minutes=self.every)
        temps = {r.id: sim.reading(r.id) for r in self.rooms if r.id in sim.temps and r.id in self.m.rooms}
        n = int(round(self.ps.horizon_h * 60 / self.ps.step_min))
        ts = [now + timedelta(minutes=self.ps.step_min * k) for k in range(n)]
        running = {u.id: sim.action(u.id) in ("heating", "cooling") for u in self.units}
        now_sp = {u.id: (self.sent[u.id][0].setpoint if u.id in self.sent else None) for u in self.units}
        p = optimizer.plan(self.m, self.rooms, self.units, temps, now, self.house, now_sp, running,
                           [sim.weather.outdoor(t) for t in ts], [sim.weather.sun(t) for t in ts], self.s, self.ps)
        self.last_plan = p
        setpoints = p.setpoints
        if self.house.mode not in ("heat", "cool"):
            # Starting from off: the "demand" is how much better the best mode is than staying off, and it goes
            # through the same start threshold and 10-minute confirmation as today's controller.
            want = p.mode if p.mode != "off" else "heat"
            benefit = 0.0 if p.mode == "off" else p.by_mode["off"]["cost"] - p.cost
            self.house, _ = core.choose_mode(benefit if want == "heat" else 0.0, benefit if want == "cool" else 0.0,
                                             self.house, now, self.s)
            if self.house.mode not in ("heat", "cool"):
                return self.house.mode
            setpoints = p.by_mode[self.house.mode]["setpoints"]
        elif p.mode != self.house.mode:
            self.house = core.HouseState(p.mode, now)
        for u in self.units:
            sp = setpoints.get(u.id)
            cmd = core.Command(self.house.mode, sp, p.reason, True, sp)
            cmd, changed = core.stabilize(cmd, self.sent.get(u.id), now, self.s)
            if changed and cmd.setpoint is not None:
                self.sent[u.id] = (cmd, now)
            sim.command(u.id, cmd.mode, cmd.setpoint)
        return self.house.mode


def run(truth: hm.HouseModel, rooms, units, controller, start: datetime, days: float, weather: Weather,
        settings=core.Settings(), temps=None, log=None, seed=0) -> dict:
    """Run `controller` on the true house; score on the true temperatures. `log`: list to append minute rows to."""
    sim = CoupledSim(truth, start, weather, temps=temps, seed=seed)
    scored = [r for r in rooms if r.unit is not None and r.id in truth.rooms]
    disc, out_min, switches, last_mode = {r.id: 0.0 for r in scored}, {r.id: 0 for r in scored}, 0, None
    worst = {r.id: 0.0 for r in scored}
    for _ in range(int(days * 1440)):
        now = sim.now
        mode = controller.tick(sim, now)
        if mode in ("heat", "cool") and last_mode in ("heat", "cool") and mode != last_mode:
            switches += 1
        if mode in ("heat", "cool"):
            last_mode = mode
        if log is not None:
            log.append(sim.row(rooms))
        sim.advance(1)
        for r in scored:
            t, tgt, w = sim.temps[r.id], core.target_temp(r, now), core.room_priority(r, now)
            disc[r.id] += optimizer.band_cost(t, tgt, w, settings) / 60
            dev = max((tgt - settings.band_below) - t, t - (tgt + settings.band_above), 0.0)
            out_min[r.id] += dev > 0
            worst[r.id] = max(worst[r.id], dev)
    return {"controller": controller.name, "discomfort": round(sum(disc.values()), 2),
            "by_room": {k: round(v, 2) for k, v in disc.items()},
            "hours_outside_band": {k: round(v / 60, 1) for k, v in out_min.items()},
            "worst_F": {k: round(v, 2) for k, v in worst.items()},
            "mode_switches": switches, "compressor_starts": sim.starts, "commands": sim.commands,
            "energy_unit_h": round(sim.energy, 1)}


def rooms_units_from_config(cfg: dict, truth: hm.HouseModel):
    """core.Room/Unit objects from a Room Climate config (the HA store or room_climate/get), keeping the real
    schedules and priorities, for the rooms the simulated house has. Every simulated room gets its own sensor."""
    rooms = []
    for d in cfg["rooms"]:
        if d["id"] not in truth.rooms or not d.get("unit"):
            continue
        rooms.append(core.Room(
            id=d["id"], name=d.get("name", d["id"]), unit=d["unit"], sensor=f"sim.{d['id']}",
            priority=int(d.get("priority", 3)), default_temp=float(d.get("default_temp", 68)),
            schedule=[core.ScheduleEntry(list(e["days"]), e["start"], float(e["temp"])) for e in d.get("schedule", [])],
            priority_schedule=[core.PriorityEntry(list(e["days"]), e["start"], int(e["priority"]))
                               for e in d.get("priority_schedule", [])],
            floor=int(d.get("floor", 1))))
    units = [core.Unit(u["id"], f"climate.{u['id']}", float(u.get("min_setpoint", 61)), float(u.get("max_setpoint", 88)))
             for u in cfg["units"] if u["id"] in truth.units]
    return rooms, units


def settings_from_config(cfg: dict) -> core.Settings:
    return core.settings_from(cfg.get("settings_effective") or cfg.get("settings"))
