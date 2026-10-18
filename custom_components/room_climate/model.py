"""Structured (grey-box) thermal model of the house. Pure Python: runs on Home Assistant and offline.

Each room's temperature changes (F per hour) by
    loss * (outdoor - T)                      heat leaking to or from outdoors
  + sum coupling[j] * (T_j - T)                heat exchanged with other rooms; directed, so heat rising from the
                                               first floor to a bedroom can be stronger than the other way
  + solar * sun_h + sum sun_dir[k] * sun_k      sun gain: overall (sun height) plus sun on east/south/west walls,
                                               each 0..1 (see sun_vector), so a south room can warm at midday and a
                                               west one late in the day
  + bias                                       people, appliances, anything unmodelled
  + sum heat[u] * y_u   (y_u > 0, heating)     each unit's effect at its current output
  - sum cool[u] * |y_u| (y_u < 0, cooling)

Units are inverters with their own controller: we only choose mode and setpoint, and the unit sets its output from
the gap between setpoint and its own reading (the mean of `sensor_rooms` plus `sensor_offset`). That inner loop is
modelled too, so a plan can be simulated in terms of the setpoints we would actually send.

Parameters are fitted from the minute log by tools/fit_model.py and stored as JSON (to_dict / from_dict).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class RoomModel:
    loss: float = 0.04  # 1/h toward outdoors
    solar: float = 0.0  # F/h in full sun (overhead: roof, skylights, anything not facing one way)
    sun_dir: dict = field(default_factory=dict)  # "e"/"s"/"w" -> F/h with full sun straight onto a wall facing that way
    bias: float = 0.0  # F/h
    heat: dict = field(default_factory=dict)  # unit id -> F/h at full heating output
    cool: dict = field(default_factory=dict)  # unit id -> F/h of cooling at full cooling output (positive)
    coupling: dict = field(default_factory=dict)  # other room id -> 1/h toward its temperature


@dataclass
class UnitModel:
    sensor_rooms: list = field(default_factory=list)  # rooms whose mean temperature the unit regulates on
    sensor_offset: float = 0.0  # unit reading minus that mean (F)
    # Inverter: while running, output = clip(k0 + k1 * error, ymin, 1); starts when error > start, stops once it has
    # overshot by more than stop (it can't modulate below ymin). error = setpoint - reading in heat, the reverse in cool.
    k0: float = 0.3
    k1: float = 0.5
    ymin: float = 0.25
    start: float = 0.5
    stop: float = 1.0
    # Its effect on the rooms lags its output by this time constant (h): a coil and ducts stay warm (or cold) and keep
    # blowing a while after the compressor slows or stops, so a room goes on rising after its unit stops. Fitted.
    lag_h: float = 0.0


@dataclass
class HouseModel:
    rooms: dict = field(default_factory=dict)  # room id -> RoomModel
    units: dict = field(default_factory=dict)  # unit id -> UnitModel
    meta: dict = field(default_factory=dict)  # how/when it was fitted, fit quality

    def to_dict(self) -> dict:
        return {"rooms": {k: asdict(v) for k, v in self.rooms.items()},
                "units": {k: asdict(v) for k, v in self.units.items()}, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "HouseModel":
        known_r = set(RoomModel.__dataclass_fields__)
        known_u = set(UnitModel.__dataclass_fields__)
        return cls({k: RoomModel(**{a: b for a, b in v.items() if a in known_r}) for k, v in d.get("rooms", {}).items()},
                   {k: UnitModel(**{a: b for a, b in v.items() if a in known_u}) for k, v in d.get("units", {}).items()},
                   d.get("meta", {}))


def solar_index(elevation_deg, cloud_pct=None) -> float:
    """0..1 sun strength from the sun's elevation and cloud cover (clear sky = 1 at the zenith)."""
    if elevation_deg is None or elevation_deg <= 0:
        return 0.0
    clear = math.sin(math.radians(elevation_deg))
    cloud = 0.0 if cloud_pct is None else max(0.0, min(100.0, cloud_pct)) / 100.0
    return clear * (1.0 - 0.75 * cloud)


WALLS = {"e": 90.0, "s": 180.0, "w": 270.0}  # wall facing -> compass bearing


def sun_vector(elevation_deg, azimuth_deg, cloud_pct=None) -> dict:
    """Sun strengths 0..1: "h" overhead (solar_index) and, per wall facing (WALLS), how squarely the sun hits a
    vertical wall that way (cos of the angle between the beam and the wall's normal), faded in over the first 10
    degrees above the horizon (trees, neighbours) and dimmed by cloud like "h". No azimuth: overhead only."""
    h = solar_index(elevation_deg, cloud_pct)
    out = {"h": h, "e": 0.0, "s": 0.0, "w": 0.0}
    if azimuth_deg is None or elevation_deg is None or elevation_deg <= 0:
        return out
    cloud = 0.0 if cloud_pct is None else max(0.0, min(100.0, cloud_pct)) / 100.0
    beam = math.cos(math.radians(elevation_deg)) * min(1.0, elevation_deg / 10.0) * (1.0 - 0.75 * cloud)
    for k, bearing in WALLS.items():
        out[k] = round(max(0.0, beam * math.cos(math.radians(azimuth_deg - bearing))), 4)
    return out


def as_sun(sun) -> dict:
    """A sun input as a vector: a plain number (older callers) is the overhead strength only."""
    if isinstance(sun, dict):
        return sun
    return {"h": sun or 0.0}


def mean_sun(xs) -> dict:
    """Average of sun inputs (numbers or vectors)."""
    vs = [as_sun(x) for x in xs]
    if not vs:
        return {"h": 0.0}
    keys = set().union(*vs)
    return {k: sum(v.get(k, 0.0) for v in vs) / len(vs) for k in keys}


def _solar_hour_angle(t: datetime, lat: float, lon: float):
    u = t.astimezone(timezone.utc) if t.tzinfo else t
    doy = u.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    b = 2 * math.pi * (doy - 81) / 364
    eot_min = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)  # equation of time
    solar_hour = u.hour + u.minute / 60 + lon / 15 + eot_min / 60
    return math.radians(15 * (solar_hour - 12)), decl, math.radians(lat)


def sun_azimuth(t: datetime, lat: float, lon: float) -> float:
    """Sun compass bearing in degrees (0 north, 90 east, 180 south) at time `t`, within a degree or two."""
    ha, decl, la = _solar_hour_angle(t, lat, lon)
    az = math.degrees(math.atan2(-math.sin(ha), math.tan(decl) * math.cos(la) - math.sin(la) * math.cos(ha)))
    return az % 360.0


def sun_at(t: datetime, lat: float, lon: float, cloud_pct=None) -> dict:
    """The sun vector at time `t` for a place (sun_vector of its elevation and azimuth)."""
    return sun_vector(sun_elevation(t, lat, lon), sun_azimuth(t, lat, lon), cloud_pct)


def sun_elevation(t: datetime, lat: float, lon: float) -> float:
    """Sun elevation in degrees at time `t` (timezone-aware; naive = UTC) for latitude/longitude, within ~1 degree."""
    u = t.astimezone(timezone.utc) if t.tzinfo else t
    doy = u.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    b = 2 * math.pi * (doy - 81) / 364
    eot_min = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)  # equation of time
    solar_hour = u.hour + u.minute / 60 + lon / 15 + eot_min / 60
    ha = math.radians(15 * (solar_hour - 12))
    la = math.radians(lat)
    s = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def unit_reading(m: HouseModel, unit_id: str, temps: dict):
    u = m.units[unit_id]
    vals = [temps[r] for r in u.sensor_rooms if temps.get(r) is not None]
    return None if not vals else sum(vals) / len(vals) + u.sensor_offset


def unit_output(u: UnitModel, mode: str, setpoint, reading, running: bool, hold: bool = False):
    """(signed output: + heating, - cooling, 0 idle; still running?) for one unit. `hold`: whether it runs is known
    (replaying what a unit really did), so only how hard it runs is modelled."""
    if mode not in ("heat", "cool") or setpoint is None or reading is None:
        return 0.0, False
    err = (setpoint - reading) if mode == "heat" else (reading - setpoint)
    if hold:
        pass
    elif running and err < -u.stop:
        running = False
    elif not running and err > u.start:
        running = True
    if not running:
        return 0.0, False
    y = max(u.ymin, min(1.0, u.k0 + u.k1 * err))
    return (y if mode == "heat" else -y), True


def derivative(m: HouseModel, temps: dict, outdoor, sun: float, outputs: dict) -> dict:
    """dT/dt (F/h) for every modelled room with a temperature. `sun`: a sun vector (sun_vector) or overhead 0..1."""
    out = {}
    sv = as_sun(sun)
    for rid, r in m.rooms.items():
        t = temps.get(rid)
        if t is None:
            continue
        d = r.bias + r.solar * sv.get("h", 0.0)
        for k, c in r.sun_dir.items():
            d += c * sv.get(k, 0.0)
        if outdoor is not None:
            d += r.loss * (outdoor - t)
        for j, c in r.coupling.items():
            tj = temps.get(j)
            if tj is not None:
                d += c * (tj - t)
        for uid, y in outputs.items():
            if y > 0:
                d += r.heat.get(uid, 0.0) * y
            elif y < 0:
                d += r.cool.get(uid, 0.0) * y  # y is negative: cooling
        out[rid] = d
    return out


def lag(u: UnitModel, z: float, y: float, dt_h: float) -> float:
    """A unit's effective output after dt_h: its output y seen through the unit's lag (z: effective output before)."""
    if u.lag_h <= 0:
        return y
    return z + (y - z) * (1.0 - math.exp(-dt_h / u.lag_h))


def step(m: HouseModel, temps: dict, outdoor, sun: float, mode, setpoints: dict, running: dict, dt_h: float,
         lagged: dict = None, hold: bool = False):
    """Advance one step: (new temps, new running flags, outputs used). Units decide output from their reading at the
    start of the step (as a real inverter does, with a little lag). `mode` is one mode for every unit, or a dict of
    unit id -> mode (what the units are really set to, which may differ). `lagged` (unit id -> effective output) carries
    the units' lag from step to step and is updated in place; without it a unit's effect follows its output at once.
    `hold`: `running` is what the units really did, not simulated (see unit_output)."""
    outputs, run2 = {}, {}
    for uid, u in m.units.items():
        md = mode.get(uid, "off") if isinstance(mode, dict) else mode
        y, run2[uid] = unit_output(u, md, setpoints.get(uid), unit_reading(m, uid, temps), running.get(uid, False), hold)
        if lagged is not None:
            # the mean over the step of the lagged output, approximated by its end value halfway in
            z = lag(u, lagged.get(uid, y), y, dt_h)
            y = (lagged.get(uid, y) + z) / 2
            lagged[uid] = z
        outputs[uid] = y
    d = derivative(m, temps, outdoor, sun, outputs)
    new = dict(temps)
    for rid, v in d.items():
        new[rid] = temps[rid] + v * dt_h
    return new, run2, outputs


def rollout(m: HouseModel, temps: dict, mode, setpoints: dict, running: dict, outdoor_seq, sun_seq,
            dt_h: float, substeps: int = 1, lagged: dict = None, control=None):
    """Simulate len(outdoor_seq) steps of dt_h with setpoints held. Returns ([temps after each step], [outputs per step],
    final running flags). `lagged`: see step (updated in place). `control(temps, rates, dt_h)` -> setpoints, when given,
    sets them before every substep instead (an outer loop on the room temperatures; rates are F/h from the last
    substep, None at first)."""
    traj, outs = [], []
    run = dict(running)
    h = dt_h / substeps
    rates = None
    for k in range(len(outdoor_seq)):
        acc = {}
        for _ in range(substeps):
            sps = control(temps, rates, h) if control is not None else setpoints
            before = temps
            temps, run, y = step(m, temps, outdoor_seq[k], sun_seq[k], mode, sps, run, h, lagged)
            rates = {r: (temps[r] - before[r]) / h for r in temps if before.get(r) is not None}
            for uid, v in y.items():
                acc[uid] = acc.get(uid, 0.0) + v / substeps
        traj.append(temps)
        outs.append(acc)
    return traj, outs, run
