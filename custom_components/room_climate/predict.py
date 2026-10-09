"""Per-room thermal model learned online, and free-drift temperature prediction.

No Home Assistant imports. Temperatures are degrees F, times are `datetime`.

Model (per room), per minute:
    dT/dt = a * (T_out - T) + b_heat * heating + b_cool * cooling + s * sun
fitted by recursive least squares with a forgetting factor from the room's own sensor history.
`sun` is a clear-sky factor (0..1) from the real sun position at the house's location and the room's
windows (which way they face), reduced by forecast cloud cover. The heat/cool terms are learned too (so the fit sees the unit's effect), but
prediction only needs the free-drift terms: how the room will move if the unit does nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Sequence

TEMP_SCALE = 10.0  # outdoor delta is fed to the fit in units of 10F to keep the regression well conditioned
PRIOR = [0.02, 0.05, 0.05, 0.0]  # a*10, b_heat, b_cool, s  (a ~ 0.002/min ~ 8 h time constant)
BOUNDS = [(0.0, 0.2), (0.0, 0.5), (0.0, 0.5), (0.0, 0.2)]
MIN_UPDATE_MIN = 5.0  # sample spacing; temperature sensors are too coarse to difference faster
MAX_GAP_MIN = 30.0  # a longer gap (restart, sensor outage) restarts differencing


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Site:
    """Where the house is (from the address / HA's configured location)."""
    lat: float = 40.0
    lon: float = -75.0
    tz_hours: float = -5.0  # used only for naive datetimes, which are taken as local standard time


def solar_position(site: Site, when: datetime) -> tuple[float, float]:
    """(elevation, azimuth) in degrees; azimuth is compass bearing (0=N, 90=E). NOAA-style approximation."""
    if when.tzinfo is not None:
        utc = when.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        utc = when - timedelta(hours=site.tz_hours)
    doy = utc.timetuple().tm_yday
    g = 2 * math.pi / 365 * (doy - 1 + (utc.hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    minutes = utc.hour * 60 + utc.minute + utc.second / 60
    tst = (minutes + eqtime + 4 * site.lon) % 1440
    ha = math.radians(tst / 4 - 180)
    lat = math.radians(site.lat)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    zen = math.acos(_clamp(cos_zen, -1.0, 1.0))
    elev = 90 - math.degrees(zen)
    sin_zen = math.sin(zen)
    if sin_zen < 1e-6:
        return elev, 180.0
    cos_az = (math.sin(lat) * cos_zen - math.sin(decl)) / (math.cos(lat) * sin_zen)
    az = math.degrees(math.acos(_clamp(cos_az, -1.0, 1.0)))
    az = (180 + az) % 360 if ha > 0 else (180 - az) % 360
    return elev, az


def sun_times(site: Site, day: datetime) -> tuple[Optional[datetime], Optional[datetime]]:
    """(sunrise, sunset) for the date of `day` (same tz handling as `solar_position`); None in polar day/night."""
    base = day.replace(hour=0, minute=0, second=0, microsecond=0)
    rise = sset = None
    prev = solar_position(site, base)[0]
    for m in range(1, 24 * 60 + 1):
        cur = solar_position(site, base + timedelta(minutes=m))[0]
        if prev < 0 <= cur and rise is None:
            rise = base + timedelta(minutes=m)
        if prev >= 0 > cur and sset is None:
            sset = base + timedelta(minutes=m)
        prev = cur
    return rise, sset


def sun_factor(windows: Sequence[dict], site: Site, now: datetime, cloud: float = 0.0) -> float:
    """Relative solar gain through a room's windows, ~0..1 (clear sky, window area weighted).

    Each window is {"facing": compass degrees the glass faces, "area": relative size (default 1)}.
    Gain per window ~ sin(elevation) x cos(angle between sun and window normal), zero when the sun is
    behind the glass or below the horizon, cut by up to 75% for full cloud cover."""
    if not windows:
        return 0.0
    elev, az = solar_position(site, now)
    if elev <= 0:
        return 0.0
    total = area = 0.0
    for w in windows:
        wa = float(w.get("area", 1.0))
        area += wa
        facing = w.get("facing")
        if facing is None:
            continue
        inc = max(0.0, math.cos(math.radians(az - facing)))
        total += wa * math.sin(math.radians(elev)) * inc
    return (total / area if area else 0.0) * (1 - 0.75 * _clamp(cloud, 0.0, 1.0))


class RoomModel:
    """Recursive least squares over x = [(T_out-T)/10, heating, cooling, sun]."""

    def __init__(self, forget: float = 0.999, min_samples: int = 144):
        self.theta = list(PRIOR)
        self.P = [[(1e3 if i == j else 0.0) for j in range(4)] for i in range(4)]
        self.n = 0
        self.forget = forget
        self.min_samples = min_samples  # 144 samples x 5 min = 12 h
        self._prev: Optional[tuple[datetime, float, list[float]]] = None

    # -- parameters in physical units
    @property
    def a(self) -> float: return self.theta[0] / TEMP_SCALE
    @property
    def b_heat(self) -> float: return self.theta[1]
    @property
    def b_cool(self) -> float: return self.theta[2]
    @property
    def s(self) -> float: return self.theta[3]

    @property
    def ready(self) -> bool:
        return self.n >= self.min_samples

    @staticmethod
    def features(temp: float, t_out: float, running: int, sun: float) -> list[float]:
        """`running`: +1 heating, -1 cooling, 0 idle (during the interval that starts now)."""
        return [(t_out - temp) / TEMP_SCALE, 1.0 if running > 0 else 0.0, 1.0 if running < 0 else 0.0, sun]

    def update(self, now: datetime, temp: float, t_out: float, running: int, sun: float) -> None:
        """Feed one observation. Fits the change since the previous accepted one."""
        if self._prev is not None:
            t0, temp0, x = self._prev
            dt = (now - t0).total_seconds() / 60.0
            if dt < MIN_UPDATE_MIN:
                return  # keep waiting; difference over a longer span
            if dt <= MAX_GAP_MIN:
                self._rls(x, (temp - temp0) / dt)
        self._prev = (now, temp, self.features(temp, t_out, running, sun))

    def _rls(self, x: list[float], y: float) -> None:
        P, th, lam = self.P, self.theta, self.forget
        Px = [sum(P[i][j] * x[j] for j in range(4)) for i in range(4)]
        denom = lam + sum(x[i] * Px[i] for i in range(4))
        k = [Px[i] / denom for i in range(4)]
        err = y - sum(th[i] * x[i] for i in range(4))
        for i in range(4):
            th[i] = _clamp(th[i] + k[i] * err, *BOUNDS[i])
        xP = [sum(x[i] * P[i][j] for i in range(4)) for j in range(4)]
        for i in range(4):
            for j in range(4):
                P[i][j] = (P[i][j] - k[i] * xP[j]) / lam
        # Terms that are never excited (e.g. cooling in winter) would otherwise wind P up forever.
        big = max(P[i][i] for i in range(4))
        if big > 1e3:
            f = 1e3 / big
            for i in range(4):
                for j in range(4):
                    P[i][j] *= f
        self.n += 1

    def to_dict(self) -> dict:
        return {"theta": self.theta, "P": self.P, "n": self.n}

    @classmethod
    def from_dict(cls, d: dict, **kw) -> "RoomModel":
        m = cls(**kw)
        m.theta, m.P, m.n = list(d["theta"]), [list(r) for r in d["P"]], int(d["n"])
        return m


def predict_temp(model: RoomModel, temp: float, now: datetime, minutes: float,
                 outdoor_at: Callable[[datetime], float], windows: Sequence[dict], site: Site,
                 cloud_at: Optional[Callable[[datetime], float]] = None, step: float = 5.0) -> float:
    """Where the room ends up after `minutes` if its unit stays idle."""
    t, cur = temp, now
    remaining = minutes
    while remaining > 1e-9:
        d = min(step, remaining)
        sun = sun_factor(windows, site, cur, cloud_at(cur) if cloud_at else 0.0)
        t += d * (model.a * (outdoor_at(cur) - t) + model.s * sun)
        cur += timedelta(minutes=d)
        remaining -= d
    return t


def interpolator(points: Sequence[tuple[datetime, float]], fallback: float) -> Callable[[datetime], float]:
    """Linear interpolation over (time, value) points; flat outside them; `fallback` if empty."""
    pts = sorted(points)
    if not pts:
        return lambda _t: fallback

    def at(t: datetime) -> float:
        if t <= pts[0][0]:
            return pts[0][1]
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t <= t1:
                span = (t1 - t0).total_seconds()
                return v1 if span <= 0 else v0 + (v1 - v0) * (t - t0).total_seconds() / span
        return pts[-1][1]
    return at
