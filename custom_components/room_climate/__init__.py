"""Room Climate: hold each room at its own target using room sensors to drive Mitsubishi units."""
from __future__ import annotations

import dataclasses
import json
import math
import logging
import os
import pathlib
import time
from collections import deque
from datetime import datetime, timedelta

import voluptuous as vol
from homeassistant.components import persistent_notification, websocket_api
from homeassistant.components.frontend import add_extra_js_url
from homeassistant.components.climate import ATTR_HVAC_MODE, DOMAIN as CLIMATE, SERVICE_SET_HVAC_MODE, SERVICE_SET_TEMPERATURE
from homeassistant.components.climate.const import ATTR_FAN_MODE, SERVICE_SET_FAN_MODE
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import Context, Event, HomeAssistant
from homeassistant.helpers import discovery
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import (async_call_later, async_track_state_change_event, async_track_time_change,
                                         async_track_time_interval)
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import TemperatureConverter

from . import core, datalog, experiment, fit, mlmodel, mpc
from . import model as hm

DOMAIN = "room_climate"
_LOGGER = logging.getLogger(__name__)
STORE_VERSION = 1
INTERVAL = timedelta(seconds=60)
F, C = UnitOfTemperature.FAHRENHEIT, UnitOfTemperature.CELSIUS

CONFIG_SCHEMA = vol.Schema({DOMAIN: vol.Schema({vol.Optional("enabled"): bool})}, extra=vol.ALLOW_EXTRA)  # legacy key, ignored

# Who drives a unit. "thermostat": its own mode and setpoint, set by hand (the default). "room_climate": today's
# controller commands it. "model": the model-based controller (mpc.py) does, once the model has proven itself.
ROOM_CLIMATE, THERMOSTAT, MODEL = "room_climate", "thermostat", "model"
OURS = (ROOM_CLIMATE, MODEL)
SIGNAL_CONTROL = f"{DOMAIN}_control"  # dispatched (unit id) when a unit's control changes
RECONNECT_GRACE_S = 300  # a unit that just came back (reboot, firmware install, wifi) reports its own mode and setpoint
#                          for a while: that is not someone using the thermostat, and the next ticks re-send ours
OURS_GRACE_S = 30  # unit state changes this soon after we sent it something are its echo of our command
BAD_STATES = ("unknown", "unavailable")
MODEL_FILE = "room_climate_model.json"  # the fitted thermal model (model.py), refitted nightly
FIT_HISTORY_FILE = "room_climate_model_fits.jsonl"  # one line per fit: when, data span, how well it predicts
ML_ROWS_FILE = "room_climate_ml_rows.jsonl"  # the ML correction's training rows (mlmodel.py), one per completed prediction
ML_FILE = "room_climate_ml.json"  # the daily-trained ML correction (ridge, trained here)
ML_EXTERNAL_FILE = "room_climate_ml_external.json"  # a better one trained elsewhere (the Mac: ridge + trees blend); wins if present
ML_KEEP_DAYS = 14
BACKTEST_FILE = "room_climate_model_backtest.json"  # the fitted model's predictions over the logged past (charts)
FIT_DAYS = 14  # fit on this many days of the minute log
HISTORY_H = 72  # the Model page's charts reach back this far (5-minute samples, rebuilt from the minute log)


def _weight(v):
    """A weight from stored or sent data: whole numbers stay ints (as the old priorities 1-5), fractions are kept."""
    v = float(v)
    return int(v) if v == int(v) else v


def _room_from_dict(d: dict) -> core.Room:
    return core.Room(
        id=d["id"], name=d.get("name", d["id"]), unit=d.get("unit"), sensor=d.get("sensor"),
        priority=_weight(d.get("priority", 3)), default_temp=float(d.get("default_temp", 68)),
        schedule=[core.ScheduleEntry(list(e["days"]), e["start"], float(e["temp"])) for e in d.get("schedule", [])],
        priority_schedule=[core.PriorityEntry(list(e["days"]), e["start"], _weight(e["priority"]))
                           for e in d.get("priority_schedule", [])],
        hold_temp=d.get("hold_temp"),
        hold_until=dt_util.parse_datetime(d["hold_until"]) if d.get("hold_until") else None,
        floor=int(d.get("floor", 1)), shape=d.get("shape", {}))


def _unit_from_dict(d: dict) -> core.Unit:
    return core.Unit(id=d["id"], entity=d["entity"], min_setpoint=float(d.get("min_setpoint", 61)),
                     max_setpoint=float(d.get("max_setpoint", 88)))


class Manager:
    """Owns persisted config, runs the control loop, serves the card."""

    def __init__(self, hass: HomeAssistant):
        self.hass = hass
        self.store = Store(hass, STORE_VERSION, DOMAIN)
        self.data: dict = {"rooms": [], "units": [], "settings": {}, "floorplan": None}
        self.states: dict[str, core.UnitState] = {}
        self.last: dict[str, core.Command] = {}
        self.house = core.HouseState()
        self.house_info: dict = {}
        self.hist: dict[str, list] = {}  # room id -> [(time, temp F)] for trend prediction
        self.last_good: dict[str, tuple] = {}  # room id -> (temp F, reading time, last seen) to bridge sensor dropouts
        self.prev_busy: set = set()
        self.sent: dict = {}  # unit id -> (last setpoint command, when); for thermostat units, what we would have sent
        self._fan_set: set = set()  # units whose fan we've changed (put back to auto when handed to the thermostat)
        self._house_restored = False  # the house mode is taken from the units on the first tick (_restore_house)
        self.exp_block = None  # the experiment block in force (experiment.py), if one is running
        self.vote: dict = {}  # room id -> its reading smoothed for the heat/cool vote
        self._defrost_at: dict = {}  # unit id -> last time it reported DEFROST
        self._run_start: dict = {}  # unit id -> when its current run started
        self._coast: dict = {}  # unit id -> {"until", "room"}: its rooms coasting on the run that just ended
        self.drives: dict[str, core.DriveState] = {}  # unit id -> its outer loop (core.drive)
        self.zone_hist: dict[str, deque] = {}  # unit id -> [(time, its rooms' weighted reading)] for the loop's rate
        self._back_at: dict[str, datetime] = {}  # unit id -> when it last came back from unavailable
        self._sent_at: dict[str, datetime] = {}  # unit id -> when we last called a service on it
        self._ours: list[str] = []  # context ids of our recent service calls
        self._audit_seen: dict[str, tuple] = {}
        self._logged_minute = None
        self._unsub = []
        # Model-based controller in "would do" mode, and the nightly model fit (see mpc.py, fit.py).
        self.shadow = mpc.ShadowPlanner()
        self._shadow_busy = False
        # Its accuracy scores, chart points and recent readings, so a restart doesn't throw away days of proving.
        self.track_store = Store(hass, 1, f"{DOMAIN}_tracking")
        self.hist5: deque = deque()  # compact 5-minute samples for the Model page's charts (mpc.compact)
        self.backtest: dict = None  # {"fitted", "split", "points": {room: {"60"/"120": [[iso, F]]}}}
        self.forecast: dict = None  # {"at", "entity", "points": [(datetime, F, cloud %)]}: hourly, for planning
        self.irr: dict = None  # {"at", "points": [(datetime, ghi, dni, dhi)]}: HRRR sunshine, 15-min (async_update_sun)
        self._forecast_logged = None
        self.fit_state: dict = {"running": False, "error": None, "started": None, "finished": None, "seconds": None}
        self.fit_history: list = []
        self.ml_state: dict = None  # the ML correction's last training / loading (mlmodel.py)
        self.ml_external = False  # an externally trained correction (ML_EXTERNAL_FILE) is in use

    async def async_load(self):
        loaded = await self.store.async_load()
        if loaded is None:  # first run: seed from the bundled example layout
            seed = pathlib.Path(__file__).parent / "default_config.json"
            loaded = await self.hass.async_add_executor_job(lambda: json.loads(seed.read_text()))
            loaded.pop("_comment", None)
        loaded.pop("enabled", None)  # the old dry-run switch; control is now per unit
        for ud in loaded.get("units", []):
            ud.setdefault("control", THERMOSTAT)
        self.data = loaded
        await self._load_model()

    async def _load_model(self):
        def read():
            p = pathlib.Path(self.hass.config.path(MODEL_FILE))
            m = hm.HouseModel.from_dict(json.loads(p.read_text())) if p.exists() else None
            h = pathlib.Path(self.hass.config.path(FIT_HISTORY_FILE))
            hist = [json.loads(x) for x in h.read_text().splitlines()[-60:] if x.strip()] if h.exists() else []
            b = pathlib.Path(self.hass.config.path(BACKTEST_FILE))
            bt = json.loads(b.read_text()) if b.exists() else None
            return m, hist, bt
        try:
            m, self.fit_history, self.backtest = await self.hass.async_add_executor_job(read)
        except (OSError, ValueError) as e:
            _LOGGER.warning("couldn't load the thermal model: %s", e)
            return
        if m is not None:
            self.shadow.set_model(m)
        try:
            self.shadow.load_state(await self.track_store.async_load(), dt_util.now())
        except (ValueError, TypeError, KeyError) as e:
            _LOGGER.warning("couldn't restore the model's accuracy scores: %s", e)
        await self._load_ml()

    async def _load_ml(self):
        """The ML correction to use: one trained elsewhere (ML_EXTERNAL_FILE) if there is one, else the one trained here last
        night. Absence or a bad file changes nothing. Also what room_climate/reload_ml runs after a new file is copied in."""
        def read():
            for name, external in ((ML_EXTERNAL_FILE, True), (ML_FILE, False)):
                p = pathlib.Path(self.hass.config.path(name))
                if p.exists():
                    try:
                        return mlmodel.load(json.loads(p.read_text())), external, name
                    except Exception as e:  # noqa: BLE001 - a bad file: fall back to the next
                        _LOGGER.warning("couldn't read %s: %s", name, e)
            return None, False, None
        try:
            c, external, name = await self.hass.async_add_executor_job(read)
        except Exception as e:  # noqa: BLE001 - never let the ML take the integration down
            _LOGGER.warning("couldn't load the ML correction: %s", e)
            return
        self.ml_external = external and c is not None
        if c is not None:
            self.shadow.set_corrector(c)
            self.ml_state = {"loaded": dt_util.now().isoformat(), "file": name, "kind": getattr(c, "kind", "ridge"),
                             "enabled": sum(c.enabled.values()), "of": len(c.enabled)}
            _LOGGER.info("ML correction loaded from %s (%s)", name, self.ml_state["kind"])

    async def async_flush_ml(self, _now=None):
        """Append the training rows the shadow planner has logged since the last flush to the ML rows file."""
        rows = self.shadow.drain_ml_rows()
        if not rows:
            return
        path = self.hass.config.path(ML_ROWS_FILE)

        def write():
            with open(path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        try:
            await self.hass.async_add_executor_job(write)
        except OSError as e:
            _LOGGER.warning("couldn't write the ML rows: %s", e)

    async def async_train_ml(self, _now=None):
        """Train the ML correction on the last ML_KEEP_DAYS of logged rows (in an executor job), save it, use it. The rows
        are the physics model's out-of-sample misses (made by whatever model was live), so each night's data is fresh."""
        await self.async_flush_ml()
        rows_path, ml_path = self.hass.config.path(ML_ROWS_FILE), self.hass.config.path(ML_FILE)
        cutoff = (dt_util.now() - timedelta(days=ML_KEEP_DAYS)).isoformat()

        def work():
            rows = []
            if os.path.exists(rows_path):
                with open(rows_path) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if r[0] >= cutoff:
                            rows.append(r)
                with open(rows_path + ".tmp", "w") as f:  # keep the file to the window
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                os.replace(rows_path + ".tmp", rows_path)
            c = mlmodel.train(rows)
            if c is not None:
                with open(ml_path + ".tmp", "w") as f:
                    json.dump(c.to_dict(), f)
                os.replace(ml_path + ".tmp", ml_path)
            return c, len(rows)
        try:
            c, n = await self.hass.async_add_executor_job(work)
            if c is not None:
                if not self.ml_external:  # a model trained elsewhere stays in use; this one is still saved
                    self.shadow.set_corrector(c)
                self.ml_state = {"trained": dt_util.now().isoformat(), "rows": n, "enabled": sum(c.enabled.values()),
                                 "of": len(c.enabled), "held_out_rmse_F": c.report, "in_use": "external" if self.ml_external else "ridge"}
                _LOGGER.info("ML correction trained on %s rows; enabled for %s of %s room/horizon pairs", n,
                             sum(c.enabled.values()), len(c.enabled))
            else:
                self.ml_state = {"trained": None, "rows": n, "why": "not enough rows yet"}
        except Exception as e:  # noqa: BLE001 - a failed training keeps yesterday's correction
            _LOGGER.warning("ML correction training failed: %s: %s", type(e).__name__, e)

    async def async_load_history(self):
        """Rebuild the charts' 5-minute samples from the last HISTORY_H of the minute log (once, at startup)."""
        logdir = self.hass.config.path("room_climate_data")
        since = dt_util.now() - timedelta(hours=HISTORY_H)

        def read():
            if not os.path.isdir(logdir):
                return []
            out = []
            for row in datalog.read_rows(logdir, since.strftime("%Y-%m-%d")):
                try:
                    t = datetime.fromisoformat(row["t"])
                except (KeyError, ValueError):
                    continue
                if t >= since and t.minute % 5 == 0:
                    out.append(mpc.compact(row))
            return out

        try:
            rows = await self.hass.async_add_executor_job(read)
        except OSError as e:
            _LOGGER.warning("couldn't rebuild the model charts' history: %s", e)
            return
        self.hist5 = deque(rows + list(self.hist5))  # anything logged meanwhile goes after

    def room_history(self, room_id: str, hours: float) -> dict:
        """One room's reading against its target (and weight) from the 5-minute samples: [[iso, temp, target, weight], ...]."""
        since = (dt_util.now() - timedelta(hours=hours)).isoformat()
        pts = [[x["t"], *x["r"][room_id]] for x in self.hist5 if x["t"] >= since and room_id in x["r"]]
        return {"room": room_id, "hours": hours, "points": pts}

    def model_history(self, hours: float) -> dict:
        """What the Model page's charts plot: samples, comfort score (actual vs the model's prediction), prediction
        accuracy by hour, predictions as they were made, and the two 2 h projections."""
        now = dt_util.now()
        since = (now - timedelta(hours=hours)).isoformat()
        samples = [x for x in self.hist5 if x["t"] >= since]
        s = core.settings_from(self.data.get("settings"))
        ent = {u["id"]: u["entity"] for u in self.data["units"]}
        m = self.shadow.model
        scored = [r["id"] for r in self.data["rooms"] if r.get("unit") and (m is None or r["id"] in m.rooms)
                  and r.get("sensor") and r.get("sensor") != ent.get(r.get("unit"))]
        points = {}
        for t, rid, h, v in self.shadow.points:
            if t.isoformat() >= since:
                points.setdefault(rid, {}).setdefault(str(h), []).append([t.isoformat(), v])
        p = self.shadow.plan
        plan = None if p is None else {r: [[(self.shadow.plan_at + timedelta(minutes=self.shadow.ps.step_min * (k + 1))).isoformat(), v]
                                           for k, v in enumerate(traj)] for r, traj in p.predicted.items()}
        bt = self.backtest or {}
        backtest = {"fitted": bt.get("fitted"), "split": bt.get("split"),
                    "points": {r: {h: [p for p in pts if p[0] >= since] for h, pts in by.items()}
                               for r, by in (bt.get("points") or {}).items()}}
        return {"now": now.isoformat(), "hours": hours, "samples": samples, "scored": scored, "backtest": backtest,
                "comfort": mpc.comfort_series(samples, scored, s), "accuracy": mpc.accuracy_by_hour(self.shadow.errors),
                "points": points, "projection": self.shadow.projection, "plan": plan,
                "band": [s.band_below, s.band_above],
                "units": {u["id"]: u.get("name") or u["id"] for u in self.data["units"]},
                "rooms": {r["id"]: r.get("name") or r["id"] for r in self.data["rooms"]}}

    async def async_update_forecast(self, _now=None):
        """Fetch the hourly forecast (outdoor temperature and cloud cover) for the planner's next hours."""
        ent = self._outdoor_entity()
        st = self.hass.states.get(ent) if ent else None
        if st is None or not ent.startswith("weather."):
            return
        try:
            resp = await self.hass.services.async_call("weather", "get_forecasts", {"entity_id": ent, "type": "hourly"},
                                                       blocking=True, return_response=True)
            items = (resp or {}).get(ent, {}).get("forecast") or []
        except Exception as e:  # no hourly forecast from this integration, or it's down: plan on current weather
            _LOGGER.debug("hourly forecast unavailable from %s: %s", ent, e)
            return
        unit = st.attributes.get("temperature_unit") or self.hass.config.units.temperature_unit
        now, pts = dt_util.now(), []
        for it in items:
            t = dt_util.parse_datetime(it.get("datetime") or "")
            if t is None or t > now + timedelta(hours=12):
                continue
            temp = it.get("temperature")
            try:
                temp = None if temp is None else TemperatureConverter.convert(float(temp), unit, F)
            except (TypeError, ValueError):
                temp = None
            pts.append((t, temp, it.get("cloud_coverage")))
        if pts:
            self.forecast = {"at": now, "entity": ent, "points": pts}

    async def async_update_sun(self, _now=None):
        """Estimated sunshine at the house every 15 min: NOAA's HRRR model through Open-Meteo (free, no key). Direct and
        diffuse sunlight rather than an hourly cloud cover: what heats a south window. Kept from 2 h back to 3 h ahead."""
        lat, lon = self.hass.config.latitude, self.hass.config.longitude
        url = ("https://api.open-meteo.com/v1/forecast?latitude=%.4f&longitude=%.4f&models=gfs_hrrr&timeformat=unixtime"
               "&minutely_15=shortwave_radiation,direct_normal_irradiance,diffuse_radiation"
               "&past_minutely_15=8&forecast_minutely_15=12" % (lat, lon))
        try:
            async with async_get_clientsession(self.hass).get(url, timeout=20) as resp:
                resp.raise_for_status()
                m = (await resp.json())["minutely_15"]
            # Each value is the mean of the 15 minutes before its time stamp: centre it in that quarter hour.
            pts = [(dt_util.utc_from_timestamp(t) - timedelta(minutes=7.5), g, n, f)
                   for t, g, n, f in zip(m["time"], m["shortwave_radiation"], m["direct_normal_irradiance"],
                                         m["diffuse_radiation"]) if None not in (g, n, f)]
        except Exception as e:  # offline or changed: fall back on the forecast's cloud cover
            _LOGGER.debug("sunshine estimate unavailable: %s", e)
            return
        if pts:
            self.irr = {"at": dt_util.now(), "points": pts}

    def irradiance_at(self, t):
        """(ghi, dni, dhi) at time t from the HRRR estimate, interpolated; None if there isn't a fresh one covering t."""
        f = self.irr
        if f is None or dt_util.now() - f["at"] > timedelta(hours=1):
            return None
        pts = f["points"]
        for (ta, *a), (tb, *b) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                w = (t - ta).total_seconds() / max(1.0, (tb - ta).total_seconds())
                return tuple(x + (y - x) * w for x, y in zip(a, b))
        return None

    def _weather_ahead(self, now, outdoor, cloud):
        """(outdoor(t) or the current value, cloud(t)) for the planner: the hourly forecast when fresh, else now."""
        f = self.forecast
        if f is None or now - f["at"] > timedelta(hours=3):
            return outdoor, (lambda t: cloud)
        return (mpc.along(now, outdoor, [(t, v) for t, v, _ in f["points"]]),
                mpc.along(now, cloud, [(t, c) for t, _, c in f["points"]]))

    def model_age_h(self):
        m = self.shadow.model
        try:
            fitted = datetime.fromisoformat(m.meta["fitted"]) if m is not None else None
        except (KeyError, TypeError, ValueError):
            fitted = None
        if fitted is None:
            return None
        if fitted.tzinfo is None:
            fitted = fitted.replace(tzinfo=dt_util.now().tzinfo)
        return (dt_util.now() - fitted).total_seconds() / 3600

    async def async_fit(self, _now=None):
        """Refit the thermal model on the last FIT_DAYS of the minute log (in an executor job), save it, use it."""
        if self.fit_state["running"]:
            return
        self.fit_state.update(running=True, error=None, started=dt_util.now().isoformat())
        rooms = [_room_from_dict(r) for r in self.data["rooms"]]
        units = [_unit_from_dict(ud) for ud in self.data["units"]]
        layout = fit.layout_for(rooms, units, {r["id"]: r["fit_since"] for r in self.data["rooms"] if r.get("fit_since")})
        since = (dt_util.now() - timedelta(days=FIT_DAYS)).strftime("%Y-%m-%d")
        logdir = self.hass.config.path("room_climate_data")
        model_path, hist_path = self.hass.config.path(MODEL_FILE), self.hass.config.path(FIT_HISTORY_FILE)
        bt_path = self.hass.config.path(BACKTEST_FILE)
        recent = (dt_util.now() - timedelta(hours=HISTORY_H)).isoformat()
        t0 = time.monotonic()

        def work():
            rows = list(datalog.read_rows(logdir, since)) if os.path.isdir(logdir) else []
            if len(rows) < 600:
                raise ValueError(f"only {len(rows)} minutes of log so far (need 600)")
            m, rep = fit.fit(rows, layout)
            tmp = model_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(m.to_dict(), f, indent=1)
            os.replace(tmp, model_path)
            # Run the new model over the logged past the charts show, to plot how well it fits it.
            bt = {"fitted": m.meta.get("fitted"), "split": rep.get("split"),
                  "points": fit.backtest(m, [r for r in rows if r.get("t", "") >= recent])}
            with open(bt_path + ".tmp", "w") as f:
                json.dump(bt, f)
            os.replace(bt_path + ".tmp", bt_path)
            entry = {k: m.meta.get(k) for k in ("fitted", "usable", "rows", "from", "to")}
            entry["holdout"] = rep.get("holdout", {})
            entry["skipped"] = {k: v["skipped"] for k, v in rep.items() if isinstance(v, dict) and "skipped" in v}
            with open(hist_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            return m, entry, bt

        try:
            m, entry, self.backtest = await self.hass.async_add_executor_job(work)
            self.shadow.set_model(m)
            self.fit_history = (self.fit_history + [entry])[-60:]
            _LOGGER.info("thermal model refitted on %s minutes: usable=%s", entry["rows"], entry["usable"])
        except Exception as e:  # a failed fit keeps the old model; the debug page shows why
            self.fit_state["error"] = f"{type(e).__name__}: {e}"
            _LOGGER.warning("thermal model fit failed: %s", self.fit_state["error"])
        finally:
            self.fit_state.update(running=False, finished=dt_util.now().isoformat(),
                                  seconds=round(time.monotonic() - t0, 1))

    def _restore_house(self, units: list, now):
        """After a restart, carry on in the mode the units we control are in. The house mode isn't stored, and starting
        from "off" left them on their old setpoints, unmanaged, until a room asked for enough (Bedroom 3's unit kept
        cooling a 66 F room for 45 min). Done once, when every unit we control reports a mode; only if they agree."""
        ours = [u for u in units if self.control(u.id) in OURS]
        if not ours:
            self._house_restored = True
            return
        modes = []
        for u in ours:
            cur = self.hass.states.get(u.entity)
            if cur is None or cur.state in BAD_STATES:
                return  # not reported yet: try again next tick
            modes.append(cur.state)
        self._house_restored = True
        if len(set(modes)) == 1 and modes[0] in ("heat", "cool"):
            # How long it's been in this mode isn't known: no minimum-time hold, just the usual switch confirmation.
            self.house = core.HouseState(modes[0], None)
            for u in ours:
                st = self.states.get(u.id, core.UnitState())
                self.states[u.id] = core.UnitState(modes[0], now, modes[0], st.active)
            _LOGGER.info("house mode %s restored from the units after a restart", modes[0])

    async def _experiment_tick(self, now, units, rooms, readings, cmds, settings, info):
        """While an experiment runs (experiment.py): its commands for the units we control (None otherwise, or in the
        last hour, which is normal control). Ends it when the time is up."""
        exp = self.data.get("experiment")
        self.exp_block = None
        if not exp:
            return None
        if now >= datetime.fromisoformat(exp["end"]):
            if exp.get("vane0") and not exp.get("vane_back"):
                await self._restore_vanes(exp)
            _LOGGER.info("experiment finished (%s to %s)", exp["start"], exp["end"])
            self.data.pop("experiment", None)
            self.drives.clear()
            await self.async_save()
            return None
        b = self.exp_block = experiment.current(exp, now)
        if b is None or b["kind"] == "restore":
            if exp.get("vane0") and not exp.get("vane_back"):
                await self._restore_vanes(exp)
                exp["vane_back"] = True
            return None
        mode = exp["mode"]
        if self.house.mode != mode:
            self.house = core.HouseState(mode, now)
        info["mode"], info["reason"] = mode, "experiment: " + experiment.describe(b, self._unit_names())
        ours = {u.id for u in units if self.control(u.id) in OURS}
        unit_rooms = {u.id: [r.id for r in rooms if r.unit == u.id] for u in units}
        return experiment.commands(b, mode, {k: v for k, v in cmds.items() if k in ours}, unit_rooms,
                                   {rid: v[0] for rid, v in readings.items()}, settings)

    def _unit_names(self) -> dict:
        return {u["id"]: (u.get("name") or u["id"]).replace(" minisplit", "").replace(" air handler", "")
                for u in self.data["units"]}

    async def async_start_experiment(self, hours: float):
        """Run an experiment for `hours` (experiment.plan). Raises ValueError if no unit is under our control."""
        settings = core.settings_from(self.data.get("settings"))
        ours = [u["id"] for u in self.data["units"] if self.control(u["id"]) in OURS]
        if not ours:
            raise ValueError("no unit is under Room Climate or Model control")
        now = dt_util.now()
        mode = experiment.pick_mode(self.house.mode, self._outdoor(settings))
        fan_units = {x.strip() for x in settings.fan_units.split(",")}
        m = self.shadow.model
        ours = experiment.order(ours, experiment.leaky_units(m.rooms.values() if m else ()))
        self.data["experiment"] = experiment.plan(now, hours, ours, fan_units, mode,
                                                    experiment.leaky_units(m.rooms.values() if m else ()),
                                                    settings.experiment_focus,
                                                    [x.strip() for x in settings.experiment_vane_units.split(",") if x.strip()])
        self.data["experiment"]["vane0"] = {u["id"]: self._vane(u["entity"]) for u in self.data["units"]
                                            if self._vane(u["entity"]) is not None}  # to put the vanes back afterwards
        _LOGGER.info("experiment started: %s h in %s mode for %s", hours, mode, ", ".join(ours))
        await self.async_save()
        await self.async_tick()

    async def async_stop_experiment(self):
        exp = self.data.get("experiment")
        if exp and exp.get("vane0"):
            await self._restore_vanes(exp)
        if self.data.pop("experiment", None) is not None:
            _LOGGER.info("experiment stopped from the card")
            self.drives.clear()
            await self.async_save()
            await self.async_tick()

    def drive_lead(self, unit_id: str):
        """Look-ahead for a unit's outer loop: its learned lag in minutes (None: the default lead)."""
        m = self.shadow.model
        u = m.units.get(unit_id) if m is not None else None
        return u.lag_h * 60 if u is not None and u.lag_h > 0 else None

    def _own_effect(self, unit_id: str, mode: str, rooms: list):
        m = self.shadow.model
        if m is None or unit_id not in m.units:
            return None, None
        eff = [(m.rooms[r.id].heat if mode == "heat" else m.rooms[r.id].cool).get(unit_id, 0.0)
               for r in rooms if r.unit == unit_id and r.id in m.rooms]
        # on the default output rule's scale, which the outer loop's gain and push limits are tuned for
        return hm.rule_effect(sum(eff) / len(eff) if eff else None, m.units[unit_id]), m.units[unit_id]

    def drive_max_push(self, unit_id: str, mode: str, rooms: list, settings: core.Settings) -> float:
        """How far past the room a unit may be pushed: set per unit, else from its learned effect (core.max_push_for)."""
        eff, _ = self._own_effect(unit_id, mode, rooms)
        return core.max_push_for(unit_id, eff, settings)

    def drive_kp(self, unit_id: str, mode: str, rooms: list, settings: core.Settings) -> float:
        """Outer-loop gain for a unit, from the model's learned effect of the unit on its own rooms (core.drive_gain)."""
        eff, um = self._own_effect(unit_id, mode, rooms)
        if um is None:
            return settings.drive_kp
        return core.drive_gain(eff, settings)

    def control(self, unit_id: str) -> str:
        ud = next((u for u in self.data["units"] if u["id"] == unit_id), {})
        return ud.get("control", THERMOSTAT)

    def model_rooms(self) -> set:
        """Rooms the model must predict well before it may control units: every room with a sensor served by a unit,
        modelled yet or not. A room the fit hasn't taken in yet (a new or moved sensor) has no scores, so it holds the
        gate shut rather than letting the model drive a unit whose room it can't predict."""
        if self.shadow.model is None:
            return set()
        return {r["id"] for r in self.data["rooms"] if r.get("unit") and r.get("sensor")}

    def model_ready(self, strict: bool = True):
        return self.shadow.readiness(dt_util.now(), self.model_rooms(), strict=strict,
                                     skip_fit=core.settings_from(self.data.get("settings")).model_gate_skip_fit)

    async def async_set_control(self, unit_id: str, control: str, why: str):
        """Raises ValueError (with the reasons) when asked to hand a unit to a model that isn't ready."""
        ud = next((u for u in self.data["units"] if u["id"] == unit_id), None)
        if ud is None or ud.get("control") == control:
            return
        if control == MODEL:
            ok, reasons = self.model_ready(strict=True)
            if not ok:
                raise ValueError("the model isn't ready to control units: " + "; ".join(reasons))
        ud["control"] = control
        if control == THERMOSTAT and unit_id in self._fan_set and not why.startswith("changed at the thermostat"):
            # handed back from the card: give the unit its own (auto) fan again; a change made at the unit is left be
            self._fan_set.discard(unit_id)
            cur = self.hass.states.get(ud["entity"])
            if cur is not None and cur.state not in BAD_STATES and cur.attributes.get("fan_mode") != "auto":
                await self._call(core.Unit(ud["id"], ud["entity"]), SERVICE_SET_FAN_MODE, {ATTR_FAN_MODE: "auto"})
        self.sent.pop(unit_id, None)  # taking over: send our setpoint now, not after the change interval
        self.drives.pop(unit_id, None)
        _LOGGER.info("%s control -> %s (%s)", unit_id, control, why)
        await self.async_save()
        async_dispatcher_send(self.hass, SIGNAL_CONTROL, unit_id)
        await self.async_tick()

    async def async_unit_changed(self, event: Event):
        """A unit under our control changed mode or setpoint without us: someone used its thermostat, the app or the
        remote. Hand it to the thermostat so we don't fight them."""
        ud = next((u for u in self.data["units"] if u["entity"] == event.data["entity_id"]), None)
        old, new = event.data.get("old_state"), event.data.get("new_state")
        if ud is None or ud.get("control") not in OURS or old is None or new is None:
            return
        if old.state in BAD_STATES or new.state in BAD_STATES:
            if new.state not in BAD_STATES:
                self._back_at[ud["id"]] = dt_util.utcnow()
            return  # dropping off and reconnecting is not a change of mind
        back = self._back_at.get(ud["id"])
        if back and core.elapsed(dt_util.utcnow(), back) < RECONNECT_GRACE_S:
            _LOGGER.info("%s came back %s: keeping it under %s (its reported %s/%s is not a change of mind)", ud["id"],
                         "just now", ud["control"], new.state, new.attributes.get("temperature"))
            return  # (2026-10-25: firmware installs handed two units to their thermostats and the house sat in cool all day)
        sp_old, sp_new = old.attributes.get("temperature"), new.attributes.get("temperature")
        fan_old, fan_new = old.attributes.get("fan_mode"), new.attributes.get("fan_mode")
        if old.state == new.state and sp_old == sp_new and fan_old == fan_new:
            return
        at = self._sent_at.get(ud["id"])
        if event.context.id in self._ours or (at and core.elapsed(dt_util.utcnow(), at) < OURS_GRACE_S):
            return
        await self.async_set_control(ud["id"], THERMOSTAT,
                                     f"changed at the thermostat: {old.state}/{sp_old}/{fan_old} -> {new.state}/{sp_new}/{fan_new}")

    async def async_save(self):
        await self.store.async_save(self.data)

    def _f(self, entity_id: str):
        """Entity state as degrees F (None if unavailable)."""
        st = self.hass.states.get(entity_id)
        if st is None or st.state in ("unknown", "unavailable"):
            return None, None
        if entity_id.startswith("climate."):
            # A room with no separate sensor can use its unit's own reading (the climate's current temperature).
            cur = st.attributes.get("current_temperature")
            if cur is None:
                return None, None
            # A climate entity only reports when something changes, so a steady room would look stale after
            # stale_s. If the unit drops off, the entity goes unavailable (handled above), so available = live.
            return TemperatureConverter.convert(float(cur), self.hass.config.units.temperature_unit, F), dt_util.now()
        try:
            val = float(st.state)
        except ValueError:
            return None, None
        unit = st.attributes.get("unit_of_measurement") or self.hass.config.units.temperature_unit
        # last_reported moves on every report; last_updated only when the value changes, which would make a
        # steady sensor look stale.
        return TemperatureConverter.convert(val, unit, F), getattr(st, "last_reported", None) or st.last_updated

    def _outdoor_entity(self, settings: core.Settings | None = None) -> str | None:
        settings = settings or core.settings_from(self.data.get("settings"))
        return settings.outdoor_entity or next(iter(sorted(self.hass.states.async_entity_ids("weather"))), None)

    def _outdoor(self, settings: core.Settings):
        """Outdoor temperature in F from `outdoor_entity`, else the first weather entity (None if unknown)."""
        ent = self._outdoor_entity(settings)
        st = self.hass.states.get(ent) if ent else None
        if st is None or st.state in ("unknown", "unavailable"):
            return None
        try:
            if ent.startswith("weather."):
                val = st.attributes.get("temperature")
                unit = st.attributes.get("temperature_unit") or self.hass.config.units.temperature_unit
            else:
                val, unit = st.state, st.attributes.get("unit_of_measurement") or self.hass.config.units.temperature_unit
            return None if val is None else TemperatureConverter.convert(float(val), unit, F)
        except (TypeError, ValueError):
            return None

    def _unit_temp(self, entity_id: str):
        st = self.hass.states.get(entity_id)
        cur = st.attributes.get("current_temperature") if st else None
        return None if cur is None else TemperatureConverter.convert(float(cur), self.hass.config.units.temperature_unit, F)

    async def async_tick(self, _now=None):
        # Decide for every unit, but send only to units under our control; for the rest the decision is shown as what
        # Room Climate would do.
        states = self.states
        now = dt_util.now()
        settings = core.settings_from(self.data.get("settings"))
        rooms = [_room_from_dict(r) for r in self.data["rooms"]]
        units = [_unit_from_dict(ud) for ud in self.data["units"]]
        readings, predicted = {}, {}
        for r in rooms:
            if not r.sensor:
                continue
            val, ts = self._f(r.sensor)
            got, self.last_good[r.id] = core.hold_reading(None if val is None else (val, ts), self.last_good.get(r.id), now, settings)
            if got is None:
                continue
            readings[r.id] = got
            if val is None:
                continue  # bridging a dropout: no new point for the trend
            h = self.hist.setdefault(r.id, [])
            h.append((now, val))
            del h[:max(0, next((k for k, (t, _) in enumerate(h) if core.elapsed(now, t) <= settings.history_s), len(h)))]
            predicted[r.id] = core.predict(h, now, settings)
        # Units under model control go back to today's controller as soon as the model stops being trustworthy
        # (unless the owner turned model_fallback off: then they stay, and a notification says the model isn't ready).
        model_units = [u for u in units if self.control(u.id) == MODEL]
        if model_units:
            ok, reasons = self.model_ready(strict=False)
            if ok:
                self._model_warned = False
            elif not settings.model_fallback:
                if not getattr(self, "_model_warned", False):
                    self._model_warned = True
                    persistent_notification.async_create(
                        self.hass, "The model isn't ready, and " + ", ".join(u.id for u in model_units)
                        + " stay on it (model_fallback is off): " + "; ".join(reasons[:3]), title="Room Climate",
                        notification_id=f"{DOMAIN}_model_not_ready")
            else:
                for u in model_units:
                    await self.async_set_control(u.id, ROOM_CLIMATE, "model not ready: " + reasons[0])
                persistent_notification.async_create(
                    self.hass, "The model-based controller handed " + ", ".join(u.id for u in model_units)
                    + " back to Room Climate: " + "; ".join(reasons[:3]), title="Room Climate",
                    notification_id=f"{DOMAIN}_model_fallback")
                return  # async_set_control already ran a tick with the new control
        manual, busy = {}, set()
        for u in units:
            if self.control(u.id) == MODEL and self.shadow.house.mode in ("heat", "cool"):
                # The model decides these units, and today's controller follows its mode for the shared compressor.
                manual[u.id] = self.shadow.house.mode
            cur = self.hass.states.get(u.entity)
            if self.control(u.id) == THERMOSTAT:  # even when unavailable: still not ours to drive
                manual[u.id] = cur.state if cur is not None else "unavailable"
            if cur is not None and cur.attributes.get("hvac_action") in ("heating", "cooling"):
                busy.add(u.id)
        for u in units:  # a compressor that started/stopped on its own makes the earlier trend misleading
            if (u.id in busy) != (u.id in self.prev_busy):
                for r in rooms:
                    if r.unit == u.id and self.hist.get(r.id):
                        self.hist[r.id] = self.hist[r.id][-1:]
                        predicted[r.id] = None
        self.prev_busy = busy
        # The heat/cool vote reads each room smoothed (core.Settings.vote_smooth_min): fresh readings only.
        k = 1.0 - math.exp(-1.0 / max(0.1, settings.vote_smooth_min))  # per one-minute tick
        for rid, (val, _ts) in readings.items():
            prev = self.vote.get(rid)
            self.vote[rid] = val if prev is None else prev + (val - prev) * k
        # Units defrosting (sub-mode DEFROST: heat briefly off, coil cold), and for 10 min after: their rooms dip and
        # recover on their own, so they sit out the vote and the outer loop holds its setpoint.
        defrosting = set()
        for u in units:
            sub = next((self.hass.states.get(e) for e in self._unit_sensors(u.entity) if e.endswith("_sub_mode")), None)
            if sub is not None and str(sub.state).upper() == "DEFROST":
                self._defrost_at[u.id] = now
            last = self._defrost_at.get(u.id)
            if last is not None and core.elapsed(now, last) < 600:
                defrosting.add(u.id)
        coasting = set()  # units that stopped recently: their rooms are still moving on the unit's own output
        for u in units:
            mine = [r for r in rooms if r.unit == u.id]
            res = core.weighted_error(mine, readings, now, settings)
            room_now = None if res is None else res[1]
            if u.id in busy:
                self._run_start.setdefault(u.id, now)
                self._coast.pop(u.id, None)
            elif u.id in self._run_start:  # just stopped: its coast starts, as long as the run that caused it
                run_s = core.elapsed(now, self._run_start.pop(u.id))
                self._coast[u.id] = {"until": core.coast_until(now, run_s, self.drive_lead(u.id) or 0.0, settings),
                                     "room": room_now}
            if core.still_coasting(self._coast.get(u.id), now, room_now, settings):
                coasting.add(u.id)
            else:
                self._coast.pop(u.id, None)
        if not self._house_restored:
            self._restore_house(units, now)
        house = self.house
        cmds, new_house, info = core.control_step(
            units, rooms, readings, predicted, {u.id: self._unit_temp(u.entity) for u in units},
            states, house, now, settings, manual, busy, self._outdoor(settings), coasting, dict(self.vote),
            defrosting)
        if new_house.mode != house.mode:
            _LOGGER.info("house mode %s -> %s (%s)", house.mode, new_house.mode, info["reason"])
        self.house = new_house
        exp_cmds = await self._experiment_tick(now, units, rooms, readings, cmds, settings, info)
        self.house_info = {**info, "since": self.house.since.isoformat() if self.house.since else None,
                           "predicted": {k: None if v is None else round(v, 1) for k, v in predicted.items()}}
        self._tick_debug = (readings, predicted, info, settings)
        sent = self.sent
        for u in units:
            ctl = self.control(u.id)
            c = cmds[u.id]
            if ctl == MODEL:
                d = self.shadow.decision.get(u.id) or {}
                reason = "model: " + ((self.shadow.live_reason if settings.model_planner == "joint" else None)
                                      or (self.shadow.plan.reason if self.shadow.plan else ""))
                if d.get("sleep"):
                    c = core.Command("off", None, reason + " (this unit fully off: no leak)")
                elif settings.model_planner == "joint" and d.get("mode") in ("heat", "cool"):
                    # the joint planner picks run / fan / setpoint itself: no outer loop
                    c = core.Command(d["mode"], d.get("setpoint"), reason, bool(d.get("run")), d.get("setpoint"), None,
                                     c.room, c.offset, d.get("fan"))
                elif d.get("mode") in ("heat", "cool") and settings.drive and d.get("target") is not None:
                    # the model picks the target; the same outer loop as today's controller sets the setpoint
                    c = core.Command(d["mode"], None, reason, True, None, d["target"], c.room, c.offset)
                elif d.get("mode") in ("heat", "cool"):
                    c = core.Command(d["mode"], d.get("setpoint"), reason, True, d.get("setpoint"))
                else:
                    c = core.Command(states.get(u.id, core.UnitState()).mode, None, "model: no change needed")
            if exp_cmds and u.id in exp_cmds:
                cmd, changed = exp_cmds[u.id], True  # an experiment block; the outer loop starts afresh after it
                self.drives.pop(u.id, None)
            elif u.id in defrosting and self.last.get(u.id) is not None and self.last[u.id].mode == c.mode:
                cmd, changed = self.last[u.id], False  # defrosting: hold what it had; the dip isn't a need
                cmd.reason = "defrosting; holding"
            elif settings.drive and c.mode in ("heat", "cool") and c.target is not None:
                # Outer loop: the setpoint follows the room, pushing gently and stopping before an overshoot.
                h = self.zone_hist.setdefault(u.id, deque())
                if c.room is not None and (not h or core.elapsed(now, h[-1][0]) >= 30):
                    h.append((now, c.room))
                while h and core.elapsed(now, h[0][0]) > 2 * settings.drive_slope_s:
                    h.popleft()
                cmd, self.drives[u.id], changed = core.drive(
                    c, self.drives.get(u.id), core.slope(list(h), now, settings.drive_slope_s),
                    self.drive_kp(u.id, c.mode, rooms, settings), u, now, settings, self.drive_lead(u.id),
                    self.drive_max_push(u.id, c.mode, rooms, settings))
                if ctl == MODEL:
                    cmd.reason = f"{c.reason} -> {cmd.reason}"
            elif ctl == MODEL:
                cmd, changed = c, False
            else:
                # Keep setpoints steady (inverter efficiency): small or too-frequent changes hold the previous one.
                cmd, changed = core.stabilize(c, sent.get(u.id), now, settings)
            if changed and cmd.setpoint is not None:
                sent[u.id] = (cmd, now)
            state = states.setdefault(u.id, core.UnitState())
            self.last[u.id] = cmd
            ours = ctl in OURS
            if ours:
                await self._send(u, state, cmd)
            self._audit(u, cmd, ours, now)
            states[u.id] = core.apply(state, cmd, now)
            if core.regime_changed(state, states[u.id]):
                for r in rooms:
                    if r.unit == u.id and self.hist.get(r.id):
                        self.hist[r.id] = self.hist[r.id][-1:]
        self._publish(now, rooms)
        await self._shadow_tick(now, rooms, units, readings, settings)
        await self._log(now, rooms, units, predicted, info)

    async def _shadow_tick(self, now, rooms, units, readings, settings):
        """Feed the model-based controller ("would do") and its accuracy tracking; plan in an executor job."""
        temps = {rid: v[0] for rid, v in readings.items()}
        disp = self.hass.config.units.temperature_unit
        actual = {}
        for u in units:
            cur = self.hass.states.get(u.entity)
            if cur is None or cur.state in BAD_STATES:
                continue
            sp = cur.attributes.get("temperature")
            actual[u.id] = {"mode": cur.state, "running": cur.attributes.get("hvac_action") in ("heating", "cooling"),
                            "fan": cur.attributes.get("fan_mode"), "temp": self._unit_temp(u.entity),
                            "setpoint": None if sp is None else TemperatureConverter.convert(float(sp), disp, F)}
        outdoor = self._outdoor(settings)
        ent = self._outdoor_entity(settings)
        wx = self.hass.states.get(ent) if ent else None
        cloud = wx.attributes.get("cloud_coverage") if wx is not None else None
        lat, lon = self.hass.config.latitude, self.hass.config.longitude

        outdoor_ahead, cloud_ahead = self._weather_ahead(now, outdoor, cloud)

        def sun_at(t):
            irr = self.irradiance_at(t)
            if irr is not None:  # estimated sunshine (HRRR); else the forecast's cloud cover
                return hm.sun_from_irradiance(hm.sun_elevation(t, lat, lon), hm.sun_azimuth(t, lat, lon), *irr)
            return hm.sun_at(t, lat, lon, cloud_ahead(t))

        self.shadow.set_correct(settings.observer_on)
        self.shadow.observe(now, temps, actual, outdoor, sun_at(now))
        if self.shadow.due(now) and not self._shadow_busy:
            unit_entity = {u.id: u.entity for u in units}
            ignore = {r.id for r in rooms if r.sensor and r.sensor == unit_entity.get(r.unit)}
            fixed = {u.id: (actual[u.id]["mode"], actual[u.id]["setpoint"]) for u in units
                     if self.control(u.id) == THERMOSTAT and u.id in actual}
            self._shadow_busy = True
            try:
                await self.hass.async_add_executor_job(self.shadow.run, now, rooms, units, temps, actual, outdoor_ahead,
                                                       sun_at, settings, ignore, fixed, dict(self.drives))
            finally:
                self._shadow_busy = False

    def model_debug(self) -> dict:
        """Everything the card's Model page shows."""
        now = dt_util.now()
        s = core.settings_from(self.data.get("settings"))
        rooms = [_room_from_dict(r) for r in self.data["rooms"]]
        targets = {}
        for r in rooms:
            if not r.sensor:
                continue
            t = core.target_temp(r, now)
            bl, ab = core.room_band(core.room_priority(r, now), s)
            targets[r.id] = {"name": r.name, "unit": r.unit, "target": t, "band": [t - bl, t + ab],
                             "prio": core.room_priority(r, now)}
        today = {}
        for ud in self.data["units"]:
            cmd, cur = self.last.get(ud["id"]), self.hass.states.get(ud["entity"])
            today[ud["id"]] = {"name": ud.get("name") or ud["id"], "control": self.control(ud["id"]),
                               "mode": cmd.mode if cmd else None, "setpoint": cmd.setpoint if cmd else None,
                               "target": None if cmd is None or cmd.target is None else round(cmd.target, 1),
                               "actual_mode": cur.state if cur is not None else None,
                               "actual_setpoint": cur.attributes.get("temperature") if cur is not None else None}
        m = self.shadow.model
        ok, reasons = self.model_ready(strict=True)
        f = self.forecast
        fc = None if f is None else {"at": f["at"].isoformat(), "entity": f["entity"], "stale": now - f["at"] > timedelta(hours=3),
                                     "next": [[t.isoformat(), v, c] for t, v, c in f["points"] if now < t <= now + timedelta(hours=3)]}
        return {"now": now.isoformat(), "model": None if m is None else m.to_dict(), "age_h": self.model_age_h(),
                "forecast": fc, "outdoor_now": self._outdoor(s),
                "sunshine": None if self.irradiance_at(now) is None else
                dict(zip(("ghi", "dni", "dhi"), (round(x) for x in self.irradiance_at(now)))),
                "ready": {"ok": ok, "why": reasons},
                "fit": self.fit_state, "fits": self.fit_history[-30:], "shadow": self.shadow.debug(now), "ml": self.ml_state,
                "targets": targets, "today": today, "house_today": self.house.mode,
                "comfort": {k: getattr(s, k) for k in ("band_below", "band_above", "warm_weight", "aim_offset")}}

    async def _log(self, now, rooms, units, predicted, info):
        """One row per minute to the long-term log in /config/room_climate_data (see datalog.py)."""
        minute = now.replace(second=0, microsecond=0)
        if minute == self._logged_minute:
            return
        self._logged_minute = minute

        def num(v):
            try:
                return None if v is None else round(float(v), 2)
            except (TypeError, ValueError):
                return None

        row = {"t": minute.isoformat(), "house": info.get("mode") or "off", "reason": info.get("reason"),
               "cold": info.get("cold"), "hot": info.get("hot"), "rooms": {}, "units": {}, "extra": {}}
        ent = self._outdoor_entity()
        wx = self.hass.states.get(ent) if ent else None
        if wx is not None:
            a = wx.attributes
            row["weather"] = {"state": wx.state, "temp": num(a.get("temperature")), "humidity": num(a.get("humidity")),
                              "cloud": num(a.get("cloud_coverage")), "wind": num(a.get("wind_speed"))}
        row["outdoor"] = num(info.get("outdoor"))
        sun = self.hass.states.get("sun.sun")
        if sun is not None:
            row["sun"] = {"elev": num(sun.attributes.get("elevation")), "az": num(sun.attributes.get("azimuth"))}
            irr = self.irradiance_at(now)
            if irr is not None:  # W/m2: global horizontal, direct normal, diffuse (HRRR estimate)
                row["sun"].update(ghi=round(irr[0]), dni=round(irr[1]), dhi=round(irr[2]))
        for r in rooms:
            if not r.sensor:
                continue
            val, _ = self._f(r.sensor)  # raw: None while the sensor is out (the controller bridges short gaps)
            c, h = info.get("rooms", {}).get(r.id, (None, None))
            row["rooms"][r.id] = {"temp": num(val), "target": core.target_temp(r, now), "prio": core.room_priority(r, now),
                                  "pred": num(predicted.get(r.id)), "unit": r.unit, "cold": c, "hot": h}
        hs = core.house_error([r for r in rooms if r.sensor], {r.id: row["rooms"][r.id]["temp"] for r in rooms if r.id in row["rooms"]},
                              now, core.settings_from(self.data.get("settings")))
        if hs:
            row["score"] = hs["weighted_rms_F"]  # whole-house weighted tracking error
        extra_ids = set()
        for u in units:
            cur, cmd = self.hass.states.get(u.entity), self.last.get(u.id)
            a = cur.attributes if cur is not None else {}
            row["units"][u.id] = {"entity": u.entity, "control": self.control(u.id),
                                  "state": cur.state if cur is not None else None, "sp": num(a.get("temperature")),
                                  "action": a.get("hvac_action"), "fan": a.get("fan_mode"), "vane": self._vane(u.entity), "temp": num(self._unit_temp(u.entity)),
                                  "cmd": cmd.mode if cmd else None, "cmd_sp": cmd.setpoint if cmd else None,
                                  "cmd_fan": cmd.fan if cmd else None,
                                  "why": cmd.reason if cmd else None}
            extra_ids |= self._unit_sensors(u.entity)
        # Everything else the unit controllers report: signal, compressor Hz, fan stage, sub-mode, outdoor unit temp...
        for eid in sorted(extra_ids):
            st = self.hass.states.get(eid)
            if st is not None and st.state not in BAD_STATES:
                row["extra"][eid] = num(st.state) if num(st.state) is not None else st.state
        row["mpc"] = self.shadow.log_fields()  # what the model-based controller would do
        if getattr(self, "exp_block", None):
            b = self.exp_block
            row["exp"] = {"kind": b["kind"], "unit": b.get("unit"), "fan": b.get("fan"), "off": b.get("off") or []}
        f = self.forecast
        if f is not None and f["at"] != self._forecast_logged:  # each fetched forecast, once (next 6 h)
            self._forecast_logged = f["at"]
            row["forecast"] = {"at": f["at"].isoformat(), "entity": f["entity"],
                               "pts": [[t.isoformat(), num(v), num(c)] for t, v, c in f["points"] if t <= now + timedelta(hours=6)]}
        self.track_store.async_delay_save(self.shadow.to_state, 30)  # about once a minute, and on shutdown
        if minute.minute % 5 == 0:
            self.hist5.append(mpc.compact(row))
            cut = (now - timedelta(hours=HISTORY_H)).isoformat()
            while self.hist5 and self.hist5[0]["t"] < cut:
                self.hist5.popleft()
        path = self.hass.config.path("room_climate_data")
        try:
            await self.hass.async_add_executor_job(datalog.append, path, row, now)
        except OSError as e:
            _LOGGER.warning("long-term log write failed: %s", e)

    def _unit_sensors(self, climate_entity: str) -> set:
        """Sensor entities on the same device as a unit's climate entity (whatever the device calls them), plus any
        named after the climate entity."""
        cache = self.__dict__.setdefault("_sensor_cache", {})
        hit = cache.get(climate_entity)
        if hit is not None and time.monotonic() - hit[0] < 600:
            return hit[1]
        reg = er.async_get(self.hass)
        ent = reg.async_get(climate_entity)
        ids = set()
        if ent is not None and ent.device_id:
            ids = {e.entity_id for e in er.async_entries_for_device(reg, ent.device_id) if e.domain == "sensor"}
        prefix = "sensor." + climate_entity.split(".", 1)[1] + "_"
        ids |= {e for e in self.hass.states.async_entity_ids("sensor") if e.startswith(prefix)}
        cache[climate_entity] = (time.monotonic(), ids)
        return ids

    def _audit(self, unit: core.Unit, cmd: core.Command, ours: bool, now):
        """Log what we want the unit to do next to what it is doing. Logged only when the decision (mode, setpoint)
        or the unit's own mode/setpoint changes; reason wording and sensor ticks alone don't repeat the line."""
        cur = self.hass.states.get(unit.entity)
        if cur is None or cur.state == "unavailable":
            line, key = f"{unit.id}: unit {unit.entity} unavailable", ("unavailable",)
        else:
            a = cur.attributes
            actual_sp = a.get("temperature")
            want_sp = None if cmd.setpoint is None else self._stored(cmd.setpoint)
            diffs = []
            if cur.state != cmd.mode:
                diffs.append(f"mode want {cmd.mode} has {cur.state}")
            if want_sp is not None and (actual_sp is None or abs(actual_sp - want_sp) > 0.15):
                diffs.append(f"setpoint want {want_sp} has {actual_sp}")
            line = (f"{unit.id}: command {cmd.mode}/{want_sp} ({cmd.reason}) | actual hvac={cur.state} "
                    f"setpoint={actual_sp} action={a.get('hvac_action')} unit_temp={a.get('current_temperature')} | "
                    + ("MISMATCH " + "; ".join(diffs) if diffs else "in sync"))
            key = (cmd.mode, want_sp, cur.state, actual_sp)
        key = (ours, *key)
        if self._audit_seen.get(unit.id) != key:
            self._audit_seen[unit.id] = key
            ctl = self.control(unit.id)
            _LOGGER.info("%s %s", "MODEL" if ctl == MODEL else "CONTROL" if ours else "THERMOSTAT (would)", line)

    async def _send(self, unit: core.Unit, state: core.UnitState, cmd: core.Command):
        cur = self.hass.states.get(unit.entity)
        if cur is None or cur.state == "unavailable":
            return
        mode = "off" if cmd.mode == "off" else cmd.mode
        if cmd.setpoint is None and cmd.mode == state.mode:
            return  # nothing to say (e.g. holding with no sensors)
        if cur.state != mode:
            _LOGGER.info("SEND %s hvac_mode %s -> %s (%s)", unit.entity, cur.state, mode, cmd.reason)
            await self._call(unit, SERVICE_SET_HVAC_MODE, {ATTR_HVAC_MODE: mode})
        if cmd.setpoint is not None:
            sp = self._stored(cmd.setpoint)
            have = cur.attributes.get("temperature")
            # Send what the unit will store (0.5 C steps) and compare against that: comparing against the F we wanted
            # (64.0 vs the 64.4 it reads back) resent the setpoint on every update, several times a minute.
            if have is None or abs(float(have) - sp) > 0.15:
                _LOGGER.info("SEND %s temperature %s -> %s (%s)", unit.entity, have, sp, cmd.reason)
                await self._call(unit, SERVICE_SET_TEMPERATURE, {ATTR_TEMPERATURE: sp})
        fan_now = cur.attributes.get("fan_mode")
        if cmd.fan is not None and cmd.fan != fan_now and cmd.fan in (cur.attributes.get("fan_modes") or []):
            _LOGGER.info("SEND %s fan_mode %s -> %s (%s)", unit.entity, fan_now, cmd.fan, cmd.reason)
            await self._call(unit, SERVICE_SET_FAN_MODE, {ATTR_FAN_MODE: cmd.fan})
            self._fan_set.add(unit.id)
        if cmd.vane is not None:
            await self._set_vane(unit.entity, cmd.vane)

    @staticmethod
    def _vane_entity(climate_entity: str) -> str:
        return "select." + climate_entity.split(".", 1)[1] + "_vertical_vane"

    def _vane(self, climate_entity: str):
        st = self.hass.states.get(self._vane_entity(climate_entity))
        return None if st is None or st.state in BAD_STATES else st.state

    async def _set_vane(self, unit_entity: str, position: str):
        ent = self._vane_entity(unit_entity)
        st = self.hass.states.get(ent)
        if st is None or st.state == position or position not in (st.attributes.get("options") or []):
            return
        _LOGGER.info("SEND %s vane %s -> %s", ent, st.state, position)
        await self.hass.services.async_call("select", "select_option", {"entity_id": ent, "option": position}, blocking=True)

    async def _restore_vanes(self, exp: dict):
        for uid, pos in (exp.get("vane0") or {}).items():
            ud = next((u for u in self.data["units"] if u["id"] == uid), None)
            if ud is not None:
                await self._set_vane(ud["entity"], pos)

    def _stored(self, setpoint_f: float) -> float:
        """`setpoint_f` as the unit will store and report it, in HA's temperature unit."""
        return round(TemperatureConverter.convert(core.stored_setpoint_c(setpoint_f), C,
                                                  self.hass.config.units.temperature_unit), 1)

    async def _call(self, unit: core.Unit, service: str, data: dict):
        ctx = Context()  # tagged so the unit's resulting state change is recognized as ours, not a manual change
        self._ours = (self._ours + [ctx.id])[-50:]
        self._sent_at[unit.id] = dt_util.utcnow()
        await self.hass.services.async_call(CLIMATE, service, {"entity_id": unit.entity, **data}, blocking=True, context=ctx)
        self._sent_at[unit.id] = dt_util.utcnow()

    def _publish(self, now, rooms):
        for r in rooms:
            val, _ = self._f(r.sensor) if r.sensor else (None, None)
            self.hass.states.async_set(f"{DOMAIN}.{r.id}", "ok" if val is not None else "no_sensor", {
                "friendly_name": r.name, "temperature": val, "target": core.target_temp(r, now),
                "priority": core.room_priority(r, now), "unit": r.unit,
                "action": self.last.get(r.unit, core.Command("off", None)).mode})
        self._publish_debug(now, rooms)

    def _publish_debug(self, now, rooms):
        """Controller internals as sensor.room_climate_* entities so the recorder keeps their history for graphs."""
        if not getattr(self, "_tick_debug", None):
            return
        readings, predicted, info, settings = self._tick_debug
        unit_f = TemperatureConverter.convert  # internal values are F; HA here is F as well
        disp = self.hass.config.units.temperature_unit

        def temp(entity, name, value, icon="mdi:thermometer"):
            self.hass.states.async_set(entity, "unknown" if value is None else round(unit_f(value, F, disp), 2), {
                "friendly_name": name, "unit_of_measurement": disp, "device_class": "temperature",
                "state_class": "measurement", "icon": icon})

        def text(entity, name, value, icon, **attrs):
            self.hass.states.async_set(entity, value, {"friendly_name": name, "icon": icon, **attrs})

        for r in rooms:
            if r.unit is None or not r.sensor:
                continue
            p = f"sensor.{DOMAIN}_{r.id}"
            t = core.target_temp(r, now)
            got = readings.get(r.id)
            temp(f"{p}_temperature", f"{r.name} temperature", got[0] if got else None)
            temp(f"{p}_target", f"{r.name} target", t, "mdi:target")
            temp(f"{p}_band_low", f"{r.name} band low", t - settings.band_below, "mdi:arrow-collapse-down")
            temp(f"{p}_band_high", f"{r.name} band high", t + settings.band_above, "mdi:arrow-collapse-up")
            temp(f"{p}_predicted", f"{r.name} predicted ({settings.horizon_s // 60} min)", predicted.get(r.id), "mdi:chart-line")
            c, h = info.get("rooms", {}).get(r.id, (0.0, 0.0))
            self.hass.states.async_set(f"{p}_demand", round(h - c, 2), {
                "friendly_name": f"{r.name} demand (hot + / cold -)", "state_class": "measurement",
                "icon": "mdi:scale-balance", "cold": c, "hot": h})
        for ud in self.data["units"]:
            cmd = self.last.get(ud["id"])
            p = f"sensor.{DOMAIN}_{ud['id']}"
            name = ud.get("name") or ud["id"]
            temp(f"{p}_commanded_setpoint", f"{name} commanded setpoint", cmd.setpoint if cmd else None, "mdi:thermostat")
            text(f"{p}_wanted_mode", f"{name} wanted mode", cmd.mode if cmd else "off", "mdi:hvac",
                 reason=cmd.reason if cmd else "")
            cur = self.hass.states.get(ud["entity"])
            temp(f"{p}_actual_setpoint", f"{name} actual setpoint",
                 None if cur is None or cur.attributes.get("temperature") is None else
                 unit_f(float(cur.attributes["temperature"]), disp, F), "mdi:thermostat-box")
            text(f"{p}_actual_mode", f"{name} actual mode", cur.state if cur else "unavailable", "mdi:hvac",
                 action=cur.attributes.get("hvac_action") if cur else None)
        text(f"sensor.{DOMAIN}_house_mode", "House mode", info.get("mode") or "off", "mdi:home-thermometer",
             reason=info.get("reason"))
        temp(f"sensor.{DOMAIN}_outdoor", "Outdoor (as seen by controller)", info.get("outdoor"), "mdi:home-export-outline")
        for k in ("cold", "hot"):
            self.hass.states.async_set(f"sensor.{DOMAIN}_house_{k}_demand", info.get(k) or 0.0, {
                "friendly_name": f"House {k} demand", "state_class": "measurement",
                "icon": "mdi:snowflake-thermometer" if k == "cold" else "mdi:sun-thermometer"})

    def snapshot(self) -> dict:
        now = dt_util.now()
        out = dict(self.data)
        if self.data.get("experiment"):
            b = experiment.current(self.data["experiment"], now)
            out["experiment"] = {**self.data["experiment"], "now": experiment.describe(b, self._unit_names()),
                                 "block_end": b["end"] if b else None}
        settings = core.settings_from(self.data.get("settings"))
        rooms = [_room_from_dict(r) for r in self.data["rooms"]]
        readings = {}
        for r in rooms:
            if r.sensor:
                val, ts = self._f(r.sensor)
                if val is not None:
                    readings[r.id] = (val, ts)
        out["now"] = now.isoformat()
        out["settings_effective"] = dataclasses.asdict(settings)
        out["units_live"] = {
            ud["id"]: core.diagnose(_unit_from_dict(ud), rooms, readings, self._unit_temp(ud["entity"]),
                                    self.states.get(ud["id"], core.UnitState()),
                                    self.last.get(ud["id"]), now, settings)
            for ud in self.data["units"]}
        for ud in self.data["units"]:  # what each unit is really doing, next to what we would command
            cur = self.hass.states.get(ud["entity"])
            out["units_live"][ud["id"]]["control"] = ud.get("control", THERMOSTAT)
            out["units_live"][ud["id"]]["actual"] = None if cur is None else {
                "state": cur.state, "setpoint": cur.attributes.get("temperature"),
                "action": cur.attributes.get("hvac_action"), "temp": self._unit_temp(ud["entity"])}
        out["house"] = self.house_info
        ok, reasons = self.model_ready(strict=True)
        out["model_ready"] = {"ok": ok, "why": reasons}
        out["live"] = {}
        for rd in self.data["rooms"]:
            r = _room_from_dict(rd)
            val, _ = self._f(r.sensor) if r.sensor else (None, None)
            c = self.last.get(r.unit)
            w = core.room_priority(r, now)
            out["live"][r.id] = {"temp": val, "target": core.target_temp(r, now), "priority": w,
                                 "allowable": round(core.allowable_error(w, core.settings_from(self.data.get("settings")).allowable_ref_f), 2),
                                 "mode": c.mode if c else "off", "reason": c.reason if c else ""}
        try:  # the whole-house tracking error, and each room's share of it
            rooms_now = [_room_from_dict(rd) for rd in self.data["rooms"]]
            out["score"] = core.house_error(rooms_now, {r.id: out["live"][r.id]["temp"] for r in rooms_now}, now,
                                            core.settings_from(self.data.get("settings")))
        except Exception:  # noqa: BLE001 - a display extra must never break the snapshot
            out["score"] = None
        return out


async def async_setup(hass: HomeAssistant, config) -> bool:
    mgr = Manager(hass)
    await mgr.async_load()
    hass.data[DOMAIN] = mgr

    from homeassistant.components.http import StaticPathConfig
    www = hass.config.path("custom_components/room_climate/www")
    await hass.http.async_register_static_paths([StaticPathConfig("/room_climate_static", www, False)])
    # Load the card on every dashboard automatically; the version query busts the browser cache on updates.
    manifest = await hass.async_add_executor_job(
        lambda: json.loads((pathlib.Path(__file__).parent / "manifest.json").read_text()))
    add_extra_js_url(hass, f"/room_climate_static/room-map-card.js?v={manifest['version']}")
    add_extra_js_url(hass, f"/room_climate_static/room-climate-probe.js?v={manifest['version']}")

    @websocket_api.websocket_command({"type": f"{DOMAIN}/get"})
    @websocket_api.async_response
    async def ws_get(hass, connection, msg):
        connection.send_result(msg["id"], mgr.snapshot())

    @websocket_api.websocket_command({
        "type": f"{DOMAIN}/set_room", vol.Required("room_id"): str,
        vol.Optional("priority"): vol.All(vol.Coerce(float), vol.Range(min=core.MIN_PRIORITY, max=core.MAX_PRIORITY)),
        vol.Optional("default_temp"): vol.Coerce(float),
        vol.Optional("schedule"): list,
        vol.Optional("priority_schedule"): [vol.Schema({
            vol.Required("days"): [vol.All(int, vol.Range(min=0, max=6))],
            vol.Required("start"): vol.Match(r"^\d{2}:\d{2}$"),
            vol.Required("priority"): vol.All(vol.Coerce(float), vol.Range(min=core.MIN_PRIORITY, max=core.MAX_PRIORITY))})],
        vol.Optional("hold_temp"): vol.Any(None, vol.Coerce(float)),
        vol.Optional("hold_hours"): vol.Any(None, vol.Coerce(float)),
        vol.Optional("sensor"): vol.Any(None, vol.Match(r"^(sensor|climate)\.[a-z0-9_]+$")),
        vol.Optional("fit_since"): vol.Any(None, str),  # learn from this room's readings from then on (sensor moved)
    })
    @websocket_api.async_response
    async def ws_set_room(hass, connection, msg):
        room = next((r for r in mgr.data["rooms"] if r["id"] == msg["room_id"]), None)
        if room is None:
            connection.send_error(msg["id"], "not_found", "unknown room")
            return
        if msg.get("sensor") and hass.states.get(msg["sensor"]) is None:
            connection.send_error(msg["id"], "not_found", f"no entity {msg['sensor']}")
            return
        if msg.get("fit_since") and dt_util.parse_datetime(msg["fit_since"]) is None:
            connection.send_error(msg["id"], "invalid", "fit_since must be an ISO date and time")
            return
        if "sensor" in msg and msg["sensor"] != room.get("sensor"):
            _LOGGER.info("%s sensor %s -> %s", room["id"], room.get("sensor"), msg["sensor"])
            room["fit_since"] = dt_util.now().isoformat()  # a different sensor: learn from its readings only
            mgr.last_good.pop(room["id"], None)  # a different sensor: don't bridge from the old one's reading
            mgr.hist.pop(room["id"], None)
        for k in ("priority", "default_temp", "schedule", "priority_schedule", "hold_temp", "sensor", "fit_since"):
            if k in msg:
                room[k] = msg[k]
        if "hold_hours" in msg:
            h = msg["hold_hours"]
            room["hold_until"] = None if not h else (dt_util.utcnow() + timedelta(hours=h)).isoformat()  # real hours, even across DST
        if msg.get("hold_temp", 1) is None:
            room.pop("hold_until", None)
        await mgr.async_save()
        await mgr.async_tick()
        connection.send_result(msg["id"], mgr.snapshot())

    @websocket_api.websocket_command({
        "type": f"{DOMAIN}/log_client", vol.Required("message"): str,
        vol.Optional("stack", default=""): str, vol.Optional("ua", default=""): str, vol.Optional("where", default=""): str,
    })
    @websocket_api.async_response
    async def ws_log_client(hass, connection, msg):
        # Frontend failures (old tablets, blank cards) never reach the server otherwise.
        _LOGGER.warning("Card error from client [%s] %s | %s | %s", msg["where"][:80], msg["message"][:400], msg["ua"][:200], msg["stack"][:800])
        connection.send_result(msg["id"])

    @websocket_api.websocket_command({
        "type": f"{DOMAIN}/set_control", vol.Required("unit_id"): str,
        vol.Required("control"): vol.In([ROOM_CLIMATE, THERMOSTAT, MODEL]),
    })
    @websocket_api.async_response
    async def ws_set_control(hass, connection, msg):
        if not any(u["id"] == msg["unit_id"] for u in mgr.data["units"]):
            connection.send_error(msg["id"], "not_found", "unknown unit")
            return
        try:
            await mgr.async_set_control(msg["unit_id"], msg["control"], "set in the Room Climate card")
        except ValueError as e:
            connection.send_error(msg["id"], "not_ready", str(e))
            return
        connection.send_result(msg["id"], mgr.snapshot())

    @websocket_api.websocket_command({"type": f"{DOMAIN}/experiment_start",
                                      vol.Required("hours"): vol.All(vol.Coerce(float), vol.Range(min=experiment.MIN_H, max=experiment.MAX_H))})
    @websocket_api.async_response
    async def ws_experiment_start(hass, connection, msg):
        try:
            await mgr.async_start_experiment(msg["hours"])
        except ValueError as e:
            connection.send_error(msg["id"], "not_ready", str(e))
            return
        connection.send_result(msg["id"], mgr.snapshot())

    @websocket_api.websocket_command({"type": f"{DOMAIN}/experiment_stop"})
    @websocket_api.async_response
    async def ws_experiment_stop(hass, connection, msg):
        await mgr.async_stop_experiment()
        connection.send_result(msg["id"], mgr.snapshot())

    @websocket_api.websocket_command({"type": f"{DOMAIN}/set_settings", vol.Required("settings"): dict})
    @websocket_api.async_response
    async def ws_set_settings(hass, connection, msg):
        known = {f.name for f in dataclasses.fields(core.Settings)}
        new = msg["settings"]
        bad = [k for k, v in new.items() if k not in known or not isinstance(v, (int, float, bool, str, type(None)))]
        if bad:
            connection.send_error(msg["id"], "invalid", f"unknown or invalid settings: {', '.join(bad)}")
            return
        mgr.data.setdefault("settings", {}).update(new)
        _LOGGER.info("settings changed: %s", new)
        await mgr.async_save()
        await mgr.async_tick()
        connection.send_result(msg["id"], mgr.snapshot())

    @websocket_api.websocket_command({"type": f"{DOMAIN}/model"})
    @websocket_api.async_response
    async def ws_model(hass, connection, msg):
        connection.send_result(msg["id"], mgr.model_debug())

    @websocket_api.websocket_command({"type": f"{DOMAIN}/model_history",
                                      vol.Optional("hours", default=24): vol.All(vol.Coerce(float), vol.Range(min=1, max=HISTORY_H))})
    @websocket_api.async_response
    async def ws_model_history(hass, connection, msg):
        connection.send_result(msg["id"], mgr.model_history(msg["hours"]))

    @websocket_api.websocket_command({"type": f"{DOMAIN}/room_history", vol.Required("room_id"): str,
                                      vol.Optional("hours", default=24): vol.All(vol.Coerce(float), vol.Range(min=1, max=HISTORY_H))})
    @websocket_api.async_response
    async def ws_room_history(hass, connection, msg):
        connection.send_result(msg["id"], mgr.room_history(msg["room_id"], msg["hours"]))

    @websocket_api.websocket_command({"type": f"{DOMAIN}/reload_ml"})
    @websocket_api.async_response
    async def ws_reload_ml(hass, connection, msg):
        """Load the ML correction again (after a new room_climate_ml_external.json was copied in)."""
        await mgr._load_ml()
        connection.send_result(msg["id"], mgr.ml_state)

    @websocket_api.websocket_command({"type": f"{DOMAIN}/fit_model"})
    @websocket_api.async_response
    async def ws_fit_model(hass, connection, msg):
        hass.async_create_task(mgr.async_fit())
        connection.send_result(msg["id"], {"started": True})

    for fn in (ws_get, ws_set_room, ws_set_control, ws_set_settings, ws_log_client, ws_model, ws_model_history, ws_room_history, ws_fit_model, ws_reload_ml,
               ws_experiment_start, ws_experiment_stop):
        websocket_api.async_register_command(hass, fn)

    unit_entities = [u["entity"] for u in mgr.data["units"]]
    if unit_entities:
        async_track_state_change_event(hass, unit_entities, mgr.async_unit_changed)
    # One "Room Climate / Thermostat" control select per unit, for placing next to its thermostat card.
    hass.async_create_task(discovery.async_load_platform(hass, "select", DOMAIN, {}, config))

    async def _on_sensor(event):
        await mgr.async_tick()

    sensors = [r["sensor"] for r in mgr.data["rooms"] if r.get("sensor")]
    if sensors:
        # Rate limiting lives in core (min run/off times), so reacting to every update is safe.
        async_track_state_change_event(hass, sensors, _on_sensor)
    async_track_time_interval(hass, mgr.async_tick, INTERVAL)
    hass.async_create_task(mgr.async_load_history())
    # The hourly forecast for planning: soon after startup, then every hour.
    async_call_later(hass, 30, mgr.async_update_forecast)
    async_call_later(hass, 20, mgr.async_update_sun)
    async_track_time_interval(hass, mgr.async_update_sun, timedelta(minutes=15))
    async_track_time_interval(hass, mgr.async_update_forecast, timedelta(hours=1))
    # Refit the thermal model nightly, and soon after startup if there is none or it's more than a day old.
    async_track_time_change(hass, mgr.async_fit, hour=3, minute=40, second=0)
    # Then the ML correction, once the physics fit (13 min) is done; its rows are flushed to disk hourly.
    async_track_time_change(hass, mgr.async_train_ml, hour=3, minute=58, second=0)
    async_track_time_interval(hass, mgr.async_flush_ml, timedelta(hours=1))
    age = mgr.model_age_h()
    if age is None or age > 26:
        async_call_later(hass, 180, mgr.async_fit)
    return True
