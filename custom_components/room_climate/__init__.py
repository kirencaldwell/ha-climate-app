"""Room Climate: hold each room at its own target using room sensors to drive Mitsubishi units."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.climate import ATTR_HVAC_MODE, DOMAIN as CLIMATE, SERVICE_SET_HVAC_MODE, SERVICE_SET_TEMPERATURE
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import TemperatureConverter

from . import core, predict

DOMAIN = "room_climate"
_LOGGER = logging.getLogger(__name__)
STORE_VERSION = 1
INTERVAL = timedelta(seconds=60)
FORECAST_TTL = timedelta(minutes=30)
SAVE_EVERY = timedelta(minutes=15)
F, C = UnitOfTemperature.FAHRENHEIT, UnitOfTemperature.CELSIUS

CONFIG_SCHEMA = vol.Schema({DOMAIN: vol.Schema({vol.Optional("enabled", default=True): bool})}, extra=vol.ALLOW_EXTRA)


def _room_from_dict(d: dict) -> core.Room:
    return core.Room(
        id=d["id"], name=d.get("name", d["id"]), unit=d["unit"], sensor=d.get("sensor"),
        priority=int(d.get("priority", 3)), default_temp=float(d.get("default_temp", 68)),
        schedule=[core.ScheduleEntry(list(e["days"]), e["start"], float(e["temp"])) for e in d.get("schedule", [])],
        hold_temp=d.get("hold_temp"),
        hold_until=dt_util.parse_datetime(d["hold_until"]) if d.get("hold_until") else None,
        floor=int(d.get("floor", 1)), shape=d.get("shape", {}), windows=d.get("windows", []))


def _unit_from_dict(d: dict) -> core.Unit:
    return core.Unit(id=d["id"], entity=d["entity"], min_setpoint=float(d.get("min_setpoint", 61)),
                     max_setpoint=float(d.get("max_setpoint", 88)))


class Manager:
    """Owns persisted config, runs the control loop, serves the card."""

    def __init__(self, hass: HomeAssistant):
        self.hass = hass
        self.store = Store(hass, STORE_VERSION, DOMAIN)
        self.data: dict = {"rooms": [], "units": [], "settings": {}, "enabled": True, "floorplan": None}
        self.sys_state = core.SystemState()
        self.models: dict[str, predict.RoomModel] = {}
        self.preds: dict[str, float] = {}
        self._forecast_at = None
        self._forecast: list[tuple] = []  # (datetime, temp F, cloud 0..1)
        self._saved_at = None
        self.last: dict[str, core.Command] = {}
        self._unsub = []

    async def async_load(self):
        loaded = await self.store.async_load()
        if loaded is None:  # first run: seed from the bundled example layout
            import json, pathlib
            seed = pathlib.Path(__file__).parent / "default_config.json"
            loaded = await self.hass.async_add_executor_job(lambda: json.loads(seed.read_text()))
            loaded.pop("_comment", None)
        self.data = loaded
        for rid, d in (loaded.get("models") or {}).items():
            try:
                self.models[rid] = predict.RoomModel.from_dict(d)
            except (KeyError, TypeError, ValueError):
                _LOGGER.warning("Discarding unreadable model for %s", rid)

    async def async_save(self):
        self.data["models"] = {rid: m.to_dict() for rid, m in self.models.items()}
        await self.store.async_save(self.data)

    def _f(self, entity_id: str):
        """Entity state as degrees F (None if unavailable)."""
        st = self.hass.states.get(entity_id)
        if st is None or st.state in ("unknown", "unavailable"):
            return None, None
        try:
            val = float(st.state)
        except ValueError:
            return None, None
        unit = st.attributes.get("unit_of_measurement") or self.hass.config.units.temperature_unit
        return TemperatureConverter.convert(val, unit, F), st.last_updated

    def _unit_temp(self, entity_id: str):
        st = self.hass.states.get(entity_id)
        cur = st.attributes.get("current_temperature") if st else None
        return None if cur is None else TemperatureConverter.convert(float(cur), self.hass.config.units.temperature_unit, F)

    def _site(self) -> predict.Site:
        """House location: HA's configured location (set it to the home address), or data["site"]."""
        o = self.data.get("site") or {}
        cfg = self.hass.config
        off = dt_util.now().utcoffset()
        return predict.Site(float(o.get("lat", cfg.latitude)), float(o.get("lon", cfg.longitude)),
                            (off.total_seconds() / 3600) if off else 0.0)

    async def _refresh_forecast(self, now):
        """Hourly forecast from the configured weather entity (HA `weather.get_forecasts`)."""
        ent = self.data.get("weather_entity")
        if not ent or (self._forecast_at and now - self._forecast_at < FORECAST_TTL):
            return
        self._forecast_at = now
        try:
            resp = await self.hass.services.async_call("weather", "get_forecasts", {"entity_id": ent, "type": "hourly"},
                                                       blocking=True, return_response=True)
            st = self.hass.states.get(ent)
            unit = (st.attributes.get("temperature_unit") if st else None) or self.hass.config.units.temperature_unit
            pts = []
            for f in resp[ent]["forecast"]:
                when = dt_util.parse_datetime(f["datetime"]) if isinstance(f["datetime"], str) else f["datetime"]
                if f.get("temperature") is None or when is None:
                    continue
                cloud = f.get("cloud_coverage")
                pts.append((dt_util.as_local(when), TemperatureConverter.convert(float(f["temperature"]), unit, F),
                            None if cloud is None else float(cloud) / 100.0))
            self._forecast = pts
        except Exception as err:  # weather is optional: log and fall back to the current outdoor temperature
            _LOGGER.warning("Weather forecast unavailable (%s); predictions use current outdoor temperature", err)

    def _outdoor_now(self):
        ent = self.data.get("weather_entity")
        st = self.hass.states.get(ent) if ent else None
        if st is None or st.attributes.get("temperature") is None:
            return None
        unit = st.attributes.get("temperature_unit") or self.hass.config.units.temperature_unit
        return TemperatureConverter.convert(float(st.attributes["temperature"]), unit, F)

    def _learn_and_predict(self, now, rooms, readings):
        """Update each room's model from this tick's data; return idle-drift predictions for ready models."""
        out_now = self._outdoor_now()
        if out_now is None:
            self.preds = {}  # no outdoor data: no learning, no prediction -> reactive control
            return None
        site = self._site()
        outdoor_at = predict.interpolator([(t, v) for t, v, _ in self._forecast], out_now)
        cloud_pts = [(t, c) for t, _, c in self._forecast if c is not None]
        cloud_at = predict.interpolator(cloud_pts, 0.0) if cloud_pts else None
        lead = core.Settings(**self.data.get("settings", {})).precompensate_min
        preds = {}
        for r in rooms:
            if r.id not in readings:
                continue
            temp = readings[r.id][0]
            ust = self.hass.states.get(next((u["entity"] for u in self.data["units"] if u["id"] == r.unit), ""))
            action = ust.attributes.get("hvac_action") if ust else None
            running = 1 if action == "heating" else -1 if action == "cooling" else 0
            cloud = cloud_at(now) if cloud_at else 0.0
            model = self.models.setdefault(r.id, predict.RoomModel())
            model.update(now, temp, out_now, running, predict.sun_factor(r.windows, site, now, cloud))
            if model.ready:
                preds[r.id] = predict.predict_temp(model, temp, now, lead, outdoor_at, r.windows, site, cloud_at)
        self.preds = preds
        return preds

    async def async_tick(self, _now=None):
        if not self.data.get("enabled", True):
            return
        now = dt_util.now()
        settings = core.Settings(**self.data.get("settings", {}))
        rooms = [_room_from_dict(r) for r in self.data["rooms"]]
        units = [_unit_from_dict(u) for u in self.data["units"]]
        readings = {}
        for r in rooms:
            if r.sensor:
                val, ts = self._f(r.sensor)
                if val is not None:
                    readings[r.id] = (val, ts)
        await self._refresh_forecast(now)
        preds = self._learn_and_predict(now, rooms, readings)
        unit_temps = {u.id: self._unit_temp(u.entity) for u in units}
        # One outdoor unit: all indoor units must share a mode, so decide them together.
        cmds, self.sys_state = core.decide_system(units, rooms, readings, unit_temps, self.sys_state, now, settings, preds)
        self.last = cmds
        for u in units:
            await self._send(u, cmds[u.id])
        self._publish(now, rooms)
        if self._saved_at is None or now - self._saved_at >= SAVE_EVERY:
            self._saved_at = now
            await self.async_save()

    async def _send(self, unit: core.Unit, cmd: core.Command):
        cur = self.hass.states.get(unit.entity)
        if cur is None or cur.state == "unavailable":
            return
        mode = cmd.hvac_mode or cmd.mode
        if cmd.setpoint is None and cmd.reason.startswith("no fresh sensor"):
            return  # no data: leave the unit alone
        if cur.state != mode:
            await self.hass.services.async_call(CLIMATE, SERVICE_SET_HVAC_MODE,
                                                {"entity_id": unit.entity, ATTR_HVAC_MODE: mode}, blocking=True)
        if cmd.setpoint is not None:
            sp = TemperatureConverter.convert(cmd.setpoint, F, self.hass.config.units.temperature_unit)
            if cur.attributes.get("temperature") != sp:
                await self.hass.services.async_call(CLIMATE, SERVICE_SET_TEMPERATURE,
                                                    {"entity_id": unit.entity, ATTR_TEMPERATURE: round(sp, 1)}, blocking=True)

    def _publish(self, now, rooms):
        for r in rooms:
            val, _ = self._f(r.sensor) if r.sensor else (None, None)
            self.hass.states.async_set(f"{DOMAIN}.{r.id}", "ok" if val is not None else "no_sensor", {
                "friendly_name": r.name, "temperature": val, "target": core.target_temp(r, now),
                "priority": r.priority, "unit": r.unit, "predicted": self.preds.get(r.id),
                "model_ready": bool(self.models.get(r.id) and self.models[r.id].ready),
                "action": self.last.get(r.unit, core.Command("off", None)).mode})

    def snapshot(self) -> dict:
        now = dt_util.now()
        out = {k: v for k, v in self.data.items() if k != "models"}  # models are big and not for the card
        out["live"] = {}
        for rd in self.data["rooms"]:
            r = _room_from_dict(rd)
            val, _ = self._f(r.sensor) if r.sensor else (None, None)
            c = self.last.get(r.unit)
            out["live"][r.id] = {"temp": val, "target": core.target_temp(r, now),
                                 "mode": c.mode if c else "off", "reason": c.reason if c else ""}
        return out


async def async_setup(hass: HomeAssistant, config) -> bool:
    mgr = Manager(hass)
    await mgr.async_load()
    hass.data[DOMAIN] = mgr

    from homeassistant.components.http import StaticPathConfig
    www = hass.config.path("custom_components/room_climate/www")
    await hass.http.async_register_static_paths([StaticPathConfig("/room_climate_static", www, False)])

    @websocket_api.websocket_command({"type": f"{DOMAIN}/get"})
    @websocket_api.async_response
    async def ws_get(hass, connection, msg):
        connection.send_result(msg["id"], mgr.snapshot())

    @websocket_api.websocket_command({
        "type": f"{DOMAIN}/set_room", vol.Required("room_id"): str,
        vol.Optional("priority"): vol.All(int, vol.Range(min=1, max=5)),
        vol.Optional("default_temp"): vol.Coerce(float),
        vol.Optional("schedule"): list,
        vol.Optional("hold_temp"): vol.Any(None, vol.Coerce(float)),
        vol.Optional("hold_hours"): vol.Any(None, vol.Coerce(float)),
    })
    @websocket_api.async_response
    async def ws_set_room(hass, connection, msg):
        room = next((r for r in mgr.data["rooms"] if r["id"] == msg["room_id"]), None)
        if room is None:
            connection.send_error(msg["id"], "not_found", "unknown room")
            return
        for k in ("priority", "default_temp", "schedule", "hold_temp"):
            if k in msg:
                room[k] = msg[k]
        if "hold_hours" in msg:
            h = msg["hold_hours"]
            room["hold_until"] = None if not h else (dt_util.now() + timedelta(hours=h)).isoformat()
        if msg.get("hold_temp", 1) is None:
            room.pop("hold_until", None)
        await mgr.async_save()
        await mgr.async_tick()
        connection.send_result(msg["id"], mgr.snapshot())

    for fn in (ws_get, ws_set_room):
        websocket_api.async_register_command(hass, fn)

    async def _on_sensor(event):
        await mgr.async_tick()

    sensors = [r["sensor"] for r in mgr.data["rooms"] if r.get("sensor")]
    if sensors:
        # Rate limiting lives in core (min run/off times), so reacting to every update is safe.
        async_track_state_change_event(hass, sensors, _on_sensor)
    async_track_time_interval(hass, mgr.async_tick, INTERVAL)
    return True
