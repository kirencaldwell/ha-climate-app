"""Room Climate: hold each room at its own target using room sensors to drive Mitsubishi units."""
from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from datetime import datetime, timedelta

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.frontend import add_extra_js_url
from homeassistant.components.climate import ATTR_HVAC_MODE, DOMAIN as CLIMATE, SERVICE_SET_HVAC_MODE, SERVICE_SET_TEMPERATURE
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import TemperatureConverter

from . import core

DOMAIN = "room_climate"
_LOGGER = logging.getLogger(__name__)
STORE_VERSION = 1
INTERVAL = timedelta(seconds=60)
F, C = UnitOfTemperature.FAHRENHEIT, UnitOfTemperature.CELSIUS

CONFIG_SCHEMA = vol.Schema({DOMAIN: vol.Schema({vol.Optional("enabled", default=True): bool})}, extra=vol.ALLOW_EXTRA)


def _room_from_dict(d: dict) -> core.Room:
    return core.Room(
        id=d["id"], name=d.get("name", d["id"]), unit=d.get("unit"), sensor=d.get("sensor"),
        priority=int(d.get("priority", 3)), default_temp=float(d.get("default_temp", 68)),
        schedule=[core.ScheduleEntry(list(e["days"]), e["start"], float(e["temp"])) for e in d.get("schedule", [])],
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
        self.data: dict = {"rooms": [], "units": [], "settings": {}, "enabled": True, "floorplan": None}
        self.states: dict[str, core.UnitState] = {}
        self.last: dict[str, core.Command] = {}
        self._unsub = []

    async def async_load(self):
        loaded = await self.store.async_load()
        if loaded is None:  # first run: seed from the bundled example layout
            seed = pathlib.Path(__file__).parent / "default_config.json"
            loaded = await self.hass.async_add_executor_job(lambda: json.loads(seed.read_text()))
            loaded.pop("_comment", None)
        self.data = loaded

    async def async_save(self):
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
        # last_reported moves on every report; last_updated only when the value changes, which would make a
        # steady sensor look stale.
        return TemperatureConverter.convert(val, unit, F), getattr(st, "last_reported", None) or st.last_updated

    def _unit_temp(self, entity_id: str):
        st = self.hass.states.get(entity_id)
        cur = st.attributes.get("current_temperature") if st else None
        return None if cur is None else TemperatureConverter.convert(float(cur), self.hass.config.units.temperature_unit, F)

    async def async_tick(self, _now=None):
        if not self.data.get("enabled", True):
            return
        now = dt_util.now()
        settings = core.Settings(**self.data.get("settings", {}))
        rooms = [_room_from_dict(r) for r in self.data["rooms"]]
        for ud in self.data["units"]:
            unit = _unit_from_dict(ud)
            mine = [r for r in rooms if r.unit == unit.id]
            readings = {}
            for r in mine:
                if r.sensor:
                    val, ts = self._f(r.sensor)
                    if val is not None:
                        readings[r.id] = (val, ts)
            state = self.states.setdefault(unit.id, core.UnitState())
            cmd = core.decide(unit, mine, readings, self._unit_temp(unit.entity), state, now, settings)
            self.last[unit.id] = cmd
            await self._send(unit, state, cmd)
            self.states[unit.id] = core.apply(state, cmd, now)
        self._publish(now, rooms)

    async def _send(self, unit: core.Unit, state: core.UnitState, cmd: core.Command):
        cur = self.hass.states.get(unit.entity)
        if cur is None or cur.state == "unavailable":
            return
        mode = "off" if cmd.mode == "off" else cmd.mode
        if cmd.setpoint is None and cmd.mode == state.mode:
            return  # nothing to say (e.g. holding with no sensors)
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
                "priority": r.priority, "unit": r.unit,
                "action": self.last.get(r.unit, core.Command("off", None)).mode})

    def snapshot(self) -> dict:
        now = dt_util.now()
        out = dict(self.data)
        settings = core.Settings(**self.data.get("settings", {}))
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
                                    self.states.get(ud["id"], core.UnitState()), self.last.get(ud["id"]), now, settings)
            for ud in self.data["units"]}
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

    @websocket_api.websocket_command({
        "type": f"{DOMAIN}/log_client", vol.Required("message"): str,
        vol.Optional("stack", default=""): str, vol.Optional("ua", default=""): str, vol.Optional("where", default=""): str,
    })
    @websocket_api.async_response
    async def ws_log_client(hass, connection, msg):
        # Frontend failures (old tablets, blank cards) never reach the server otherwise.
        _LOGGER.warning("Card error from client [%s] %s | %s | %s", msg["where"][:80], msg["message"][:400], msg["ua"][:200], msg["stack"][:800])
        connection.send_result(msg["id"])

    for fn in (ws_get, ws_set_room, ws_log_client):
        websocket_api.async_register_command(hass, fn)

    async def _on_sensor(event):
        await mgr.async_tick()

    sensors = [r["sensor"] for r in mgr.data["rooms"] if r.get("sensor")]
    if sensors:
        # Rate limiting lives in core (min run/off times), so reacting to every update is safe.
        async_track_state_change_event(hass, sensors, _on_sensor)
    async_track_time_interval(hass, mgr.async_tick, INTERVAL)
    return True
