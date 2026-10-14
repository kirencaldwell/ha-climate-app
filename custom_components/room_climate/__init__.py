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
        self.shadow: dict[str, core.UnitState] = {}  # decision state while disabled (dry run), kept apart from the real one
        self.house = core.HouseState()
        self.house_shadow = core.HouseState()
        self.house_info: dict = {}
        self.hist: dict[str, list] = {}  # room id -> [(time, temp F)] for trend prediction
        self.prev_busy: set = set()
        self.sent: dict[bool, dict] = {True: {}, False: {}}  # live/dry-run -> unit id -> (last setpoint command, when)
        self._audit_seen: dict[str, tuple] = {}
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
        if entity_id.startswith("climate."):
            # A room with no separate sensor can use its unit's own reading (the climate's current temperature).
            cur = st.attributes.get("current_temperature")
            if cur is None:
                return None, None
            return (TemperatureConverter.convert(float(cur), self.hass.config.units.temperature_unit, F),
                    getattr(st, "last_reported", None) or st.last_updated)
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
        # Disabled means dry run: decide exactly as normal and log it against what the unit is really doing,
        # but send nothing.
        enabled = self.data.get("enabled", True)
        states = self.states if enabled else self.shadow
        now = dt_util.now()
        settings = core.Settings(**self.data.get("settings", {}))
        rooms = [_room_from_dict(r) for r in self.data["rooms"]]
        units = [_unit_from_dict(ud) for ud in self.data["units"]]
        readings, predicted = {}, {}
        for r in rooms:
            if not r.sensor:
                continue
            val, ts = self._f(r.sensor)
            if val is None:
                continue
            readings[r.id] = (val, ts)
            h = self.hist.setdefault(r.id, [])
            h.append((now, val))
            del h[:max(0, next((k for k, (t, _) in enumerate(h) if (now - t).total_seconds() <= settings.history_s), len(h)))]
            predicted[r.id] = core.predict(h, now, settings)
        manual, busy = {}, set()
        for u in units:
            cur = self.hass.states.get(u.entity)
            if cur is not None and cur.state in core.MANUAL_MODES:
                manual[u.id] = cur.state
            if cur is not None and cur.attributes.get("hvac_action") in ("heating", "cooling"):
                busy.add(u.id)
        for u in units:  # a compressor that started/stopped on its own makes the earlier trend misleading
            if (u.id in busy) != (u.id in self.prev_busy):
                for r in rooms:
                    if r.unit == u.id and self.hist.get(r.id):
                        self.hist[r.id] = self.hist[r.id][-1:]
                        predicted[r.id] = None
        self.prev_busy = busy
        house = self.house if enabled else self.house_shadow
        cmds, new_house, info = core.control_step(
            units, rooms, readings, predicted, {u.id: self._unit_temp(u.entity) for u in units},
            states, house, now, settings, manual, busy)
        if new_house.mode != house.mode:
            _LOGGER.info("%s house mode %s -> %s (%s)", "LIVE" if enabled else "DRY-RUN", house.mode, new_house.mode, info["reason"])
        if enabled:
            self.house = new_house
        else:
            self.house_shadow = new_house
        self.house_info = {**info, "since": new_house.since.isoformat() if new_house.since else None,
                           "predicted": {k: None if v is None else round(v, 1) for k, v in predicted.items()}}
        self._tick_debug = (readings, predicted, info, settings)
        sent = self.sent[enabled]
        for u in units:
            # Keep setpoints steady (inverter efficiency): small or too-frequent changes hold the previous one.
            cmd, changed = core.stabilize(cmds[u.id], sent.get(u.id), now, settings)
            if changed and cmd.setpoint is not None:
                sent[u.id] = (cmd, now)
            state = states.setdefault(u.id, core.UnitState())
            self.last[u.id] = cmd
            if enabled:
                await self._send(u, state, cmd)
            self._audit(u, cmd, enabled, now)
            states[u.id] = core.apply(state, cmd, now)
            if core.regime_changed(state, states[u.id]):
                for r in rooms:
                    if r.unit == u.id and self.hist.get(r.id):
                        self.hist[r.id] = self.hist[r.id][-1:]
        self._publish(now, rooms)

    def _audit(self, unit: core.Unit, cmd: core.Command, enabled: bool, now):
        """Log what we want the unit to do next to what it is doing. Logged when anything changes, else every 30 min."""
        cur = self.hass.states.get(unit.entity)
        if cur is None or cur.state == "unavailable":
            line, key = f"{unit.id}: unit {unit.entity} unavailable", ("unavailable",)
        else:
            a = cur.attributes
            actual_sp = a.get("temperature")
            want_sp = None if cmd.setpoint is None else round(
                TemperatureConverter.convert(cmd.setpoint, F, self.hass.config.units.temperature_unit), 1)
            diffs = []
            if cur.state != cmd.mode:
                diffs.append(f"mode want {cmd.mode} has {cur.state}")
            if want_sp is not None and (actual_sp is None or abs(actual_sp - want_sp) > 0.3):
                diffs.append(f"setpoint want {want_sp} has {actual_sp}")
            line = (f"{unit.id}: command {cmd.mode}/{want_sp} ({cmd.reason}) | actual hvac={cur.state} "
                    f"setpoint={actual_sp} action={a.get('hvac_action')} unit_temp={a.get('current_temperature')} | "
                    + ("MISMATCH " + "; ".join(diffs) if diffs else "in sync"))
            key = (cmd.mode, want_sp, cmd.reason, cur.state, actual_sp, bool(diffs))
        seen = self._audit_seen.get(unit.id)
        if seen is None or seen[0] != key or (now - seen[1]).total_seconds() > 1800:
            self._audit_seen[unit.id] = (key, now)
            _LOGGER.info("%s %s", "LIVE" if enabled else "DRY-RUN", line)

    async def _send(self, unit: core.Unit, state: core.UnitState, cmd: core.Command):
        cur = self.hass.states.get(unit.entity)
        if cur is None or cur.state == "unavailable":
            return
        mode = "off" if cmd.mode == "off" else cmd.mode
        if cmd.setpoint is None and cmd.mode == state.mode:
            return  # nothing to say (e.g. holding with no sensors)
        if cur.state != mode:
            _LOGGER.info("SEND %s hvac_mode %s -> %s (%s)", unit.entity, cur.state, mode, cmd.reason)
            await self.hass.services.async_call(CLIMATE, SERVICE_SET_HVAC_MODE,
                                                {"entity_id": unit.entity, ATTR_HVAC_MODE: mode}, blocking=True)
        if cmd.setpoint is not None:
            sp = TemperatureConverter.convert(cmd.setpoint, F, self.hass.config.units.temperature_unit)
            if cur.attributes.get("temperature") != sp:
                _LOGGER.info("SEND %s temperature %s -> %s (%s)", unit.entity, cur.attributes.get("temperature"), round(sp, 1), cmd.reason)
                await self.hass.services.async_call(CLIMATE, SERVICE_SET_TEMPERATURE,
                                                    {"entity_id": unit.entity, ATTR_TEMPERATURE: round(sp, 1)}, blocking=True)

    def _publish(self, now, rooms):
        for r in rooms:
            val, _ = self._f(r.sensor) if r.sensor else (None, None)
            self.hass.states.async_set(f"{DOMAIN}.{r.id}", "ok" if val is not None else "no_sensor", {
                "friendly_name": r.name, "temperature": val, "target": core.target_temp(r, now),
                "priority": r.priority, "unit": r.unit,
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
             reason=info.get("reason"), dry_run=not self.data.get("enabled", True))
        for k in ("cold", "hot"):
            self.hass.states.async_set(f"sensor.{DOMAIN}_house_{k}_demand", info.get(k) or 0.0, {
                "friendly_name": f"House {k} demand", "state_class": "measurement",
                "icon": "mdi:snowflake-thermometer" if k == "cold" else "mdi:sun-thermometer"})

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
                                    (self.states if self.data.get("enabled", True) else self.shadow).get(ud["id"], core.UnitState()),
                                    self.last.get(ud["id"]), now, settings)
            for ud in self.data["units"]}
        for ud in self.data["units"]:  # what each unit is really doing, next to what we would command
            cur = self.hass.states.get(ud["entity"])
            out["units_live"][ud["id"]]["actual"] = None if cur is None else {
                "state": cur.state, "setpoint": cur.attributes.get("temperature"),
                "action": cur.attributes.get("hvac_action"), "temp": self._unit_temp(ud["entity"])}
        out["house"] = self.house_info
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
