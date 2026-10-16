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
from homeassistant.core import Context, Event, HomeAssistant
from homeassistant.helpers import discovery
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import TemperatureConverter

from . import core, datalog

DOMAIN = "room_climate"
_LOGGER = logging.getLogger(__name__)
STORE_VERSION = 1
INTERVAL = timedelta(seconds=60)
F, C = UnitOfTemperature.FAHRENHEIT, UnitOfTemperature.CELSIUS

CONFIG_SCHEMA = vol.Schema({DOMAIN: vol.Schema({vol.Optional("enabled"): bool})}, extra=vol.ALLOW_EXTRA)  # legacy key, ignored

# Who drives a unit. "thermostat": its own mode and setpoint, set by hand (the default). "room_climate": we command it.
ROOM_CLIMATE, THERMOSTAT = "room_climate", "thermostat"
SIGNAL_CONTROL = f"{DOMAIN}_control"  # dispatched (unit id) when a unit's control changes
OURS_GRACE_S = 30  # unit state changes this soon after we sent it something are its echo of our command
BAD_STATES = ("unknown", "unavailable")


def _room_from_dict(d: dict) -> core.Room:
    return core.Room(
        id=d["id"], name=d.get("name", d["id"]), unit=d.get("unit"), sensor=d.get("sensor"),
        priority=int(d.get("priority", 3)), default_temp=float(d.get("default_temp", 68)),
        schedule=[core.ScheduleEntry(list(e["days"]), e["start"], float(e["temp"])) for e in d.get("schedule", [])],
        priority_schedule=[core.PriorityEntry(list(e["days"]), e["start"], int(e["priority"]))
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
        self._sent_at: dict[str, datetime] = {}  # unit id -> when we last called a service on it
        self._ours: list[str] = []  # context ids of our recent service calls
        self._audit_seen: dict[str, tuple] = {}
        self._logged_minute = None
        self._unsub = []

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

    def control(self, unit_id: str) -> str:
        ud = next((u for u in self.data["units"] if u["id"] == unit_id), {})
        return ud.get("control", THERMOSTAT)

    async def async_set_control(self, unit_id: str, control: str, why: str):
        ud = next((u for u in self.data["units"] if u["id"] == unit_id), None)
        if ud is None or ud.get("control") == control:
            return
        ud["control"] = control
        self.sent.pop(unit_id, None)  # taking over: send our setpoint now, not after the change interval
        _LOGGER.info("%s control -> %s (%s)", unit_id, control, why)
        await self.async_save()
        async_dispatcher_send(self.hass, SIGNAL_CONTROL, unit_id)
        await self.async_tick()

    async def async_unit_changed(self, event: Event):
        """A unit under our control changed mode or setpoint without us: someone used its thermostat, the app or the
        remote. Hand it to the thermostat so we don't fight them."""
        ud = next((u for u in self.data["units"] if u["entity"] == event.data["entity_id"]), None)
        old, new = event.data.get("old_state"), event.data.get("new_state")
        if ud is None or ud.get("control") != ROOM_CLIMATE or old is None or new is None:
            return
        if old.state in BAD_STATES or new.state in BAD_STATES:
            return  # dropping off and reconnecting is not a change of mind
        sp_old, sp_new = old.attributes.get("temperature"), new.attributes.get("temperature")
        if old.state == new.state and sp_old == sp_new:
            return
        at = self._sent_at.get(ud["id"])
        if event.context.id in self._ours or (at and core.elapsed(dt_util.utcnow(), at) < OURS_GRACE_S):
            return
        await self.async_set_control(ud["id"], THERMOSTAT,
                                     f"changed at the thermostat: {old.state}/{sp_old} -> {new.state}/{sp_new}")

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
        manual, busy = {}, set()
        for u in units:
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
        house = self.house
        cmds, new_house, info = core.control_step(
            units, rooms, readings, predicted, {u.id: self._unit_temp(u.entity) for u in units},
            states, house, now, settings, manual, busy, self._outdoor(settings))
        if new_house.mode != house.mode:
            _LOGGER.info("house mode %s -> %s (%s)", house.mode, new_house.mode, info["reason"])
        self.house = new_house
        self.house_info = {**info, "since": new_house.since.isoformat() if new_house.since else None,
                           "predicted": {k: None if v is None else round(v, 1) for k, v in predicted.items()}}
        self._tick_debug = (readings, predicted, info, settings)
        sent = self.sent
        for u in units:
            # Keep setpoints steady (inverter efficiency): small or too-frequent changes hold the previous one.
            cmd, changed = core.stabilize(cmds[u.id], sent.get(u.id), now, settings)
            if changed and cmd.setpoint is not None:
                sent[u.id] = (cmd, now)
            state = states.setdefault(u.id, core.UnitState())
            self.last[u.id] = cmd
            ours = self.control(u.id) == ROOM_CLIMATE
            if ours:
                await self._send(u, state, cmd)
            self._audit(u, cmd, ours, now)
            states[u.id] = core.apply(state, cmd, now)
            if core.regime_changed(state, states[u.id]):
                for r in rooms:
                    if r.unit == u.id and self.hist.get(r.id):
                        self.hist[r.id] = self.hist[r.id][-1:]
        self._publish(now, rooms)
        await self._log(now, rooms, units, predicted, info)

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
        for r in rooms:
            if not r.sensor:
                continue
            val, _ = self._f(r.sensor)  # raw: None while the sensor is out (the controller bridges short gaps)
            c, h = info.get("rooms", {}).get(r.id, (None, None))
            row["rooms"][r.id] = {"temp": num(val), "target": core.target_temp(r, now), "prio": core.room_priority(r, now),
                                  "pred": num(predicted.get(r.id)), "unit": r.unit, "cold": c, "hot": h}
        prefixes = []
        for u in units:
            cur, cmd = self.hass.states.get(u.entity), self.last.get(u.id)
            a = cur.attributes if cur is not None else {}
            row["units"][u.id] = {"entity": u.entity, "control": self.control(u.id),
                                  "state": cur.state if cur is not None else None, "sp": num(a.get("temperature")),
                                  "action": a.get("hvac_action"), "fan": a.get("fan_mode"), "temp": num(self._unit_temp(u.entity)),
                                  "cmd": cmd.mode if cmd else None, "cmd_sp": cmd.setpoint if cmd else None,
                                  "why": cmd.reason if cmd else None}
            prefixes.append("sensor." + u.entity.split(".", 1)[1] + "_")
        # Anything else the unit controllers report (signal now; fan stage, compressor Hz, outdoor coil temp if added).
        for st in self.hass.states.async_all("sensor"):
            if st.entity_id.startswith(tuple(prefixes)) and st.state not in BAD_STATES:
                row["extra"][st.entity_id] = num(st.state) if num(st.state) is not None else st.state
        path = self.hass.config.path("room_climate_data")
        try:
            await self.hass.async_add_executor_job(datalog.append, path, row, now)
        except OSError as e:
            _LOGGER.warning("long-term log write failed: %s", e)

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
            _LOGGER.info("%s %s", "CONTROL" if ours else "THERMOSTAT (would)", line)

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
        out["live"] = {}
        for rd in self.data["rooms"]:
            r = _room_from_dict(rd)
            val, _ = self._f(r.sensor) if r.sensor else (None, None)
            c = self.last.get(r.unit)
            out["live"][r.id] = {"temp": val, "target": core.target_temp(r, now), "priority": core.room_priority(r, now),
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
        vol.Optional("priority_schedule"): [vol.Schema({
            vol.Required("days"): [vol.All(int, vol.Range(min=0, max=6))],
            vol.Required("start"): vol.Match(r"^\d{2}:\d{2}$"),
            vol.Required("priority"): vol.All(int, vol.Range(min=1, max=5))})],
        vol.Optional("hold_temp"): vol.Any(None, vol.Coerce(float)),
        vol.Optional("hold_hours"): vol.Any(None, vol.Coerce(float)),
    })
    @websocket_api.async_response
    async def ws_set_room(hass, connection, msg):
        room = next((r for r in mgr.data["rooms"] if r["id"] == msg["room_id"]), None)
        if room is None:
            connection.send_error(msg["id"], "not_found", "unknown room")
            return
        for k in ("priority", "default_temp", "schedule", "priority_schedule", "hold_temp"):
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
        vol.Required("control"): vol.In([ROOM_CLIMATE, THERMOSTAT]),
    })
    @websocket_api.async_response
    async def ws_set_control(hass, connection, msg):
        if not any(u["id"] == msg["unit_id"] for u in mgr.data["units"]):
            connection.send_error(msg["id"], "not_found", "unknown unit")
            return
        await mgr.async_set_control(msg["unit_id"], msg["control"], "set in the Room Climate card")
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

    for fn in (ws_get, ws_set_room, ws_set_control, ws_set_settings, ws_log_client):
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
    return True
