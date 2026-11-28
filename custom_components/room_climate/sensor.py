"""Per-room sensors for wall screens and automations: the room's temperature as the controller sees it, its target now, its
weight now, and the minutes left on a hold. Read-only views of what the controller already computes."""
from __future__ import annotations

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.util import dt as dt_util

import json

from . import DOMAIN, SIGNAL_STATE, _room_from_dict
from . import core


async def async_setup_platform(hass: HomeAssistant, config, async_add_entities, discovery_info=None):
    if discovery_info is None:
        return
    mgr = hass.data[DOMAIN]
    ents = []
    for rd in mgr.data["rooms"]:
        if rd.get("unit"):  # rooms a unit serves
            ents += [RoomSensor(mgr, rd, kind) for kind in ("temperature", "target", "weight", "hold")]
    ents.append(WallScreenSensor(mgr))
    async_add_entities(ents)


class RoomSensor(SensorEntity):
    _attr_should_poll = False

    def __init__(self, mgr, rd: dict, kind: str):
        self._mgr, self._room, self._kind = mgr, rd["id"], kind
        self._attr_unique_id = f"{DOMAIN}_{rd['id']}_{kind}"
        self.entity_id = f"sensor.{DOMAIN}_{rd['id']}_{kind}"
        self._attr_name = f"{rd.get('name') or rd['id']} {kind}"
        if kind in ("temperature", "target"):
            self._attr_native_unit_of_measurement = "°F"
            self._attr_device_class = "temperature"
            self._attr_state_class = SensorStateClass.MEASUREMENT
        elif kind == "hold":
            self._attr_native_unit_of_measurement = "min"
            self._attr_icon = "mdi:timer-sand"
        else:
            self._attr_icon = "mdi:scale-balance"

    def _data(self):
        rd = next((r for r in self._mgr.data["rooms"] if r["id"] == self._room), None)
        return rd, (_room_from_dict(rd) if rd else None)

    @property
    def available(self) -> bool:
        return self._data()[0] is not None

    @property
    def native_value(self):
        rd, r = self._data()
        if r is None:
            return None
        now = dt_util.now()
        if self._kind == "temperature":
            val, _ = self._mgr._f(r.sensor) if r.sensor else (None, None)
            return None if val is None else round(val, 1)
        if self._kind == "target":
            return round(core.target_temp(r, now), 1)
        if self._kind == "weight":
            return core.room_priority(r, now)
        # hold: minutes left (0 when there is none)
        if r.hold_temp is None and r.hold_priority is None:
            return 0
        if r.hold_until is None:
            return None  # held with no end time
        return max(0, round(core.elapsed(r.hold_until, now) / 60))

    @property
    def extra_state_attributes(self):
        rd, r = self._data()
        if r is None:
            return {}
        if self._kind == "hold":
            return {"hold_temperature": r.hold_temp, "hold_weight": r.hold_priority,
                    "hold_until": r.hold_until.isoformat() if r.hold_until else None}
        if self._kind == "weight":
            return {"allowable_error_f": round(core.allowable_error(core.room_priority(r, dt_util.now())), 2)}
        return {}

    async def async_added_to_hass(self) -> None:
        @callback
        def _changed() -> None:
            self.async_write_ha_state()

        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_STATE, _changed))


class WallScreenSensor(SensorEntity):
    """Everything a wall screen needs in one entity, so it never has to be recompiled when something here changes: the rooms
    (id, name, reading, target, weight, hold minutes left) and the screen's settings (weight steps, hold hours, range).
    Both attributes are JSON strings, which is what ESPHome's Home Assistant text sensors can read."""
    _attr_should_poll = False
    _attr_icon = "mdi:monitor-dashboard"

    def __init__(self, mgr):
        self._mgr = mgr
        self._attr_unique_id = f"{DOMAIN}_wall_screen"
        self.entity_id = f"sensor.{DOMAIN}_wall_screen"
        self._attr_name = "Wall screen data"

    def _build(self):
        mgr = self._mgr
        now = dt_util.now()
        st = core.settings_from(mgr.data.get("settings"))
        rooms = []
        ent = {u["id"]: u["entity"] for u in mgr.data["units"]}
        for rd in mgr.data["rooms"]:
            if not rd.get("unit"):
                continue
            r = _room_from_dict(rd)
            val, _ = mgr._f(r.sensor) if r.sensor else (None, None)
            if r.hold_temp is None and r.hold_priority is None:
                hold = 0
            elif r.hold_until is None:
                hold = -1  # held with no end time
            else:
                hold = max(0, round(core.elapsed(r.hold_until, now) / 60))
            unit = mgr.hass.states.get(ent.get(r.unit)) if r.unit in ent else None  # the unit that serves this room
            mode = unit.state if unit else "unavailable"
            act = (unit.attributes.get("hvac_action") or "") if unit and unit.state not in ("unavailable", "unknown") else ""
            rooms.append({"id": r.id, "name": r.name, "temp": None if val is None else round(val, 1),
                          "target": round(core.target_temp(r, now), 1), "weight": core.room_priority(r, now), "hold": hold,
                          "mode": mode, "act": str(act)})
        try:
            hours = [float(x) for x in str(st.screen_hold_hours).split(",") if x.strip()][:4] or [1, 2, 4, 8]
        except ValueError:
            hours = [1, 2, 4, 8]
        cfg = {"ladder": list(core.WEIGHT_LADDER), "hours": hours, "tmin": st.screen_temp_min, "tmax": st.screen_temp_max,
               "ref": st.allowable_ref_f}
        return rooms, cfg

    @property
    def native_value(self):
        return len(self._build()[0])

    @property
    def extra_state_attributes(self):
        rooms, cfg = self._build()
        shapes = [{"id": rd["id"], "f": int(rd.get("floor", 1)), "x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]}
                  for rd in self._mgr.data["rooms"] if (s := rd.get("shape")) and all(k in s for k in ("x", "y", "w", "h"))]
        return {"rooms_json": json.dumps({"house": (getattr(self._mgr, "house_info", None) or {}).get("mode") or self._mgr.house.mode or "off", "rooms": rooms}, separators=(",", ":")), "config_json": json.dumps(cfg, separators=(",", ":")),
                "shapes_json": json.dumps({"shapes": shapes}, separators=(",", ":"))}

    async def async_added_to_hass(self) -> None:
        @callback
        def _changed() -> None:
            self.async_write_ha_state()

        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_STATE, _changed))
