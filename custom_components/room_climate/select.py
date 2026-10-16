"""Per-unit control select: "Room Climate" (today's controller commands the unit), "Model" (the model-based controller
does, once it's ready) or "Thermostat" (its own mode and setpoint)."""
from __future__ import annotations

from homeassistant.components.select import SelectEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from . import DOMAIN, MODEL, ROOM_CLIMATE, SIGNAL_CONTROL, THERMOSTAT

LABELS = {ROOM_CLIMATE: "Room Climate", MODEL: "Model", THERMOSTAT: "Thermostat"}


async def async_setup_platform(hass: HomeAssistant, config, async_add_entities, discovery_info=None):
    if discovery_info is None:
        return
    mgr = hass.data[DOMAIN]
    async_add_entities([ControlSelect(mgr, ud) for ud in mgr.data["units"]])


class ControlSelect(SelectEntity):
    _attr_should_poll = False
    _attr_icon = "mdi:thermostat-auto"
    _attr_options = list(LABELS.values())

    def __init__(self, mgr, ud: dict):
        self._mgr, self._unit = mgr, ud["id"]
        self._attr_unique_id = f"{DOMAIN}_{ud['id']}_control"
        self.entity_id = f"select.{DOMAIN}_{ud['id']}_control"
        self._attr_name = f"{ud.get('name') or ud['id']} control"
        self._attr_extra_state_attributes = {"unit": ud["id"], "entity": ud["entity"]}

    @property
    def current_option(self) -> str:
        return LABELS[self._mgr.control(self._unit)]

    async def async_select_option(self, option: str) -> None:
        control = next(k for k, v in LABELS.items() if v == option)
        try:
            await self._mgr.async_set_control(self._unit, control, "set from its control select")
        except ValueError as e:
            raise HomeAssistantError(str(e)) from e

    async def async_added_to_hass(self) -> None:
        @callback
        def _changed(unit_id: str) -> None:
            if unit_id == self._unit:
                self.async_write_ha_state()

        self.async_on_remove(async_dispatcher_connect(self.hass, SIGNAL_CONTROL, _changed))
