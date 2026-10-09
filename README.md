# Room Climate (Home Assistant)

Holds each room at its own temperature (e.g. 68°F) using a wifi sensor in the room, instead of the
Mitsubishi unit's own sensor and its 70-75°F-minded auto mode.

- `custom_components/room_climate/core.py` – pure control logic (schedules, priority weighting, hysteresis, min run/off times, sensor-offset compensation). No HA imports.
- `custom_components/room_climate/__init__.py` – HA integration: persisted rooms/units, 60 s loop + reacts to sensor updates, websocket API (`room_climate/get`, `room_climate/set_room`), publishes `room_climate.<room>` states.
- `custom_components/room_climate/www/room-map-card.js` – Lovelace card: floorplan map, click a room, set default temp, priority (1-5), schedule, temporary hold.
- `sim/house.py` + `tests/` – thermal simulator and tests (`python -m pytest`).
- `esphome/` – example config for the ESP32 boards (assumed CN105 component; unverified).

## How control works
Each unit serves one or more rooms (the air handler can serve several first-floor rooms; each mini split serves one).
The unit's error is the priority-weighted average of (room sensor − room target). It heats/cools when the error passes the
deadband (±1°F), stops at target, and respects minimum run/off times and a heat↔cool lockout. The unit's setpoint is
`target + (unit sensor − room sensor) ± 1`, so the unit's own thermostat chases the *room* temperature. Stale/missing
sensors are ignored; with none, the unit is left alone.

## Install
1. Copy `custom_components/room_climate` into HA's `config/custom_components/`.
2. Add `room_climate:` to `configuration.yaml`, restart.
3. Edit rooms/units (entity ids, floorplan rectangles) – seeded from `default_config.json` on first run.
4. Add resource `/room_climate_static/room-map-card.js` (module), then a card `type: custom:room-map-card` (optional `floorplan: /local/floorplan.svg`).

## Status
Core logic and simulator are tested (13 tests). The HA integration and card are written but **not yet run inside Home Assistant**; no hardware yet.
Not built yet: editing rooms/units from the UI, outdoor-temperature awareness, away mode.
