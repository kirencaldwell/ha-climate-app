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

## Shared mode (2-zone system)
All three indoor units share one outdoor unit, so they must all be in the same mode (heat or cool). `decide_system` handles that:
- Every room votes: priority x degrees off target, past the 1°F deadband, for heat or cool. Higher total wins; a tie goes to cool.
- The running mode is only displaced when the opposite vote exceeds the running mode's *remaining* shortfall by 1.5x (`switch_ratio`), after the 5 min minimum run. It then goes fully off and the new mode waits out the 15 min heat/cool lockout.
- Units that don't need the winning mode coast (`idle_mode`: `off` by default, or `fan_only`), they never run the opposite mode.
- A room with no fresh sensor reading doesn't vote.

## Predictive control
Instead of waiting for a room to leave its band, the app predicts where each room is heading and acts early.
- **Model:** each room learns `dT/dt = a·(T_out − T) + heat/cool effect + s·sun` online (recursive least squares, 5 min samples, forgets old data over a few days) from its own sensor history. It is saved with the integration's storage and needs about 12 h of data before it is used; until then (or without a weather entity) control is exactly the reactive behaviour above.
- **Inputs:** the weather entity's hourly forecast (`weather_entity` in the config; temperature and cloud cover), the real sun position at the house (from HA's configured location, i.e. your address, or `site: {lat, lon}`), and each room's `windows: [{facing: compass degrees, area: relative size}]`. Sunrise/sunset are computed from the same position (`predict.sun_times`).
- **Use:** the temperature the room would reach in `precompensate_min` (60) with its unit idle is blended into the error (`pred_weight` 0.6). A room that will leave the band starts heating or cooling early, and votes for the shared mode on the blend. While running it carries on up to `preheat_max` (0.5°F) past target to coast through the swing. A room already ≥ `pred_guard` (3°F) off ignores prediction.
- **Not covered yet:** schedule changes are not anticipated (only drift), so a setback never triggers active cooling. Floorplan-derived window positions will replace the hand-entered `windows` once the plan is uploaded.

## Install
1. Copy `custom_components/room_climate` into HA's `config/custom_components/`.
2. Add `room_climate:` to `configuration.yaml`, restart.
3. Edit rooms/units (entity ids, floorplan rectangles) – seeded from `default_config.json` on first run.
4. Add resource `/room_climate_static/room-map-card.js` (module), then a card `type: custom:room-map-card` (optional `floorplan: /local/floorplan.svg`).

## Status
Core logic and simulator are tested . The HA integration and card are written but **not yet run inside Home Assistant**; no hardware yet.
Not built yet: editing rooms/units from the UI, outdoor-temperature awareness, away mode.
