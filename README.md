# Room Climate (Home Assistant)

Holds each room at its own temperature (e.g. 68°F) using a wifi sensor in the room, instead of the
Mitsubishi unit's own sensor and its 70-75°F-minded auto mode.

- `custom_components/room_climate/core.py` – pure control logic (schedules, priority weighting, hysteresis, min run/off times, sensor-offset compensation). No HA imports.
- `custom_components/room_climate/__init__.py` – HA integration: persisted rooms/units, 60 s loop + reacts to sensor updates, websocket API (`room_climate/get`, `room_climate/set_room`), publishes `room_climate.<room>` states.
- `custom_components/room_climate/www/room-map-card.js` – Lovelace card: floorplan map, click a room, set default temp, priority (1-5), schedule, temporary hold.
- `sim/house.py` + `tests/` – single-zone thermal simulator and tests.
- `esphome/` – example config for the ESP32 boards (assumed CN105 component; unverified).

## How control works
Each unit serves one or more rooms (the air handler can serve several first-floor rooms; each mini split serves one).
The unit's error is the priority-weighted average of (room sensor − room target). It heats/cools when the error passes the
deadband (±1°F), stops at target, and respects minimum run/off times and a heat↔cool lockout. The unit's setpoint is
`target + (unit sensor − room sensor) ± 1`, so the unit's own thermostat chases the *room* temperature. Stale/missing
sensors are ignored; with none, the unit is left alone.

## Model-based control ("would do" mode: shown, not sent)
A learned thermal model plus a model-predictive optimizer. HA refits the model every night at 3:40 on the last 14 days
of the minute log; the optimizer plans every 5 minutes next to today's controller and its decisions are logged and
shown on the card's **Model** page, with live prediction accuracy (30/60/120 min ahead, replaying what the units
actually did, vs "nothing changes"), predicted-vs-actual charts and the learned room x unit sensitivities.
- `custom_components/room_climate/model.py` – grey-box model: each room's loss to outdoors, sun gain, directed coupling
  to other rooms (heat rising up the stairs), each unit's heating/cooling effect, and the units' own inverter loop.
- `custom_components/room_climate/optimizer.py` – every few minutes, tries heat/cool and setpoints (two blocks: now and
  later), simulates ~2 h ahead and minimizes Σ priority × (miss outside the comfort band)², plus charges for mode
  switches, setpoint moves and energy. Priorities and their schedules are the weights.
- `custom_components/room_climate/datalog.py` – the minute log in `/config/room_climate_data` the model learns from.
- `custom_components/room_climate/fit.py` – fits the model to logs (pure Python; non-negative ridge regression on
  30-min windows, unit sensor rooms detected from the log) and checks it by predicting held-out data 2 h ahead; a fit
  is only `usable` if it beats "nothing changes" for every room. `tools/fit_model.py` runs it from the command line.
- `custom_components/room_climate/mpc.py` – the optimizer in "would do" mode plus live accuracy tracking.
- `sim/coupled.py` + `tools/compare_controllers.py` – the simulated house (real rooms, schedules and priorities;
  quantized sensors; 0.5 °C unit setpoints; real sun path) and a head-to-head of today's controller vs the optimizer.

Tests: plain-Python runners in `tests/` (`python3 tests/test_model.py` etc.); the card's are in `dev/test_editor.html`.

## Install (HACS)
1. HACS → ⋮ → **Custom repositories** → add `https://github.com/kirencaldwell/ha-climate-app`, category **Integration**, then download **Room Climate**.
2. Add `room_climate:` to `configuration.yaml` and restart Home Assistant.
3. Add the card to a dashboard: `type: custom:room-map-card` (the integration loads the card's JavaScript itself; no resource to add).
   Optional card keys: `theme: tech`, `north: left`, `floorplan`, `image_size`, `floor_views` (see the header of `room-map-card.js`).
4. Rooms/units are seeded from `default_config.json` on first run and then stored in `.storage/room_climate`; the entity ids in the seed are placeholders until the hardware exists.

**Updating:** push to GitHub; HACS shows an update, click **Update**, then restart HA (Python changes) and refresh the browser (card changes). Bump `version` in `manifest.json` so browsers pick up a new card.

Manual install also works: copy `custom_components/room_climate` into HA's `config/custom_components/`.

## Status
Core logic and simulator are tested (13 tests). The integration loads in a real Home Assistant container and its websocket API was exercised end to end with fake sensors; it has not driven real climate entities, and there is no hardware yet.
Not built yet: editing rooms/units from the UI, outdoor-temperature awareness, away mode.
