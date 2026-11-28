# Climate wall screen (Waveshare ESP32-S3-Touch-LCD-4, rev 4.0)

A 480x480 touch thermostat for Room Climate: pick a room (< >), set the target temperature (left + / -) and the weight
(right + / -), see the room's current temperature in the middle, then hold the new values for 1/2/4/8 h or clear a hold.
The hold buttons are faded until something is changed.

- `gen_wall_screen.py` generates `wall-screen.yaml` (edit the generator, not the YAML); `wall_screen.h` parses the data.
- **Synced from Room Climate at run time, no reflashing:** the room list, order and names, each room's current temperature, target,
  weight and hold time left, the weight steps (`core.WEIGHT_LADDER`), the hold buttons' hours (setting `screen_hold_hours`,
  default `1,2,4,8`), the temperature range (`screen_temp_min` / `screen_temp_max`) and the weight reference
  (`allowable_ref_f`). All of it comes from `sensor.room_climate_wall_screen` (attributes `rooms_json`, `config_json`).
- **Needs a new firmware (recompile, then flash over USB or Wi-Fi):** the look and layout, the number of hold buttons (four) and
  the colours.
- Actions: `room_climate.set_hold` (room_id, temperature, weight, hours) and `room_climate.clear_hold` (room_id).
- Build (ESPHome container): `docker run --rm -v $PWD:/config ghcr.io/esphome/esphome compile wall-screen.yaml` in a folder
  with `wall-screen.yaml`, `wall_screen.h` and a `secrets.yaml` (wifi_ssid, wifi_password, api_key, ota_password).
- Board notes: rev 4.0 uses a CH32V003 expander at 0x24 (not a TCA9554); the buzzer stays on from boot unless switched off
  (done in the config); pixel clock 10 MHz and a larger data cache stop flicker; Waveshare's init sequence fixes the colours.
