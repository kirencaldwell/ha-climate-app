// Data the wall screen reads from Home Assistant at run time (sensor.room_climate_wall_screen), so rooms, names, weights,
// weight steps, hold times and the temperature range follow the controller without recompiling this firmware.
#pragma once
#include <cmath>
#include <string>
#include <vector>
#include "esphome/components/json/json_util.h"

namespace ws {
struct Room {
  std::string id, name;
  float temp = NAN, target = NAN, weight = NAN, hold = NAN;
};
struct State {
  std::vector<Room> rooms;
  std::vector<float> ladder{0.5f, 1.f, 2.f, 3.f, 4.f, 5.f, 10.f, 25.f, 100.f};
  std::vector<float> hours{1.f, 2.f, 4.f, 8.f};
  int tmin = 55, tmax = 85;
  float ref = 1.0f;
};
inline State S;

inline float num(JsonVariantConst v) { return v.is<float>() ? v.as<float>() : NAN; }

inline void parse_rooms(const std::string &text) {
  esphome::json::parse_json(text, [](JsonObject root) -> bool {
    std::vector<Room> out;
    for (JsonObject o : root["rooms"].as<JsonArray>()) {
      Room r;
      r.id = o["id"] | "";
      r.name = o["name"] | "";
      r.temp = num(o["temp"]);
      r.target = num(o["target"]);
      r.weight = num(o["weight"]);
      float h = num(o["hold"]);
      r.hold = h;
      out.push_back(r);
    }
    if (!out.empty()) S.rooms = out;
    return true;
  });
}

inline void parse_config(const std::string &text) {
  esphome::json::parse_json(text, [](JsonObject root) -> bool {
    std::vector<float> l;
    for (JsonVariant v : root["ladder"].as<JsonArray>()) l.push_back(v.as<float>());
    if (!l.empty()) S.ladder = l;
    std::vector<float> h;
    for (JsonVariant v : root["hours"].as<JsonArray>()) h.push_back(v.as<float>());
    while (h.size() < 4 && !h.empty()) h.push_back(h.back());
    if (h.size() >= 4) { h.resize(4); S.hours = h; }
    if (root["tmin"].is<int>()) S.tmin = (int) root["tmin"].as<float>();
    if (root["tmax"].is<int>()) S.tmax = (int) root["tmax"].as<float>();
    if (root["ref"].is<float>()) S.ref = root["ref"].as<float>();
    return true;
  });
}

inline int n_rooms() { return (int) S.rooms.size(); }
inline const Room *room(int i) { return (i >= 0 && i < n_rooms()) ? &S.rooms[i] : nullptr; }
inline std::string room_name(int i) { auto r = room(i); return r ? r->name : std::string("..."); }
inline std::string room_id(int i) { auto r = room(i); return r ? r->id : std::string(""); }
inline float cur_temp(int i) { auto r = room(i); return r ? r->temp : NAN; }
inline float cur_target(int i) { auto r = room(i); return r ? r->target : NAN; }
inline float cur_weight(int i) { auto r = room(i); return r ? r->weight : NAN; }
inline float cur_hold(int i) { auto r = room(i); return r ? r->hold : NAN; }

inline int n_rungs() { return (int) S.ladder.size(); }
inline float rung(int k) { return S.ladder[k < 0 ? 0 : (k >= n_rungs() ? n_rungs() - 1 : k)]; }
inline int nearest_rung(float w) {
  int idx = 0;
  if (std::isnan(w) || w <= 0) return n_rungs() > 1 ? 1 : 0;
  float best = 1e9f;
  for (int k = 0; k < n_rungs(); k++) {
    float d = fabsf(logf(S.ladder[k] / w));
    if (d < best) { best = d; idx = k; }
  }
  return idx;
}
inline float hours(int k) { return S.hours[k < 0 ? 0 : (k > 3 ? 3 : k)]; }
inline float allowable(float w) { return S.ref / sqrtf(w); }
}  // namespace ws
