"""Standalone dev harness: serves the Lovelace card against the real control logic + thermal sim, no Home Assistant.

    python dev/server.py [--port 8124] [--speed 10]     # speed = simulated minutes per real second
"""
import argparse, dataclasses, json, os, sys, threading, time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMP = os.path.join(ROOT, "custom_components", "room_climate")
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(ROOT, "sim"))
import core  # noqa: E402
from house import SimHouse  # noqa: E402

cfg = json.load(open(os.path.join(COMP, "default_config.json")))
cfg.pop("_comment", None)


def room_from_dict(d):
    return core.Room(
        id=d["id"], name=d.get("name", d["id"]), unit=d.get("unit"), sensor=d.get("sensor"),
        priority=int(d.get("priority", 3)), default_temp=float(d.get("default_temp", 68)),
        schedule=[core.ScheduleEntry(list(e["days"]), e["start"], float(e["temp"])) for e in d.get("schedule", [])],
        hold_temp=d.get("hold_temp"),
        hold_until=datetime.fromisoformat(d["hold_until"]) if d.get("hold_until") else None,
        floor=int(d.get("floor", 1)), shape=d.get("shape", {}))


state = {"outdoor": 40.0, "speed": 10.0}
lock = threading.Lock()


def build_house():
    rooms = [room_from_dict(r) for r in cfg["rooms"]]
    units = [core.Unit(u["id"], u["entity"]) for u in cfg["units"]]
    return SimHouse(rooms, units, lambda _t: state["outdoor"], datetime.now().replace(second=0, microsecond=0))


house = build_house()


def sync_rooms():
    """Rebuild core rooms from the (possibly edited) config dicts."""
    for d in cfg["rooms"]:
        house.rooms[d["id"]] = room_from_dict(d)


def snapshot():
    out = dict(cfg)
    out["live"] = {}
    for r in house.rooms.values():
        c = house.cmd.get(r.unit)
        out["live"][r.id] = {"temp": round(house.sim[r.id].temp, 2), "target": core.target_temp(r, house.now),
                             "mode": c.mode if c else "off", "reason": c.reason if c else ""}
    readings = {r.id: (house.sim[r.id].temp, house.now) for r in house.rooms.values()}
    out["now"] = house.now.isoformat()
    out["settings_effective"] = dataclasses.asdict(house.settings)
    out["units_live"] = {}
    for uid, unit in house.units.items():
        cmd = house.cmd.get(uid)  # last command from the simulated control loop
        out["units_live"][uid] = core.diagnose(unit, list(house.rooms.values()), readings, house.unit_temp(uid),
                                               house.state[uid], cmd, house.now, house.settings)
    out["sim"] = {"time": house.now.strftime("%a %H:%M"), **state}
    return out


def set_room(msg):
    room = next((r for r in cfg["rooms"] if r["id"] == msg["room_id"]), None)
    if room is None:
        raise KeyError("unknown room")
    for k in ("priority", "default_temp", "schedule", "hold_temp"):
        if k in msg:
            room[k] = msg[k]
    if "hold_hours" in msg:
        h = msg["hold_hours"]
        room["hold_until"] = None if not h else (house.now + timedelta(hours=h)).isoformat()
    if msg.get("hold_temp", 1) is None:
        room.pop("hold_until", None)
    sync_rooms()
    return snapshot()


def ticker():
    while True:
        time.sleep(1)
        with lock:
            house.step(max(1, int(state["speed"])))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.path = self.path.split("?")[0]
        if self.path in ("/", "/index.html"):
            self._send(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html")
        elif self.path == "/test_editor.html":
            self._send(open(os.path.join(HERE, "test_editor.html"), "rb").read(), "text/html")
        elif self.path == "/floorplan.jpg":
            self._send(open(os.path.join(ROOT, "floorplan.jpg"), "rb").read(), "image/jpeg")
        elif self.path == "/room-map-card.js":
            self._send(open(os.path.join(COMP, "www", "room-map-card.js"), "rb").read(), "text/javascript")
        else:
            self.send_error(404)

    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            if msg["type"] == "room_climate/get":
                self._send(snapshot())
            elif msg["type"] == "room_climate/set_room":
                self._send(set_room(msg))
            elif msg["type"] == "sim/set":
                for k in ("outdoor", "speed"):
                    if k in msg:
                        state[k] = float(msg[k])
                self._send(snapshot())
            else:
                self.send_error(400)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--speed", type=float, default=10)
    a = ap.parse_args()
    state["speed"] = a.speed
    threading.Thread(target=ticker, daemon=True).start()
    print(f"Room Climate dev harness: http://localhost:{a.port}  (sim speed {a.speed} min/s)")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
