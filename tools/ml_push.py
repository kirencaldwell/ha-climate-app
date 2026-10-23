"""Copy an ML correction trained on this machine to the Home Assistant Green and make it load it.

    python3 tools/ml_push.py ~/.local/share/room-climate-watch/ml/blend.json

Checks the file loads (mlmodel.load), writes it to /config/room_climate_ml_external.json on the Green (to a temp name, then
moved into place), and asks the integration to reload it (websocket room_climate/reload_ml: no restart). The Green keeps
using it in preference to the ridge it trains for itself, until the file is removed. The ML is shadow-only: it scores
predictions, it doesn't control anything.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: select.py must not shadow stdlib
import comfort_fix as cf  # noqa: E402  (needs the websockets package)
import ha_sync  # noqa: E402
import mlmodel  # noqa: E402

DEST = "/config/room_climate_ml_external.json"


def main():
    path = sys.argv[1]
    raw = open(path, "rb").read()
    c = mlmodel.load(json.loads(raw))  # raises if it isn't a model we can run
    if len(raw) > 8_000_000:
        raise SystemExit("model file is over 8 MB: refusing")
    subprocess.run(ha_sync.SSH + [f"cat > {DEST}.tmp && mv {DEST}.tmp {DEST}"], input=raw, check=True, timeout=120)
    state = cf.call({"type": "room_climate/reload_ml"})[0]
    print(f"pushed {len(raw) / 1024:.0f} KiB ({getattr(c, 'kind', 'ridge')}); the Green says: {state}")


if __name__ == "__main__":
    main()
