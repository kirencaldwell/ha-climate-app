"""Mirror what the ML training needs from the Home Assistant Green to this machine, incrementally.

    python3 tools/ha_sync.py [--dest ~/.local/share/room-climate-watch/ha]

Mirrors (under --dest, same paths as on the Green):
  room_climate_data/*.jsonl[.gz]         the minute logs (finished days are gzipped and never change: fetched once;
                                         today's file changes: refetched)
  room_climate_ml_rows.jsonl             the live planner's ML training rows (append-only, trimmed nightly: only the new
                                         bytes are fetched, a full refetch if the Green trimmed it)
  room_climate_model.json, room_climate_model_fits.jsonl, room_climate_ml.json, .storage/room_climate
                                         the fitted model, its history, the Green's own ML correction, the room layout

The Green's SSH add-on has no rsync, so this does what rsync would over plain ssh (size/hash checks, byte offsets). If
rsync is ever added there (the add-on's `apks` option), use it instead. Nothing on the Green is changed.
"""
import argparse, hashlib, os, subprocess, sys, time

KEY = os.path.expanduser("~/.ssh/homeassistant")
HOST = "root@192.168.4.54"
SSH = ["ssh", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST]
CONFIG = "/config"
SMALL = ["room_climate_model.json", "room_climate_model_fits.jsonl", "room_climate_ml.json", ".storage/room_climate"]
ROWS = "room_climate_ml_rows.jsonl"


def remote(cmd: str, timeout: int = 120) -> bytes:
    return subprocess.run(SSH + [cmd], check=True, capture_output=True, timeout=timeout).stdout


def stat(path: str):
    """(size, mtime) of a file on the Green, None if it isn't there."""
    out = remote(f"stat -c '%s %Y' {path} 2>/dev/null || true").decode().split()
    return (int(out[0]), int(out[1])) if len(out) == 2 else None


def fetch(path: str, dest: str, offset: int = 0) -> int:
    """Copy a remote file (from byte `offset`) to dest (appending when offset > 0). -> bytes fetched."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = remote(f"tail -c +{offset + 1} {path}" if offset else f"cat {path}", timeout=300)
    tmp = dest + ".part"
    if offset:
        with open(dest, "rb") as f, open(tmp, "wb") as g:
            g.write(f.read(offset))
            g.write(data)
    else:
        with open(tmp, "wb") as g:
            g.write(data)
    os.replace(tmp, dest)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.expanduser("~/.local/share/room-climate-watch/ha"))
    a = ap.parse_args()
    got, t0 = {}, time.time()
    # the minute logs
    listing = remote(f"cd {CONFIG}/room_climate_data && stat -c '%n %s' *").decode().split("\n")
    for line in listing:
        if not line.strip():
            continue
        name, size = line.rsplit(" ", 1)
        local = os.path.join(a.dest, "room_climate_data", name)
        if name.endswith(".gz") and os.path.exists(local) and os.path.getsize(local) == int(size):
            continue  # a finished day: never changes
        if not name.endswith(".gz") and os.path.exists(local) and os.path.getsize(local) == int(size):
            continue  # today's file, unchanged since the last sync
        got[name] = fetch(f"{CONFIG}/room_climate_data/{name}", local)
    # the append-only ML rows
    st = stat(f"{CONFIG}/{ROWS}")
    local = os.path.join(a.dest, ROWS)
    if st:
        size = st[0]
        have = os.path.getsize(local) if os.path.exists(local) else 0
        if have == size:
            pass
        elif have < size and have > 0 and remote(f"head -c 4096 {CONFIG}/{ROWS} | sha256sum").split()[0].decode() == \
                hashlib.sha256(open(local, "rb").read(4096)).hexdigest() and have >= 4096:
            got[ROWS + " (new bytes)"] = fetch(f"{CONFIG}/{ROWS}", local, offset=have)  # appended since: just the tail
        else:
            got[ROWS] = fetch(f"{CONFIG}/{ROWS}", local)  # first time, or the Green trimmed it
    # the small files: refetch when their size or content changed
    for rel in SMALL:
        st = stat(f"{CONFIG}/{rel}")
        if not st:
            continue
        local = os.path.join(a.dest, rel)
        if os.path.exists(local) and os.path.getsize(local) == st[0]:
            sha_remote = remote(f"sha256sum {CONFIG}/{rel}").split()[0].decode()
            if hashlib.sha256(open(local, "rb").read()).hexdigest() == sha_remote:
                continue
        got[rel] = fetch(f"{CONFIG}/{rel}", local)
    total = sum(got.values())
    print(f"ha_sync: {len(got)} file(s), {total / 1024:.0f} KiB fetched in {time.time() - t0:.1f}s -> {a.dest}")
    for k, v in sorted(got.items()):
        print(f"  {k}: {v / 1024:.0f} KiB")


if __name__ == "__main__":
    sys.exit(main())
