"""Long-term minute log: one JSON object per line in <dir>/YYYY-MM-DD.jsonl, finished days gzipped.

No Home Assistant imports, so the file handling is testable on its own (and runs in an executor job).
Read it back with `read_rows`.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
from datetime import datetime


def append(directory: str, row: dict, now: datetime) -> None:
    """Append `row` to today's file and compress any finished plain-text days."""
    os.makedirs(directory, exist_ok=True)
    today = f"{now:%Y-%m-%d}.jsonl"
    path = os.path.join(directory, today)
    lead = ""
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                lead = "\n"  # the last write was cut short (power loss): don't glue this row onto it
    with open(path, "a", encoding="utf-8") as f:
        f.write(lead + json.dumps(row, separators=(",", ":"), default=str) + "\n")
    for name in os.listdir(directory):
        if name.endswith(".jsonl") and name != today:
            _gzip(os.path.join(directory, name))


def _gzip(path: str) -> None:
    tmp = path + ".gz.tmp"
    with open(path, "rb") as src, gzip.open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(tmp, path + ".gz")  # the .gz appears complete or not at all
    os.remove(path)


def read_rows(directory: str):
    """Every logged row, oldest first (plain and gzipped days)."""
    names = sorted(n for n in os.listdir(directory) if n.endswith((".jsonl", ".jsonl.gz")))
    for name in names:
        path = os.path.join(directory, name)
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue  # a line cut short by a crash or power loss
