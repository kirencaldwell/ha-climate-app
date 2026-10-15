import gzip, os, sys, tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))
import datalog  # noqa: E402


def test_rows_append_rotate_and_read_back():
    with tempfile.TemporaryDirectory() as d:
        datalog.append(d, {"t": 1}, datetime(2026, 10, 14, 23, 59))
        datalog.append(d, {"t": 2}, datetime(2026, 10, 14, 23, 59, 30))
        datalog.append(d, {"t": 3}, datetime(2026, 10, 15, 0, 0))  # new day: yesterday gets gzipped
        assert sorted(os.listdir(d)) == ["2026-10-14.jsonl.gz", "2026-10-15.jsonl"]
        with gzip.open(os.path.join(d, "2026-10-14.jsonl.gz"), "rt") as f:
            assert len(f.read().splitlines()) == 2
        with open(os.path.join(d, "2026-10-15.jsonl"), "a") as f:
            f.write('{"t": 4, "cut sh')  # power lost mid-write
        assert [r["t"] for r in datalog.read_rows(d)] == [1, 2, 3]
        datalog.append(d, {"t": 5}, datetime(2026, 10, 15, 0, 1))
        assert [r["t"] for r in datalog.read_rows(d)] == [1, 2, 3, 5]  # the cut line doesn't swallow the next one


if __name__ == "__main__":
    test_rows_append_rotate_and_read_back()
    print("ok")
