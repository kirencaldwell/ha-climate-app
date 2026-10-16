"""The fitter recovers the simulated house from its own logs. Run: python3 tests/test_fit.py"""
import os
import sys
from datetime import datetime

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import coupled  # noqa: E402
import fit  # noqa: E402


def test_fit_recovers_the_simulated_house():
    truth = coupled.the_house()
    cfg = {"rooms": [{"id": r, "unit": u, "priority": 3, "default_temp": 67} for r, u in (
        ("kitchen", "air_handler"), ("living_room", "air_handler"), ("dining_area", "air_handler"),
        ("bedroom_1", "air_handler"), ("bedroom_2", "air_handler"),
        ("primary_bedroom", "minisplit_1"), ("bedroom_3", "minisplit_2"))],
        "units": [{"id": u} for u in truth.units], "settings": {"floor": 63.0}}
    rooms, units = coupled.rooms_units_from_config(cfg, truth)
    s = coupled.settings_from_config(cfg)
    start = datetime(2026, 10, 16)
    log = []
    coupled.run(truth, rooms, units, coupled.HeuristicController(rooms, units, s), start, 7,
                coupled.Weather(start, 9, mean=55, seed=11), s, log=log, seed=11)
    neighbors = {rid: sorted(set(r.coupling) | {j for j, o in truth.rooms.items() if rid in o.coupling})
                 for rid, r in truth.rooms.items()}
    layout = {"rooms": {rid: {"units": list(truth.units), "neighbors": neighbors[rid]} for rid in truth.rooms},
              "units": {u: {"zone": [r.id for r in rooms if r.unit == u]} for u in truth.units}}  # sensor rooms: detected
    m, rep = fit.fit(log, layout)
    assert m.meta["usable"]
    for rid, tr in truth.rooms.items():
        fr = m.rooms[rid]
        assert abs(fr.loss - tr.loss) < 0.25 * tr.loss, (rid, fr.loss, tr.loss)
        own = next(iter(tr.heat))
        assert abs(fr.heat[own] - tr.heat[own]) < 0.25 * tr.heat[own], (rid, fr.heat, tr.heat)
        assert abs(fr.cool[own] - tr.cool[own]) < 0.25 * tr.cool[own], (rid, fr.cool, tr.cool)
        others = [v for u, v in fr.heat.items() if u != own]
        assert max(others) < 0.1 * tr.heat[own], (rid, fr.heat)  # no effect where there is none
    up = m.rooms["primary_bedroom"].coupling.get("living_room", 0)
    assert 0.05 < up < 0.2, up  # the stairs (true 0.12)
    assert {u: v.sensor_rooms for u, v in m.units.items()} == {u: v.sensor_rooms for u, v in truth.units.items()}
    for rid, v in rep["holdout"].items():
        assert v["model_rmse_F"] < 0.5 * v["no_change_rmse_F"], (rid, v)


if __name__ == "__main__":
    test_fit_recovers_the_simulated_house()
    print("ok")
