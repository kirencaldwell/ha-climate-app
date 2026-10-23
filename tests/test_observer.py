"""The planner's feedback: model corrections (offsets, leak scale), the observer that learns them, and the switch margin.
Run: python3 tests/test_observer.py"""
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))  # appended: its select.py must not shadow the stdlib select module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import core  # noqa: E402
import coupled  # noqa: E402
import disturbed  # noqa: E402
import model as hm  # noqa: E402
import observer  # noqa: E402
import optimizer  # noqa: E402

START = datetime.fromisoformat("2026-10-16T00:00:00-07:00")


def test_model_hooks_add_offsets_and_scale_the_leak():
    m = coupled.the_house()
    temps = {r: 66.0 for r in m.rooms}
    base = hm.derivative(m, temps, 50.0, 0.0, {}, {"minisplit_2": 1.0})
    off = hm.derivative(m, temps, 50.0, 0.0, {}, {"minisplit_2": 1.0}, offsets={"kitchen": 0.7})
    assert abs(off["kitchen"] - base["kitchen"] - 0.7) < 1e-9 and off["bedroom_3"] == base["bedroom_3"]
    dg = {}
    big = hm.derivative(m, temps, 50.0, 0.0, {}, {"minisplit_2": 1.0}, leak_scale={"bedroom_3": 2.0}, diag=dg)
    leak = m.rooms["bedroom_3"].leak["minisplit_2"]
    assert abs(dg["leak"]["bedroom_3"] - 2.0 * leak) < 1e-9
    assert abs(big["bedroom_3"] - base["bedroom_3"] - leak) < 1e-9  # one more leak's worth
    es = hm.derivative(m, temps, 50.0, 0.0, {"air_handler": 0.5}, effect_scale={"kitchen": {"air_handler": 0.4}})
    plain = hm.derivative(m, temps, 50.0, 0.0, {"air_handler": 0.5})
    eff = m.rooms["kitchen"].heat["air_handler"] * 0.5
    assert abs((plain["kitchen"] - es["kitchen"]) - 0.6 * eff) < 1e-9  # 40% of the unit's effect is left
    acc = {}
    hm.step(m, temps, 50.0, 0.0, "heat", {"air_handler": 75.0, "minisplit_1": 60.0, "minisplit_2": 60.0},
            {"air_handler": True}, 0.25, contrib=acc)
    assert acc["unit"]["kitchen"]["air_handler"] > 0 and "bedroom_3" in acc["leak"]  # contributions accounted


def _loop(truth, model, hours, commands, obs, seed=3):
    """Run `truth` with fixed unit commands and feed the observer, as the integration would, every minute."""
    w = coupled.Weather(START, 2, mean=50, seed=seed)
    sim = coupled.CoupledSim(truth, START, w, seed=seed, temps={r: 66.0 for r in truth.rooms})
    for uid, (mode, sp) in commands.items():
        sim.command(uid, mode, sp)
    for _ in range(int(hours * 60)):
        temps = {r: sim.reading(r) for r in truth.rooms}
        actual = {u: {"mode": sim.mode[u], "setpoint": sim.sp[u], "running": sim.action(u) in ("heating", "cooling"),
                      "fan": sim.fan[u]} for u in truth.units}
        obs.update(model, sim.now, temps, actual, w.outdoor(sim.now), w.sun(sim.now), dict(sim.effect))
        sim.advance(1)
    return sim


def test_observer_stays_quiet_when_the_model_is_right():
    m = coupled.the_house()
    obs = observer.Observer()
    _loop(m, m, 4, {"air_handler": ("heat", 70.0), "minisplit_1": ("heat", 60.0), "minisplit_2": ("heat", 60.0)}, obs)
    assert all(abs(v) < 0.4 for v in obs.offsets.values()), obs.offsets
    assert all(0.7 < v < 1.4 for v in obs.leak_scale.values()), obs.leak_scale
    assert all(0.7 < s < 1.4 for by in obs.effect_scale.values() for s in by.values()), obs.effect_scale


def test_observer_attributes_each_error_to_its_cause():
    m = coupled.the_house()
    # the kitchen warms at 40% of the modelled rate, and the leak is twice as big
    d = disturbed.Disturbance(leak_mult=2.0, kitchen_gain=0.4)
    truth = disturbed.perturbed_truth(m, d)
    obs = observer.Observer(leak_on=True)
    # the air handler heats (the kitchen and Bedroom 3's leak both have a cause acting); Primary and Bedroom 3 idle
    _loop(truth, m, 6, {"air_handler": ("heat", 72.0), "minisplit_1": ("heat", 60.0), "minisplit_2": ("heat", 60.0)}, obs)
    assert 1.5 < obs.leak_scale.get("bedroom_3", 1.0) < 2.8, obs.leak_scale
    assert 0.25 < obs.effect_scale.get("kitchen", {}).get("air_handler", 1.0) < 0.65, obs.effect_scale
    # the gain error must not be smeared into a constant drift that would also apply while the unit is off
    assert abs(obs.offsets.get("kitchen", 0.0)) < 0.2, obs.offsets
    assert all(abs(v) <= obs.d_max + 1e-9 for v in obs.offsets.values())
    assert all(obs.s_min - 1e-9 <= v <= obs.s_max + 1e-9 for v in obs.leak_scale.values())


def test_observer_learns_a_passive_drift_only_while_nothing_acts():
    m = coupled.the_house()
    truth = disturbed.perturbed_truth(m, disturbed.Disturbance(bias={"dining_area": -0.7}))
    obs = observer.Observer()
    _loop(truth, m, 5, {"air_handler": ("off", None), "minisplit_1": ("off", None), "minisplit_2": ("off", None)}, obs)
    assert -0.95 < obs.offsets.get("dining_area", 0.0) < -0.4, obs.offsets
    assert all(abs(v - 1) < 0.05 for v in obs.leak_scale.values()), obs.leak_scale  # no cause acted: no gain learned
    assert all(abs(v - 1) < 0.05 for by in obs.effect_scale.values() for v in by.values()), obs.effect_scale


def test_observer_state_round_trips():
    o = observer.Observer()
    o.offsets, o.leak_scale, o.effect_scale = {"kitchen": -0.5}, {"bedroom_3": 1.8}, {"kitchen": {"air_handler": 0.4}}
    p = observer.Observer()
    p.load_state(o.to_state())
    assert p.offsets == {"kitchen": -0.5} and p.leak_scale == {"bedroom_3": 1.8}
    assert p.effect_scale == {"kitchen": {"air_handler": 0.4}}
    assert p.corrections() == {"offsets": {"kitchen": -0.5}, "leak_scale": {"bedroom_3": 1.8},
                               "effect_scale": {"kitchen": {"air_handler": 0.4}}}
    m = coupled.the_house()  # a learned observer round-trips with its regression sums, so a restart doesn't forget
    truth = disturbed.perturbed_truth(m, disturbed.Disturbance(kitchen_gain=0.4))
    a = observer.Observer()
    _loop(truth, m, 3, {"air_handler": ("heat", 72.0), "minisplit_1": ("off", None), "minisplit_2": ("off", None)}, a)
    b = observer.Observer()
    b.load_state(a.to_state())
    assert b.stats.keys() == a.stats.keys() and b.stats["kitchen"]["keys"] == a.stats["kitchen"]["keys"]
    assert abs(b.stats["kitchen"]["A"][0][0] - a.stats["kitchen"]["A"][0][0]) < 1e-3


def test_switch_margin_keeps_the_house_in_its_mode_unless_the_other_is_clearly_better():
    truth = coupled.the_house()
    cfg = {"rooms": [{"id": r, "unit": u, "priority": p, "default_temp": 67} for r, u, p in (
        ("kitchen", "air_handler", 2), ("living_room", "air_handler", 1), ("primary_bedroom", "minisplit_1", 5),
        ("bedroom_3", "minisplit_2", 4))], "units": [{"id": u} for u in truth.units], "settings": {"floor": 63.0}}
    rooms, units = coupled.rooms_units_from_config(cfg, truth)
    s = coupled.settings_from_config(cfg)
    now = START + timedelta(hours=12)
    house = core.HouseState("heat", now - timedelta(hours=3))
    temps = {r: 72.0 for r in truth.rooms}  # hot everywhere: cooling is clearly better than staying in heat
    kw = dict(house=house, setpoints_now={}, running={}, outdoor_seq=[55.0], sun_seq=[0.0], settings=s)
    free = optimizer.plan(truth, rooms, units, temps, now, ps=optimizer.PlanSettings(switch_margin=0.0), **kw)
    held = optimizer.plan(truth, rooms, units, temps, now, ps=optimizer.PlanSettings(switch_margin=0.999), **kw)
    assert free.mode == "cool" and held.mode == "heat", (free.mode, held.mode)


def _cycle_house():
    nominal = coupled.the_house()
    cfg = {"rooms": [{"id": r, "unit": u, "priority": p, "default_temp": 67} for r, u, p in (
        ("kitchen", "air_handler", 2), ("living_room", "air_handler", 1), ("bedroom_1", "air_handler", 1),
        ("primary_bedroom", "minisplit_1", 5), ("bedroom_3", "minisplit_2", 5))],
        "units": [{"id": u} for u in nominal.units], "settings": {"floor": 63.0}}
    rooms, units = coupled.rooms_units_from_config(cfg, nominal)
    return nominal, rooms, units, coupled.settings_from_config(cfg)


def test_no_limit_cycle_when_the_leak_is_much_bigger_than_modelled():
    """The heat/cool cycle of 2026-10-17..22: a heat run lifts Bedroom 3 through the leak, the house flips to cool, the
    main floor cools, it flips back. The planner as first deployed hunted like this in simulation; with the observer and
    the switch margin/confirmation the house must hold a mode for the evening and night, with the leak 2.6x the model's
    (and spiking as the compressor starts), a poorly heated kitchen and cold weather."""
    nominal, rooms, units, s = _cycle_house()
    d = disturbed.Disturbance(leak_mult=2.6, leak_boost=1.2, leak_tau_min=20, kitchen_gain=0.4, b1_evening=1.0, ou_sigma=0.3)
    truth = disturbed.perturbed_truth(nominal, d)
    start = datetime.fromisoformat("2026-10-19T17:00:00-07:00")
    w = coupled.Weather(start, 2, mean=45, seed=11)
    sim = disturbed.DisturbedSim(truth, start, w, d, seed=11, temps={r: 66.5 for r in truth.rooms})
    ctl = coupled.MPCController(rooms, units, nominal, s, ps=optimizer.PlanSettings(switch_margin=0.15),
                                observer=observer.Observer(), confirm_plans=3)
    res = coupled.run(truth, rooms, units, ctl, start, 14 / 24, w, s, sim=sim)
    assert res["mode_switches"] <= 2, res
    assert all(abs(v) <= ctl.observer.d_max + 1e-9 for v in ctl.observer.offsets.values())
    assert all(ctl.observer.s_min - 1e-9 <= x <= ctl.observer.s_max + 1e-9
               for by in ctl.observer.effect_scale.values() for x in by.values())


if __name__ == "__main__":
    import traceback
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for f in fns:
        try:
            f()
        except Exception:
            bad += 1
            print(f.__name__)
            traceback.print_exc()
    print(f"{len(fns) - bad} passed, {bad} failed")
    sys.exit(1 if bad else 0)
