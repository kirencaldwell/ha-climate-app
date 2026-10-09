import math, os, random, sys
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_components", "room_climate"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
import core, predict
from core import Room, Unit, Settings, SystemState
from house import SimHouse
from predict import RoomModel, Site

NYC = Site(40.7, -74.0, -5)
JUN21 = datetime(2026, 6, 21)
T0 = datetime(2026, 1, 12, 12, 0)


def test_sun_position_and_times_match_known_values():
    elev, az = predict.solar_position(NYC, JUN21.replace(hour=12, minute=0))
    assert 70 < elev < 76 and 170 < az < 195  # near solar noon, high and due south
    rise, sset = predict.sun_times(NYC, JUN21)
    assert abs(rise.hour * 60 + rise.minute - (4 * 60 + 25)) < 15  # 05:25 EDT = 04:25 EST
    assert abs(sset.hour * 60 + sset.minute - (19 * 60 + 30)) < 15
    r2, s2 = predict.sun_times(NYC, datetime(2026, 12, 21))
    assert (s2 - r2) < (sset - rise) - timedelta(hours=4)


def test_aware_datetimes_are_converted_to_utc():
    local = datetime(2026, 6, 21, 12, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert predict.solar_position(NYC, local) == predict.solar_position(NYC, JUN21.replace(hour=12))


def test_windows_gain_sun_when_facing_it():
    west = [{"facing": 270}]
    am, pm = predict.sun_factor(west, NYC, JUN21.replace(hour=8)), predict.sun_factor(west, NYC, JUN21.replace(hour=16))
    assert am == 0.0 and pm > 0.4
    assert predict.sun_factor([{"facing": 90}], NYC, JUN21.replace(hour=8)) > 0.3
    assert predict.sun_factor(west, NYC, JUN21.replace(hour=2)) == 0.0  # night
    assert predict.sun_factor([], NYC, JUN21.replace(hour=16)) == 0.0
    assert predict.sun_factor(west, NYC, JUN21.replace(hour=16), cloud=1.0) < pm * 0.3


def _truth_run(model, days, a=0.003, bh=0.06, s=0.03, windows=({"facing": 270},), seed=1):
    rnd = random.Random(seed)
    t, temp, heating = T0, 68.0, 0
    for i in range(days * 24 * 12):
        out = 35 + 12 * math.sin(2 * math.pi * (t.hour + t.minute / 60 - 9) / 24)
        sun = predict.sun_factor(list(windows), NYC, t)
        model.update(t, temp, out, heating, sun)
        temp += 5 * (a * (out - temp) + bh * (1 if heating else 0) + s * sun)
        if i % 6 == 0:  # excite the heater now and then
            heating = 1 if (temp < 68.5 or rnd.random() < 0.3) and temp < 71 else 0
        t += timedelta(minutes=5)
    return model


def test_model_learns_loss_heat_and_sun_terms():
    m = _truth_run(RoomModel(), 6)
    assert m.ready
    assert abs(m.a - 0.003) < 0.0008
    assert abs(m.s - 0.03) < 0.012
    assert abs(m.b_heat - 0.06) < 0.02


def test_model_not_ready_until_enough_history_and_ignores_fast_updates():
    m = RoomModel(min_samples=10)
    for i in range(30):  # one reading per minute: only every 5th is used
        m.update(T0 + timedelta(minutes=i), 68.0, 40, 0, 0.0)
    assert m.n == 5 and not m.ready
    m.update(T0 + timedelta(hours=3), 68.0, 40, 0, 0.0)  # gap > 30 min restarts differencing
    assert m.n == 5


def test_prediction_drifts_toward_outdoor_and_rises_with_sun():
    m = RoomModel()
    m.theta = [0.03, 0.05, 0.05, 0.02]  # a=0.003/min, s=0.02
    cold = predict.predict_temp(m, 68, T0.replace(hour=2), 120, lambda t: 20.0, [], NYC)
    assert abs(cold - (20 + 48 * math.exp(-0.003 * 120))) < 0.5
    no_sun = predict.predict_temp(m, 68, JUN21.replace(hour=13), 120, lambda t: 68.0, [], NYC)
    sunny = predict.predict_temp(m, 68, JUN21.replace(hour=13), 120, lambda t: 68.0, [{"facing": 270}], NYC)
    assert no_sun == 68 and sunny > 69


def test_persistence_roundtrip_and_interpolator():
    m = _truth_run(RoomModel(), 2)
    m2 = RoomModel.from_dict(m.to_dict())
    assert m2.theta == m.theta and m2.n == m.n
    f = predict.interpolator([(T0, 30.0), (T0 + timedelta(hours=2), 40.0)], 0)
    assert f(T0 + timedelta(hours=1)) == 35.0 and f(T0 - timedelta(hours=1)) == 30.0 and f(T0 + timedelta(hours=5)) == 40.0
    assert predict.interpolator([], 55.0)(T0) == 55.0


# --- control using predictions -------------------------------------------------------------------

def _one(preds, temp, state=None, now=T0, settings=Settings()):
    r = Room("r", "R", "u", default_temp=68)
    return core.decide_system([Unit("u", "c")], [r], {"r": (temp, now)}, {}, state or SystemState(), now, settings, preds)


def test_no_predictions_is_unchanged_behaviour():
    for temp in (66.5, 67.5, 68.0, 69.5):
        assert _one(None, temp)[1].mode == _one({}, temp)[1].mode


def test_predicted_drop_starts_heating_before_room_is_cold():
    assert _one(None, 68.0)[1].mode == "off"
    cmds, st = _one({"r": 63.0}, 68.0)  # predicted 5F under target in an hour
    assert st.mode == "heat" and cmds["u"].mode == "heat"


def test_small_predicted_drift_inside_band_does_nothing():
    assert _one({"r": 67.0}, 68.0)[1].mode == "off"


def test_predicted_warming_does_not_suppress_heat_when_already_far_off():
    assert _one({"r": 69.0}, 64.0)[1].mode == "heat"  # |error| >= guard: act on the present


def test_predicted_rise_starts_cooling_early():
    cmds, st = _one({"r": 73.0}, 68.0)
    assert st.mode == "cool"


def test_heating_runs_a_little_past_target_when_a_drop_is_predicted():
    _, st = _one({"r": 63.0}, 66.5)
    later = T0 + timedelta(minutes=20)
    # At target but predicted to fall: keep going. Without a prediction it would stop here.
    assert _one({"r": 63.0}, 68.2, st, later)[1].mode == "heat"
    assert _one(None, 68.2, st, later)[1].mode == "off"
    # Past the pre-heat allowance (+0.5F) it stops.
    assert _one({"r": 63.0}, 68.8, st, later)[1].mode == "off"


def test_prediction_never_causes_active_cooling_for_a_setback():
    # Drift-only: a scheduled drop in target must not make a room that is on target look "hot".
    r = Room("r", "R", "u", default_temp=68, schedule=[core.ScheduleEntry(list(range(7)), "07:00", 68), core.ScheduleEntry(list(range(7)), "22:00", 62)])
    now = T0.replace(hour=21, minute=50)
    cmds, st = core.decide_system([Unit("u", "c")], [r], {"r": (68.0, now)}, {}, SystemState(), now, Settings(), {"r": 67.8})
    assert st.mode == "off"


# --- end to end in the simulator ------------------------------------------------------------------

def _front(predictive):
    start = datetime(2026, 1, 10)
    front = start + timedelta(days=4, hours=14)

    def out(t):
        d = min(max((t - front).total_seconds() / (3 * 3600), 0), 1)
        return 40 + 10 * math.sin(2 * math.pi * (t.hour + t.minute / 60 - 9) / 24) - 30 * d
    r = Room("w", "W", "u", default_temp=68, windows=[{"facing": 270}])
    h = SimHouse([r], [Unit("u", "c")], out, start, shared_mode=True, power=0.15, predictive=predictive, site=NYC)
    h.sim["w"].sun_gain = 0.02
    lo = hi = 0.0
    for m in range(6 * 24 * 60):
        h.step(1)
        if m >= 4 * 24 * 60 + 12 * 60:
            d = h.sim["w"].temp - 68
            lo, hi = min(lo, d), max(hi, d)
    return h, lo, hi


def test_sim_cold_front_is_held_closer_to_target_with_prediction():
    _, lo_r, hi_r = _front(False)
    h, lo_p, hi_p = _front(True)
    assert h.models["w"].ready
    assert lo_p > lo_r + 0.3          # noticeably less undershoot
    assert max(abs(lo_p), hi_p) < max(abs(lo_r), hi_r) - 0.3  # and worst case overall is better
    assert hi_p <= 1.0                # without wild overshoot


def test_sim_wrong_forecast_degrades_gracefully():
    h = SimHouse([Room("w", "W", "u", default_temp=68)], [Unit("u", "c")], lambda t: 35.0,
                 datetime(2026, 1, 10), shared_mode=True, power=0.15, predictive=True, site=NYC, forecast_error=15.0)
    worst = 0.0
    for m in range(5 * 24 * 60):
        h.step(1)
        if m > 2 * 24 * 60:
            worst = max(worst, abs(h.sim["w"].temp - 68))
    assert worst < 3.0
