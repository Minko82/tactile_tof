"""
Unit tests for the live Kalman filter (live_filter.KalmanCV / LiveFilter).

Runs WITHOUT pytest:   ~/ur5-env/bin/python3 test_live_filter.py
Or under pytest:       pytest test_live_filter.py

Each test asserts one property of the filter: initialisation, convergence, noise
reduction, lag-free ramp tracking, outlier rejection, step re-lock, dropout
coasting, covariance behaviour, causality/determinism, validation, and speed.
"""
import math
import time
import numpy as np
from live_filter import KalmanCV, LiveFilter, stream

DT = 1.0 / 30.0


def assert_raises(exc_type, fn, *args, **kwargs):
    """Small assertion helper that keeps the suite runnable without pytest."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(f"expected {exc_type.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"expected {exc_type.__name__}")


# ---- initialisation ---------------------------------------------------------
def test_initialises_on_first_sample():
    kf = KalmanCV()
    assert not kf.initialized
    assert kf.update(100.0, DT) == 100.0          # first estimate == first measurement
    assert kf.initialized


def test_reset_returns_to_uninitialised():
    kf = KalmanCV()
    kf.update(100.0, DT)
    kf.reset()
    assert not kf.initialized
    assert math.isnan(kf.x)
    assert kf.update(200.0, DT) == 200.0          # re-initialises on next sample


def test_invalid_before_initialisation_returns_nan():
    kf = KalmanCV()
    for z in (None, float("nan"), float("inf"), float("-inf")):
        assert math.isnan(kf.update(z, DT))
        assert not kf.initialized and kf.coasting
    assert kf.update(42.0, DT) == 42.0


def test_invalid_params_raise():
    bad_configs = (
        dict(process_accel_psd=0.0), dict(process_accel_psd=float("nan")),
        dict(measurement_var=0.0), dict(measurement_var=float("inf")),
        dict(reject_sigma=0.0), dict(reject_sigma=-1.0),
        dict(max_consec_reject=-1), dict(max_consec_reject=1.5),
        dict(max_coast_s=-1.0), dict(max_coast_s=float("nan")),
        dict(init_vel_var=-1.0), dict(init_vel_var=float("inf")),
        dict(maneuver_accel_psd=100.0), dict(maneuver_hold_s=-1.0),
        dict(maneuver_consistency_sigma=0.0), dict(measurement_adapt_rate=0.0),
        dict(measurement_adapt_rate=1.1), dict(measurement_var_min=0.0),
        dict(measurement_var_min=10.0, measurement_var_max=1.0),
    )
    for bad in bad_configs:
        assert_raises(ValueError, KalmanCV, **bad)
    KalmanCV(reject_sigma=None, max_consec_reject=0, max_coast_s=0.0, init_vel_var=0.0)

    kf = KalmanCV()
    for bad_dt in (-0.1, float("nan"), float("inf"), "bad"):
        assert_raises(ValueError, kf.update, 100.0, bad_dt)


# ---- convergence / noise ----------------------------------------------------
def test_converges_to_constant():
    kf = KalmanCV()
    for _ in range(300):
        kf.update(50.0, DT)
    assert abs(kf.x - 50.0) < 0.1                  # locks onto the true constant


def test_reduces_noise_on_constant():
    rng = np.random.default_rng(0)
    kf = KalmanCV()
    est = np.array([kf.update(100.0 + rng.normal(0, 5.0), DT) for _ in range(600)])
    assert np.std(est[100:] - 100.0) < 0.6 * 5.0   # filtered σ well under raw σ (5)


def test_reduces_noise_while_moving():
    rng = np.random.default_rng(1)
    kf = KalmanCV()
    raw, est, truth = [], [], []
    for i in range(600):
        x = 100.0 + 30.0 * i * DT
        z = x + rng.normal(0, 5.0)
        raw.append(z); est.append(kf.update(z, DT)); truth.append(x)
    raw, est, truth = (np.array(a)[100:] for a in (raw, est, truth))
    assert np.std(est - truth) < np.std(raw - truth)   # denoises even on a ramp


# ---- lag-free tracking ------------------------------------------------------
def test_tracks_ramp_without_lag():
    kf = KalmanCV()
    v_true = 30.0
    err = []
    for i in range(400):
        x = 100.0 + v_true * i * DT
        e = kf.update(x, DT)                        # noise-free ramp
        err.append(abs(e - x))
    assert max(err[150:]) < 1.0                     # no steady-state lag
    assert abs(kf.velocity - v_true) < 1.0          # velocity estimate correct


# ---- robustness -------------------------------------------------------------
def test_rejects_isolated_outlier():
    kf = KalmanCV()
    for _ in range(100):
        kf.update(100.0, DT)                        # settle (confident)
    before = kf.x
    kf.update(500.0, DT)                            # one gross spike
    assert kf.rejected and kf.downweighted and not kf.coasting
    assert abs(kf.x - before) < 5.0                 # estimate barely moves


def test_two_consistent_large_measurements_confirm_maneuver():
    kf = KalmanCV(measurement_var=25.0)
    for _ in range(100):
        kf.update(100.0, DT)
    first = kf.update(300.0, DT)
    assert kf.downweighted and not kf.maneuver and first < 120.0
    second = kf.update(301.0, DT)
    assert kf.maneuver and not kf.downweighted
    assert abs(second - 300.5) < 1e-9
    assert kf.active_process_accel_psd == kf.q_maneuver


def test_external_maneuver_hint_allows_first_frame_relock():
    kf = KalmanCV(measurement_var=25.0)
    for _ in range(100):
        kf.update(100.0, DT)
    estimate = kf.update(300.0, DT, maneuver_hint=True)
    assert estimate == 300.0 and kf.maneuver
    assert not kf.downweighted


def test_inconsistent_large_measurements_stay_downweighted():
    kf = KalmanCV(measurement_var=25.0)
    for _ in range(100):
        kf.update(100.0, DT)
    kf.update(300.0, DT)
    kf.update(-100.0, DT)
    assert kf.downweighted and not kf.maneuver
    assert 90.0 < kf.x < 110.0


def test_relocks_after_sustained_step():
    kf = KalmanCV()
    for _ in range(100):
        kf.update(100.0, DT)                        # settle at 100
    for _ in range(15):
        kf.update(300.0, DT)                        # a real 200 mm step, sustained
    assert abs(kf.x - 300.0) < 5.0                  # re-locks (not rejected forever)


def test_coasts_through_dropout():
    kf = KalmanCV()
    for i in range(100):
        kf.update(100.0 + 30.0 * i * DT, DT)        # moving
    x0, v0 = kf.x, kf.velocity
    kf.update(None, DT)                             # dropped frame
    assert kf.coasting
    assert abs(kf.x - (x0 + v0 * DT)) < 1e-6         # predicts forward with velocity
    assert math.isfinite(kf.x)


def test_nan_treated_as_dropout():
    kf = KalmanCV()
    kf.update(100.0, DT)
    kf.update(float("nan"), DT)                     # must not crash / go NaN
    assert math.isfinite(kf.x)


def test_all_nonfinite_measurements_coast_without_poisoning_state():
    kf = KalmanCV()
    kf.update(100.0, DT)
    for z in (float("nan"), float("inf"), float("-inf"), None):
        kf.update(z, DT)
        assert math.isfinite(kf.x) and kf.coasting


def test_long_dropout_resets_to_no_estimate():
    kf = KalmanCV(max_coast_s=0.01)
    kf.update(100.0, 0.0)
    assert math.isnan(kf.update(None, 0.02))
    assert not kf.initialized


# ---- covariance -------------------------------------------------------------
def test_variance_below_measurement_after_settling():
    kf = KalmanCV()
    for _ in range(200):
        kf.update(100.0, DT)
    assert 0.0 < kf.variance < kf.r                 # fusing beats a single measurement


def test_variance_grows_when_coasting():
    kf = KalmanCV()
    for _ in range(100):
        kf.update(100.0, DT)
    settled = kf.variance
    for _ in range(5):
        kf.update(None, DT)                         # no measurements
    assert kf.variance > settled                    # uncertainty grows without data


def test_gain_in_unit_interval():
    kf = KalmanCV()
    for _ in range(50):
        kf.update(100.0, DT)
    assert 0.0 <= kf.gain <= 1.0


def test_joseph_covariance_stays_symmetric_and_nonnegative():
    kf = KalmanCV()
    for i in range(300):
        z = 100.0 + math.sin(i / 10.0) * 10.0
        if i in (80, 160):
            z += 400.0
        kf.update(z, DT)
        assert abs(kf.P[0][1] - kf.P[1][0]) < 1e-8
        assert kf.P[0][0] >= 0.0 and kf.P[1][1] >= 0.0
        assert kf.P[0][0] * kf.P[1][1] - kf.P[0][1] * kf.P[1][0] >= -1e-8


def test_per_measurement_variance_controls_update_strength():
    low_r = KalmanCV(reject_sigma=None)
    high_r = KalmanCV(reject_sigma=None)
    for _ in range(100):
        low_r.update(100.0, DT)
        high_r.update(100.0, DT)
    before = low_r.x
    after_low = low_r.update(105.0, DT, measurement_var=1.0)
    after_high = high_r.update(105.0, DT, measurement_var=100.0)
    assert after_low - before > after_high - before > 0.0
    assert low_r.effective_measurement_var == 1.0
    assert high_r.effective_measurement_var == 100.0


def test_adaptive_measurement_variance_learns_larger_noise():
    rng = np.random.default_rng(4)
    kf = KalmanCV(measurement_var=1.0, reject_sigma=None,
                  adapt_measurement_var=True, measurement_adapt_rate=0.02)
    for _ in range(1000):
        kf.update(100.0 + rng.normal(0.0, 5.0), DT)
    assert 2.0 < kf.r < 100.0


# ---- causality / determinism -----------------------------------------------
def test_deterministic_and_causal():
    seq = [100.0, 102.0, 98.0, 105.0, 110.0, 108.0]
    kf1 = KalmanCV(); full = [kf1.update(z, DT) for z in seq]
    kf2 = KalmanCV(); again = [kf2.update(z, DT) for z in seq]
    assert full == again                             # deterministic
    kf3 = KalmanCV(); prefix = [kf3.update(z, DT) for z in seq[:3]]
    assert prefix == full[:3]                        # causal: no dependence on future samples


def test_livefilter_handles_timestamps_and_duplicates():
    lf = LiveFilter()
    lf.update(100.0, 0.000)
    lf.update(102.0, 0.033)
    lf.update(104.0, 0.033)                          # duplicate timestamp -> dt=0, must not crash
    assert math.isfinite(lf.variance) and lf.variance >= 0.0


def test_livefilter_accepts_vl53l5cx_per_sample_variance():
    lf = LiveFilter()
    lf.update(100.0, 0.0, measurement_var=9.0)
    lf.update(101.0, DT, measurement_var=16.0)
    assert lf.kf.effective_measurement_var == 16.0


def test_livefilter_forwards_maneuver_hint():
    lf = LiveFilter(measurement_var=25.0)
    for i in range(20):
        lf.update(100.0, i * DT)
    assert lf.update(250.0, 20 * DT, maneuver_hint=True) == 250.0
    assert lf.maneuver


def test_livefilter_rejects_bad_timestamps_without_advancing_clock():
    lf = LiveFilter()
    lf.update(100.0, 1.0)
    for bad_t in (0.9, float("nan"), float("inf"), "bad"):
        assert_raises(ValueError, lf.update, 101.0, bad_t)
    assert math.isfinite(lf.update(102.0, 1.1))


def test_stream_helper_matches_manual():
    zs = [(i * DT, 100.0 + i) for i in range(20)]
    via_stream = [e for _, _, e in stream(iter(zs))]
    lf = LiveFilter(); manual = [lf.update(z, t) for t, z in zs]
    assert via_stream == manual


# ---- performance ------------------------------------------------------------
def test_realtime_speed():
    kf = KalmanCV()
    n = 20000
    t0 = time.perf_counter()
    for i in range(n):
        kf.update(100.0 + (i % 10), DT)
    per_us = (time.perf_counter() - t0) / n * 1e6
    assert per_us < 200.0, f"{per_us:.1f} us/update too slow"   # « a 15 Hz frame (66,667 us)


# ---- standalone harness (no pytest needed) ----------------------------------
if __name__ == "__main__":
    import sys
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"  PASS  {name}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}"); failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed  ({len(tests)} tests)")
    sys.exit(1 if failed else 0)
