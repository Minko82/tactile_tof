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
    assert kf.update(200.0, DT) == 200.0          # re-initialises on next sample


def test_invalid_params_raise():
    for bad in (dict(process_accel_psd=0.0), dict(process_accel_psd=-1.0), dict(measurement_var=0.0)):
        try:
            KalmanCV(**bad); assert False, "expected ValueError"
        except ValueError:
            pass
    kf = KalmanCV()
    try:
        kf.update(100.0, -0.1); assert False, "expected ValueError for dt<0"
    except ValueError:
        pass


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
    assert kf.rejected
    assert abs(kf.x - before) < 5.0                 # estimate barely moves


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


# ---- causality / determinism -----------------------------------------------
def test_deterministic_and_causal():
    seq = [100.0, 102.0, 98.0, 105.0, 110.0, 108.0]
    out_a = [KalmanCV().update(z, DT) for z in seq]  # (fresh filter each -> only first sample)
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
