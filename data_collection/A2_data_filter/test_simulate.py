"""Unit tests for the deterministic Kalman-filter simulation."""
import math
import os
import tempfile
from dataclasses import replace

import numpy as np

from simulate import (MOTIONS, SensorData, SimulationConfig, build_parser,
                      calculate_metrics, fake_sensor, ground_truth, main,
                      noise_description, run_all, run_scenario, save_plot)


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(f"expected {exc_type.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"expected {exc_type.__name__}")


def test_config_matches_measurement_variance_by_default():
    assert SimulationConfig(noise_std_mm=5.0).resolved_measurement_var == 25.0
    assert SimulationConfig(noise_std_mm=5.0, measurement_var=9.0).resolved_measurement_var == 9.0
    assert SimulationConfig(noise_std_mm="4").resolved_measurement_var == 16.0


def test_config_rejects_invalid_values():
    bad = (
        dict(fps=0), dict(duration_s=-1), dict(noise_std_mm=0),
        dict(outlier_rate=-0.1), dict(outlier_rate=1.1),
        dict(dropout_rate=float("nan")), dict(outlier_mm=-1),
        dict(process_accel_psd=float("inf")), dict(measurement_var=0),
        dict(maneuver_accel_psd=100, process_accel_psd=500),
        dict(reject_sigma=0), dict(max_consec_reject=0),
        dict(warmup_s=-1), dict(settle_s=-1),
    )
    for values in bad:
        assert_raises(ValueError, SimulationConfig, **values)


def test_ground_truth_motions():
    t = np.array([0.0, 1.5, 3.0, 4.5])
    assert np.all(ground_truth("static", t) == 150.0)
    assert np.allclose(ground_truth("ramp", t), 50.0 + 60.0 * t)
    assert np.array_equal(ground_truth("steps", t), [60.0, 160.0, 300.0, 120.0])
    assert np.all(np.isfinite(ground_truth("sine", t)))
    assert_raises(ValueError, ground_truth, "unknown", t)


def test_fake_sensor_is_deterministic_and_exposes_masks():
    truth = np.full(100, 100.0)
    config = SimulationConfig(outlier_rate=1.0, dropout_rate=0.0, noise_std_mm=1.0)
    a = fake_sensor(truth, np.random.default_rng(12), config)
    b = fake_sensor(truth, np.random.default_rng(12), config)
    assert np.array_equal(a.readings, b.readings)
    assert a.outlier.all() and not a.dropped.any()
    assert np.all(np.abs(a.readings - truth) > 50.0)

    dropped = fake_sensor(truth, np.random.default_rng(1),
                          SimulationConfig(dropout_rate=1.0))
    assert dropped.dropped.all() and np.isnan(dropped.readings).all()


def test_metric_formulas_on_known_signal():
    config = SimulationConfig(fps=10.0, duration_s=0.4, noise_std_mm=1.0,
                              outlier_rate=0.0, dropout_rate=0.0,
                              warmup_s=0.0, settle_s=0.0)
    t = np.arange(4) / config.fps
    truth = np.full(4, 100.0)
    readings = np.array([101.0, 99.0, 101.0, 99.0])
    sensor = SensorData(readings, np.zeros(4, bool), np.zeros(4, bool))
    estimate = np.array([100.5, 99.5, 100.5, 99.5])
    metrics = calculate_metrics(t, truth, sensor, estimate, config)
    assert math.isclose(metrics.raw_sigma, 1.0)
    assert math.isclose(metrics.filtered_sigma, 0.5)
    assert math.isclose(metrics.reduction, 2.0)
    assert math.isclose(metrics.removed_pct, 50.0)
    assert math.isclose(metrics.raw_rmse, 1.0)
    assert math.isclose(metrics.filtered_rmse, 0.5)
    assert math.isnan(metrics.coast_rmse) and math.isnan(metrics.lag_ms)


def test_noise_description_distinguishes_improvement_from_regression():
    results = run_all(SimulationConfig(), seed=7)
    sine = results[MOTIONS.index("sine")]
    assert "noise removed" in noise_description(sine)
    assert "more residual noise" in noise_description(replace(sine, removed_pct=-10.0))


def test_metric_inputs_require_matching_shapes():
    config = SimulationConfig(warmup_s=0.0, settle_s=0.0)
    sensor = SensorData(np.ones(3), np.zeros(3, bool), np.zeros(3, bool))
    assert_raises(ValueError, calculate_metrics, np.arange(2), np.ones(2), sensor,
                  np.ones(2), config)


def test_scenarios_are_deterministic_and_order_independent():
    config = SimulationConfig()
    first = run_scenario("static", config, seed=7)
    run_scenario("ramp", config, seed=7)
    again = run_scenario("static", config, seed=7)
    assert np.array_equal(first.noisy, again.noisy, equal_nan=True)
    assert np.array_equal(first.estimate, again.estimate, equal_nan=True)
    assert first.raw_sigma == again.raw_sigma
    assert first.filtered_sigma == again.filtered_sigma
    assert first.reduction == again.reduction


def test_default_scenarios_report_expected_behavior():
    config = SimulationConfig()
    results = run_all(config, seed=7)
    assert tuple(result.motion for result in results) == MOTIONS
    for result in results:
        for value in (result.raw_sigma, result.filtered_sigma, result.reduction,
                      result.removed_pct, result.raw_rmse, result.filtered_rmse,
                      result.final_measurement_var):
            assert math.isfinite(value)
    static = results[MOTIONS.index("static")]
    ramp = results[MOTIONS.index("ramp")]
    assert static.reduction >= 1.4                  # 10 Hz VL53L5CX 8x8 default
    assert math.isnan(static.lag_ms)
    assert abs(ramp.lag_ms) <= 1000.0 / config.fps
    sine = results[MOTIONS.index("sine")]
    assert sine.filtered_sigma < sine.raw_sigma


def test_cli_options_and_no_plot_mode():
    args = build_parser().parse_args(["--hz", "15", "--noise-std", "8",
                                      "--measurement-var", "25",
                                      "--process-accel-psd", "800",
                                      "--maneuver-accel-psd", "8000",
                                      "--adaptive-r", "--seed", "12", "--no-plot"])
    assert args.fps == 15.0 and args.noise_std_mm == 8.0
    assert args.measurement_var == 25.0 and args.seed == 12
    assert args.process_accel_psd == 800.0 and args.maneuver_accel_psd == 8000.0
    assert args.adaptive_r
    with tempfile.TemporaryDirectory() as folder:
        output = os.path.join(folder, "must-not-exist.png")
        results = main(["--seed", "12", "--output", output, "--no-plot"])
        assert len(results) == 4 and not os.path.exists(output)


def test_plot_is_written_to_requested_path():
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    results = run_all(SimulationConfig(), seed=7)
    with tempfile.TemporaryDirectory() as folder:
        output = os.path.join(folder, "simulation.png")
        assert save_plot(results, output) == os.path.abspath(output)
        assert os.path.getsize(output) > 1000


if __name__ == "__main__":
    import sys
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"  PASS  {name}"); passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}"); failed += 1
    print(f"\n{passed} passed, {failed} failed  ({len(tests)} tests)")
    sys.exit(1 if failed else 0)
