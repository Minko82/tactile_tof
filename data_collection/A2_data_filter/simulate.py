"""Simulate a noisy distance sensor and evaluate the live Kalman filter.

The four deterministic motions cover steady, constant-velocity, discontinuous,
and accelerating targets. Measurements contain Gaussian noise plus configurable
outliers and dropped frames. The default Kalman measurement variance is matched
to the simulated Gaussian variance; override it to explore deliberate mistuning.

Examples:
    python3 simulate.py
    python3 simulate.py --hz 15
    python3 simulate.py --noise-std 8 --measurement-var 25
    python3 simulate.py --seed 12 --no-plot

By default the script prints per-motion metrics and writes figs/simulation.png.
See KALMAN_FILTER_THEORY.md for the model, VL53L5CX mapping, and tuning guide.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os

import numpy as np

from live_filter import LiveFilter


HERE = os.path.dirname(os.path.abspath(__file__))
MOTIONS = ("static", "ramp", "steps", "sine")


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters shared by signal generation and the Kalman filter."""

    fps: float = 10.0
    duration_s: float = 15.0
    noise_std_mm: float = 5.0
    outlier_rate: float = 0.02
    outlier_mm: float = 60.0
    dropout_rate: float = 0.03
    process_accel_psd: float = 500.0
    maneuver_accel_psd: float = 5000.0
    reject_sigma: float = 5.0
    max_consec_reject: int = 1
    adapt_measurement_var: bool = False
    measurement_var: float | None = None
    warmup_s: float = 0.5
    settle_s: float = 0.3

    def __post_init__(self):
        for name in ("fps", "duration_s", "noise_std_mm", "process_accel_psd",
                     "maneuver_accel_psd", "reject_sigma"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)
        if self.maneuver_accel_psd < self.process_accel_psd:
            raise ValueError("maneuver_accel_psd must be >= process_accel_psd")
        if not isinstance(self.max_consec_reject, int) or self.max_consec_reject < 1:
            raise ValueError("max_consec_reject must be an integer >= 1")
        for name in ("outlier_mm", "warmup_s", "settle_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        for name in ("outlier_rate", "dropout_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        if self.measurement_var is not None:
            value = float(self.measurement_var)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("measurement_var must be finite and > 0")
            object.__setattr__(self, "measurement_var", value)

    @property
    def resolved_measurement_var(self) -> float:
        """Explicit R, or the variance of the generated Gaussian noise."""
        return (self.noise_std_mm ** 2 if self.measurement_var is None
                else float(self.measurement_var))


@dataclass(frozen=True)
class SensorData:
    readings: np.ndarray
    outlier: np.ndarray
    dropped: np.ndarray


@dataclass(frozen=True)
class SimulationMetrics:
    raw_sigma: float
    filtered_sigma: float
    reduction: float
    removed_pct: float
    raw_rmse: float
    filtered_rmse: float
    coast_rmse: float
    lag_ms: float


@dataclass(frozen=True)
class ScenarioResult:
    motion: str
    t: np.ndarray
    truth: np.ndarray
    noisy: np.ndarray
    estimate: np.ndarray
    outlier: np.ndarray
    dropped: np.ndarray
    raw_sigma: float
    filtered_sigma: float
    reduction: float
    removed_pct: float
    raw_rmse: float
    filtered_rmse: float
    coast_rmse: float
    lag_ms: float
    downweighted_count: int
    maneuver_count: int
    final_measurement_var: float


def ground_truth(kind: str, t: np.ndarray) -> np.ndarray:
    """Return the exact distance in millimetres for one named motion."""
    if kind == "static":
        return np.full_like(t, 150.0, dtype=float)
    if kind == "ramp":
        return 50.0 + 60.0 * t
    if kind == "steps":
        x = np.full_like(t, 60.0, dtype=float)
        for level, start in ((160.0, 1.5), (300.0, 3.0), (120.0, 4.5)):
            x[t >= start] = level
        return x
    if kind == "sine":
        return 200.0 + 80.0 * np.sin(2.0 * np.pi * 0.15 * t)
    raise ValueError(f"unknown motion: {kind}")


def fake_sensor(truth: np.ndarray, rng: np.random.Generator,
                config: SimulationConfig) -> SensorData:
    """Add noise, signed gross outliers, and NaN-marked dropped frames."""
    truth = np.asarray(truth, dtype=float)
    readings = truth + rng.normal(0.0, config.noise_std_mm, truth.size)
    outlier = rng.random(truth.size) < config.outlier_rate
    if outlier.any():
        readings[outlier] += rng.choice((-1.0, 1.0), outlier.sum()) * config.outlier_mm
    dropped = rng.random(truth.size) < config.dropout_rate
    readings[dropped] = np.nan
    return SensorData(readings=readings, outlier=outlier, dropped=dropped)


def lag_samples(estimate: np.ndarray, truth: np.ndarray, max_lag: int = 30) -> float:
    """Estimate lag by cross-correlation; return NaN when motion is absent."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = np.isfinite(estimate) & np.isfinite(truth)
    if valid.sum() < 3 or np.std(truth[valid]) < 1.0:
        return float("nan")
    first, last = np.where(valid)[0][[0, -1]]
    idx = np.arange(first, last + 1)
    est = np.interp(idx, np.where(valid)[0], estimate[valid])
    ref = truth[idx]
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    xc = np.correlate(est, ref, mode="full")
    lags = np.arange(-len(est) + 1, len(est))
    window = np.abs(lags) <= max_lag
    return float(lags[window][np.argmax(xc[window])])


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(error, dtype=float) ** 2)))


def _steady_mask(t: np.ndarray, truth: np.ndarray,
                 config: SimulationConfig) -> np.ndarray:
    """Mask initialization and post-step settling only for noise statistics."""
    mask = t - t[0] >= config.warmup_s
    transitions = np.abs(np.diff(truth, prepend=truth[0])) > 10.0
    settle_samples = int(math.ceil(config.settle_s * config.fps))
    for offset in range(settle_samples + 1):
        indices = np.where(transitions)[0] + offset
        mask[indices[indices < mask.size]] = False
    return mask


def calculate_metrics(t: np.ndarray, truth: np.ndarray, sensor: SensorData,
                      estimate: np.ndarray,
                      config: SimulationConfig) -> SimulationMetrics:
    """Calculate comparable accuracy, denoising, coasting, and lag metrics."""
    t = np.asarray(t, dtype=float)
    truth = np.asarray(truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if not (t.shape == truth.shape == sensor.readings.shape == estimate.shape ==
            sensor.outlier.shape == sensor.dropped.shape):
        raise ValueError("metric inputs must have identical shapes")

    finite_estimate = np.isfinite(estimate)
    common = ~sensor.dropped & np.isfinite(sensor.readings) & finite_estimate
    steady = common & ~sensor.outlier & _steady_mask(t, truth, config)
    if common.sum() == 0 or steady.sum() < 2:
        raise ValueError("simulation produced too few comparable samples")

    raw_error = sensor.readings - truth
    filtered_error = estimate - truth
    raw_sigma = float(np.std(raw_error[steady]))
    filtered_sigma = float(np.std(filtered_error[steady]))
    reduction = raw_sigma / filtered_sigma if filtered_sigma > 0 else float("inf")
    removed_pct = 100.0 * (1.0 - filtered_sigma / raw_sigma) if raw_sigma > 0 else 0.0
    coast = sensor.dropped & finite_estimate
    lag = lag_samples(estimate, truth)
    return SimulationMetrics(
        raw_sigma=raw_sigma,
        filtered_sigma=filtered_sigma,
        reduction=float(reduction),
        removed_pct=float(removed_pct),
        raw_rmse=_rmse(raw_error[common]),
        filtered_rmse=_rmse(filtered_error[common]),
        coast_rmse=_rmse(filtered_error[coast]) if coast.any() else float("nan"),
        lag_ms=lag / config.fps * 1000.0 if math.isfinite(lag) else float("nan"),
    )


def run_scenario(kind: str, config: SimulationConfig, seed: int = 7) -> ScenarioResult:
    """Generate and evaluate one motion with a stable per-motion random stream."""
    if kind not in MOTIONS:
        raise ValueError(f"unknown motion: {kind}")
    t = np.arange(0.0, config.duration_s, 1.0 / config.fps)
    truth = ground_truth(kind, t)
    motion_seed = np.random.SeedSequence((int(seed), MOTIONS.index(kind)))
    sensor = fake_sensor(truth, np.random.default_rng(motion_seed), config)
    filt = LiveFilter(process_accel_psd=config.process_accel_psd,
                      maneuver_accel_psd=config.maneuver_accel_psd,
                      measurement_var=config.resolved_measurement_var,
                      reject_sigma=config.reject_sigma,
                      max_consec_reject=config.max_consec_reject,
                      adapt_measurement_var=config.adapt_measurement_var)
    estimates = []
    downweighted_count = maneuver_count = 0
    for z, ti in zip(sensor.readings, t):
        estimates.append(filt.update(None if not np.isfinite(z) else float(z), ti))
        downweighted_count += int(filt.kf.downweighted)
        maneuver_count += int(filt.kf.maneuver)
    estimate = np.asarray(estimates)
    metrics = calculate_metrics(t, truth, sensor, estimate, config)

    return ScenarioResult(
        motion=kind, t=t, truth=truth, noisy=sensor.readings, estimate=estimate,
        outlier=sensor.outlier, dropped=sensor.dropped,
        raw_sigma=metrics.raw_sigma, filtered_sigma=metrics.filtered_sigma,
        reduction=metrics.reduction, removed_pct=metrics.removed_pct,
        raw_rmse=metrics.raw_rmse, filtered_rmse=metrics.filtered_rmse,
        coast_rmse=metrics.coast_rmse, lag_ms=metrics.lag_ms,
        downweighted_count=downweighted_count, maneuver_count=maneuver_count,
        final_measurement_var=filt.kf.r,
    )


def run_all(config: SimulationConfig, seed: int = 7) -> list[ScenarioResult]:
    return [run_scenario(kind, config, seed) for kind in MOTIONS]


def _display(value: float, suffix: str = "", width: int = 7) -> str:
    return (f"{value:{width}.2f}{suffix}" if math.isfinite(value)
            else f"{'n/a':>{width + len(suffix)}}")


def print_report(results: list[ScenarioResult], config: SimulationConfig) -> None:
    print(f"Gaussian noise σ={config.noise_std_mm:g} mm; "
          f"Kalman R={config.resolved_measurement_var:g} mm²")
    print(f"{'motion':8s} {'raw σ':>7s} {'filt σ':>7s} {'ratio':>8s} {'removed':>9s} "
          f"{'raw RMSE':>9s} {'filt RMSE':>10s} {'coast':>7s} {'lag':>8s} "
          f"{'soft':>5s} {'moves':>5s} {'R end':>7s}")
    for result in results:
        print(f"{result.motion:8s} {_display(result.raw_sigma)} {_display(result.filtered_sigma)} "
              f"{_display(result.reduction, '×', 6)} {_display(result.removed_pct, '%', 7)} "
              f"{_display(result.raw_rmse)} {_display(result.filtered_rmse, '', 8)} "
              f"{_display(result.coast_rmse)} {_display(result.lag_ms, 'ms', 6)} "
              f"{result.downweighted_count:5d} {result.maneuver_count:5d} "
              f"{result.final_measurement_var:7.2f}")


def noise_description(result: ScenarioResult) -> str:
    if result.removed_pct >= 0:
        return f"{result.removed_pct:.0f}% noise removed"
    return f"{abs(result.removed_pct):.0f}% more residual noise"


def save_plot(results: list[ScenarioResult], output: str) -> str:
    """Write the four-panel comparison; import matplotlib only when requested."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, result in zip(axes.ravel(), results):
        ax.plot(result.t, result.noisy, ".", ms=3, alpha=0.35, color="#b0b8c0",
                label="noisy sensor")
        ax.plot(result.t, result.truth, "-", lw=2.2, color="#2a9d5c",
                label="ground truth")
        ax.plot(result.t, result.estimate, "-", lw=1.6, color="#d1354a",
                label="live filter")
        lag = "n/a" if not math.isfinite(result.lag_ms) else f"{result.lag_ms:.0f} ms"
        ax.set_title(f"{result.motion} — {noise_description(result)}, "
                     f"RMSE {result.filtered_rmse:.2f} mm, lag {lag}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("distance (mm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.suptitle("Live Kalman filter on simulated noise, outliers, and dropouts",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", "--hz", dest="fps", type=float, default=10.0,
                        help="simulation/sample rate in Hz (VL53L5CX 8x8 maximum: 15)")
    parser.add_argument("--duration", type=float, default=15.0, dest="duration_s")
    parser.add_argument("--noise-std", type=float, default=5.0, dest="noise_std_mm")
    parser.add_argument("--outlier-rate", type=float, default=0.02)
    parser.add_argument("--outlier-mm", type=float, default=60.0)
    parser.add_argument("--dropout-rate", type=float, default=0.03)
    parser.add_argument("--process-accel-psd", type=float, default=500.0)
    parser.add_argument("--maneuver-accel-psd", type=float, default=5000.0)
    parser.add_argument("--reject-sigma", type=float, default=5.0)
    parser.add_argument("--max-consec-reject", type=int, default=1,
                        help="consistent large samples required after the first")
    parser.add_argument("--adaptive-r", action="store_true",
                        help="adapt R online when no per-sample variance is supplied")
    parser.add_argument("--measurement-var", type=float, default=None,
                        help="Kalman R in mm² (default: noise-std squared)")
    parser.add_argument("--output", default=os.path.join(HERE, "figs", "simulation.png"))
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> list[ScenarioResult]:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = SimulationConfig(
            fps=args.fps, duration_s=args.duration_s, noise_std_mm=args.noise_std_mm,
            outlier_rate=args.outlier_rate, outlier_mm=args.outlier_mm,
            dropout_rate=args.dropout_rate, process_accel_psd=args.process_accel_psd,
            maneuver_accel_psd=args.maneuver_accel_psd,
            reject_sigma=args.reject_sigma, max_consec_reject=args.max_consec_reject,
            adapt_measurement_var=args.adaptive_r, measurement_var=args.measurement_var,
        )
        results = run_all(config, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    print_report(results, config)
    if not args.no_plot:
        print(f"wrote {save_plot(results, args.output)}")
    return results


if __name__ == "__main__":
    main()
