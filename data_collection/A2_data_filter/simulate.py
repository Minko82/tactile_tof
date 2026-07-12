"""
simulate.py — fake a noisy sensor and watch the live filter denoise it.

Everything here is SIMULATED (no hardware, no captures): we make a ground-truth
motion, corrupt it into a noisy sensor stream (Gaussian noise + occasional
gross outliers + dropped frames), push it through live_filter.LiveFilter one
sample at a time, and compare truth vs noisy vs filtered.

Four motions exercise the filter's regimes:
  static   — hold still            (pure noise rejection)
  ramp     — constant velocity     (should track with NO lag)
  steps    — hold / jump / hold    (maneuvers + settling)
  sine     — smoothly accelerating (curvature)

Run:  ~/ur5-env/bin/python3 simulate.py
Writes figs/simulation.png and prints per-motion noise-reduction + RMSE.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from live_filter import LiveFilter

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30.0
DUR = 6.0
NOISE_MM = 5.0            # sensor Gaussian noise (1 sigma)
OUTLIER_RATE = 0.02      # fraction of frames that are gross outliers
OUTLIER_MM = 60.0        # outlier magnitude
DROP_RATE = 0.03         # fraction of frames dropped (no reading)


def ground_truth(kind, t):
    if kind == "static":
        return np.full_like(t, 150.0)
    if kind == "ramp":
        return 50.0 + 60.0 * t
    if kind == "steps":
        x = np.full_like(t, 60.0)
        for lvl, t0 in [(160, 1.5), (300, 3.0), (120, 4.5)]:
            x[t >= t0] = lvl
        return x
    if kind == "sine":
        return 200.0 + 80.0 * np.sin(2 * np.pi * 0.15 * t)   # ~75 mm/s peak (ToF-realistic)
    raise ValueError(kind)


def fake_sensor(truth, rng):
    """Corrupt ground truth into a realistic noisy stream (None = dropped frame)."""
    z = truth + rng.normal(0, NOISE_MM, len(truth))
    out = rng.random(len(truth)) < OUTLIER_RATE            # gross outliers
    z[out] += rng.choice([-1.0, 1.0], out.sum()) * (OUTLIER_MM + rng.normal(0, 15, out.sum()))
    z = [None if rng.random() < DROP_RATE else float(v) for v in z]  # dropped frames
    return z


def lag_samples(est, truth):
    """Estimate residual time lag by cross-correlation (0 = no lag)."""
    if np.std(truth) < 1.0:                      # motionless -> lag undefined
        return 0
    a = est - np.mean(est); b = truth - np.mean(truth); n = len(a)
    xc = np.correlate(a, b, mode="full"); lags = np.arange(-n + 1, n)
    w = np.abs(lags) <= 30
    return int(lags[w][np.argmax(xc[w])])


def run(kind, rng):
    t = np.arange(0, DUR, 1 / FPS)
    truth = ground_truth(kind, t)
    noisy = fake_sensor(truth, rng)
    filt = LiveFilter()
    est = np.array([filt.update(z, ti) for z, ti in zip(noisy, t)])
    zn = np.array([np.nan if z is None else z for z in noisy])
    valid = np.isfinite(zn)
    # NOISE is measured on steady stretches — exclude the brief step-edge transients
    # from BOTH raw and filtered (else a few 180 mm jumps dominate the std). RMSE below
    # is the honest overall number and DOES include those transients.
    settle = np.abs(np.diff(truth, prepend=truth[0])) > 10.0
    for _ in range(int(0.3 * FPS)):
        settle[1:] |= settle[:-1]
    steady = ~settle
    raw_sigma = np.std((zn - truth)[steady & valid])
    filt_sigma = np.std((est - truth)[steady])
    rmse = np.sqrt(np.mean((est - truth) ** 2))
    dt = 1 / FPS
    return dict(t=t, truth=truth, noisy=zn, est=est, raw_sigma=raw_sigma,
                filt_sigma=filt_sigma, rmse=rmse, lag_ms=lag_samples(est, truth) * dt * 1000)


def main():
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    rng = np.random.default_rng(7)
    kinds = ["static", "ramp", "steps", "sine"]
    results = {k: run(k, rng) for k in kinds}

    print(f"{'motion':8s} {'raw σ':>7s} {'filt σ':>7s} {'reduction':>10s} {'RMSE':>7s} {'lag':>7s}")
    for k in kinds:
        r = results[k]
        print(f"{k:8s} {r['raw_sigma']:6.2f}  {r['filt_sigma']:6.2f}  "
              f"{r['raw_sigma']/r['filt_sigma']:8.1f}×  {r['rmse']:6.2f}  {r['lag_ms']:5.0f}ms")

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, k in zip(axes.ravel(), kinds):
        r = results[k]
        ax.plot(r["t"], r["noisy"], ".", ms=3, alpha=0.35, color="#b0b8c0", label="noisy sensor")
        ax.plot(r["t"], r["truth"], "-", lw=2.2, color="#2a9d5c", label="ground truth")
        ax.plot(r["t"], r["est"], "-", lw=1.6, color="#d1354a", label="live filter")
        ax.set_title(f"{k}  —  {r['raw_sigma']/r['filt_sigma']:.1f}× less noise, "
                     f"RMSE {r['rmse']:.2f} mm, lag {r['lag_ms']:.0f} ms",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("time (s)"); ax.set_ylabel("mm"); ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="upper right", framealpha=.95)
    fig.suptitle("Live Kalman filter on simulated noisy data (Gaussian noise + outliers + dropouts)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(HERE, "figs", "simulation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
