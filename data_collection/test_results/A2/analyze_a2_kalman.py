#!/usr/bin/env python3
"""
A2 — Baseline signal & Kalman-filter characterization
-----------------------------------------------------
Quantifies the noise reduction a 1-D Kalman filter delivers on the ToF
peak-distance estimate, per the A2 test spec:

    Metric     : peak-distance jitter (raw vs filtered); noise-reduction factor
    Unit       : mm (jitter); x (factor)
    Formula    : Jitter = sigma(peak-distance)
                 Reduction = jitter_raw / jitter_filtered
    Reference  : the raw stream (controller-invariant)

Input is an A2 static recording (``timestamp,zone_00..zone_63`` mm CSV) from
a2_record.py. The per-frame "peak-distance" is reduced to one scalar (default:
median of the gated valid zones) and filtered along time.

Kalman model
============
Default ``--model cp`` is a constant-position random walk (the standard choice
for a static target). Per sample, with dt = interval to the previous sample:

    predict :  x⁻ = x            P⁻ = P + q·dt
    update  :  K  = P⁻/(P⁻+R)    x  = x⁻ + K·(z − x⁻)    P = (1−K)·P⁻

    q  = process-noise variance growth   [mm²/s]   (how fast the true distance may drift)
    R  = measurement-noise variance      [mm²]     (= sensor jitter², i.e. σ_raw²)

``--model cv`` is a constant-velocity filter (state = [distance, velocity]); there
q is the acceleration spectral density [mm²/s³]. Pass the deployed controller's
q and R with --q / --r so the reported factor reflects that filter.

WHY TUNING MATTERS: on a perfectly static target, a tighter filter (smaller q/R
ratio) always removes more noise — as q→0 the estimate converges to the mean and
the "reduction" grows without bound. The factor is only meaningful for a filter
that must still track real target motion, which is exactly what the q value sets.

Run:
    python3 data_collection/analyze_a2_kalman.py <recording.csv> --q 25 --r 4
    python3 data_collection/analyze_a2_kalman.py <recording.csv> --q 25 --r 4 --accept 1.0
    python3 data_collection/analyze_a2_kalman.py <recording.csv> --model cv --q 500 --r 4
"""

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_ZONES = 64
MAX_RANGE_MM = 800.0    # far-range gate (matches tof_common): drop 0 and background returns


# ── Load & peak-distance reduction ─────────────────────────────────────────────
def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def load_peak(path: str, mode: str, zone: int, max_range: float = MAX_RANGE_MM):
    """Return (secs, peak_mm) — one peak-distance scalar per frame (NaN if no valid zones).

    Applies only the far-range gate (raw stream, no MAD), then reduces the frame:
      median : robust median of gated valid zones (default)
      mean   : mean of gated valid zones
      zone   : a single zone index (genuine per-pixel jitter)
    """
    ts, peaks = [], []
    with open(path) as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            if not row or len(row) < 1 + N_ZONES:
                continue
            try:
                ts.append(parse_ts(row[0]))
            except ValueError:
                continue
            vals = np.array([float(x) for x in row[1:1 + N_ZONES]])
            if mode == "zone":
                v = vals[zone]
                peaks.append(v if (0 < v <= max_range) else np.nan)
            else:
                g = vals[(vals > 0) & (vals <= max_range)]
                if g.size == 0:
                    peaks.append(np.nan)
                else:
                    peaks.append(float(np.median(g)) if mode == "median" else float(np.mean(g)))
    if not ts:
        sys.exit(f"No frames parsed from {path}")
    t0 = ts[0]
    secs = np.array([(t - t0).total_seconds() for t in ts])
    return secs, np.array(peaks, dtype=float)


# ── Kalman filters ──────────────────────────────────────────────────────────────
def kalman_cp(z, dt, q, R, q_per_step=False):
    """Constant-position (random-walk) scalar Kalman filter.

    z may contain NaN (dropped frames): those do predict-only (no measurement
    update), which is the correct handling for a missing sample. Returns
    (x_filt, P, K) with x_filt defined at every index.
    """
    n = len(z)
    xf = np.empty(n)
    Pf = np.empty(n)
    Kf = np.full(n, np.nan)
    # init on first finite measurement
    first = next((i for i in range(n) if np.isfinite(z[i])), 0)
    x = z[first] if np.isfinite(z[first]) else 0.0
    P = R
    for k in range(n):
        Q = q if q_per_step else q * dt[k]
        P = P + Q                                   # predict
        if np.isfinite(z[k]):                       # update only when measured
            S = P + R
            K = P / S
            x = x + K * (z[k] - x)
            P = (1.0 - K) * P
            Kf[k] = K
        xf[k] = x
        Pf[k] = P
    return xf, Pf, Kf


def kalman_cv(z, dt, q, R):
    """Constant-velocity Kalman filter (state = [distance, velocity]).

    q = acceleration spectral density [mm²/s³]. Continuous white-noise-accel Q.
    """
    n = len(z)
    xf = np.empty(n)
    first = next((i for i in range(n) if np.isfinite(z[i])), 0)
    x = np.array([z[first] if np.isfinite(z[first]) else 0.0, 0.0])
    P = np.diag([R, (50.0) ** 2])                   # loose initial velocity variance
    H = np.array([[1.0, 0.0]])
    for k in range(n):
        t = dt[k]
        F = np.array([[1.0, t], [0.0, 1.0]])
        Q = q * np.array([[t ** 3 / 3.0, t ** 2 / 2.0],
                          [t ** 2 / 2.0, t]])
        x = F @ x                                   # predict
        P = F @ P @ F.T + Q
        if np.isfinite(z[k]):                       # update
            S = H @ P @ H.T + R
            K = (P @ H.T) / S                       # (2,1)
            x = x + (K[:, 0] * (z[k] - (H @ x)[0]))
            P = (np.eye(2) - K @ H) @ P
        xf[k] = x[0]
    return xf, None, None


# ── Metrics ─────────────────────────────────────────────────────────────────────
def detrended_std(y):
    """Std after removing a linear trend (isolates jitter from slow drift)."""
    x = np.arange(len(y), dtype=float)
    a, b = np.polyfit(x, y, 1)
    return float(np.std(y - (a * x + b), ddof=1))


def main():
    ap = argparse.ArgumentParser(description="A2 Kalman-filter noise-reduction characterization.")
    ap.add_argument("csv", help="A2 static recording (timestamp,zone_00..zone_63).")
    ap.add_argument("--q", type=float, required=True,
                    help="Process-noise q. cp: mm²/s (or per-step with --q-per-step). cv: mm²/s³.")
    ap.add_argument("--r", "--R", dest="R", type=float, required=True,
                    help="Measurement-noise variance R [mm²] (≈ raw jitter σ²).")
    ap.add_argument("--model", choices=["cp", "cv"], default="cp",
                    help="cp = constant-position random walk (default); cv = constant-velocity.")
    ap.add_argument("--q-per-step", action="store_true",
                    help="Interpret q as absolute per-step variance (ignore dt). cp model only.")
    ap.add_argument("--peak", choices=["median", "mean", "zone"], default="median",
                    help="Per-frame peak-distance reduction (default median of valid zones).")
    ap.add_argument("--zone", type=int, default=27, help="Zone index for --peak zone (0..63).")
    ap.add_argument("--max-range", type=float, default=MAX_RANGE_MM,
                    help=f"Far-range gate in mm (default {MAX_RANGE_MM:g}). Raise if the "
                         "static target sits beyond it (e.g. a ~1.6 m standoff).")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="Seconds to discard at the start (filter convergence) before scoring.")
    ap.add_argument("--accept", type=float, default=None,
                    help="Acceptance threshold: filtered jitter ≤ this many mm → PASS.")
    ap.add_argument("--out", default=None, help="Figure path (defaults next to the CSV).")
    args = ap.parse_args()

    secs, peak = load_peak(args.csv, args.peak, args.zone, args.max_range)
    dt = np.empty_like(secs)
    dt[0] = 0.0
    dt[1:] = np.diff(secs)
    span = secs[-1] - secs[0]
    eff_hz = (len(secs) - 1) / span if span > 0 else float("nan")
    n_valid = int(np.isfinite(peak).sum())
    n_drop = len(peak) - n_valid

    # Filter the full stream.
    if args.model == "cp":
        filt, P, K = kalman_cp(peak, dt, args.q, args.R, q_per_step=args.q_per_step)
        Kss = K[np.isfinite(K)]
        k_report = float(np.median(Kss[len(Kss) // 2:])) if Kss.size else float("nan")
    else:
        filt, _, _ = kalman_cv(peak, dt, args.q, args.R)
        k_report = float("nan")

    # Score after warmup, on frames with a valid raw measurement (fair like-for-like).
    score = (secs >= secs[0] + args.warmup) & np.isfinite(peak)
    if score.sum() < 3:
        sys.exit("Too few valid frames after warmup to compute jitter.")
    raw_s = peak[score]
    flt_s = filt[score]

    sigma_raw = float(np.std(raw_s, ddof=1))
    sigma_flt = float(np.std(flt_s, ddof=1))
    reduction = sigma_raw / sigma_flt if sigma_flt > 0 else float("inf")
    # Supplementary: jitter with slow drift removed (thermal drift ≠ jitter).
    sigma_raw_dt = detrended_std(raw_s)
    sigma_flt_dt = detrended_std(flt_s)
    reduction_dt = sigma_raw_dt / sigma_flt_dt if sigma_flt_dt > 0 else float("inf")

    # ── Figure ──────────────────────────────────────────────────────────────────
    out = args.out or os.path.splitext(args.csv)[0] + "_a2_kalman.png"
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(11, 11))
    fig.suptitle("A2 — Baseline signal & Kalman-filter characterization",
                 fontsize=14, fontweight="bold")

    peak_label = {"median": "median of valid zones", "mean": "mean of valid zones",
                  "zone": f"zone {args.zone}"}[args.peak]
    mean_lvl = float(np.nanmean(raw_s))

    # A: full time-series overlay.
    axA.plot(secs, peak, color="#bbbbbb", lw=0.8, label=f"raw peak-distance ({peak_label})")
    axA.plot(secs, filt, color="#d62728", lw=1.6, label=f"Kalman filtered ({args.model})")
    axA.axvspan(secs[0], secs[0] + args.warmup, color="k", alpha=0.06, label="warm-up (excluded)")
    axA.set_xlabel("Time (s)")
    axA.set_ylabel("Peak-distance (mm)")
    axA.set_title(f"Raw vs filtered stream  ·  {eff_hz:.1f} Hz effective  ·  {n_valid} frames"
                  + (f"  ·  {n_drop} dropped" if n_drop else ""))
    axA.grid(True, alpha=0.3)
    axA.legend(loc="best", fontsize=9)

    # B: zoom on a representative 8 s window to show the smoothing.
    z0 = secs[0] + args.warmup
    z1 = min(z0 + 8.0, secs[-1])
    zm = (secs >= z0) & (secs <= z1)
    axB.plot(secs[zm], peak[zm], color="#999999", lw=1.0, marker=".", ms=3, label="raw")
    axB.plot(secs[zm], filt[zm], color="#d62728", lw=2.0, label="filtered")
    axB.axhline(mean_lvl, color="#1f77b4", lw=0.8, ls="--", label=f"mean {mean_lvl:.1f} mm")
    axB.set_xlabel("Time (s)")
    axB.set_ylabel("Peak-distance (mm)")
    axB.set_title(f"Zoom ({z0:.0f}–{z1:.0f} s) — jitter suppression")
    axB.grid(True, alpha=0.3)
    axB.legend(loc="best", fontsize=9)

    # C: deviation-from-mean histograms (raw vs filtered) with σ annotations.
    dev_raw = raw_s - raw_s.mean()
    dev_flt = flt_s - flt_s.mean()
    lim = max(np.abs(dev_raw).max(), 1e-6)
    bins = np.linspace(-lim, lim, 61)
    axC.hist(dev_raw, bins=bins, color="#bbbbbb", alpha=0.8, label=f"raw  σ={sigma_raw:.3f} mm")
    axC.hist(dev_flt, bins=bins, color="#d62728", alpha=0.6, label=f"filtered  σ={sigma_flt:.3f} mm")
    axC.set_xlabel("Deviation from mean (mm)")
    axC.set_ylabel("Count")
    axC.set_title(f"Jitter distribution  ·  reduction = {reduction:.2f}×"
                  + (f"  ·  steady-state K≈{k_report:.3f}" if np.isfinite(k_report) else ""))
    axC.grid(True, alpha=0.3)
    axC.legend(loc="best", fontsize=9)

    verdict = ""
    if args.accept is not None:
        verdict = ("  ·  PASS" if sigma_flt <= args.accept else "  ·  FAIL") + \
                  f" (accept ≤ {args.accept:.2f} mm)"
    fig.text(0.5, 0.005,
             f"Model {args.model} · q={args.q:g}{' /step' if args.q_per_step else ''} · R={args.R:g} mm² · "
             f"warmup {args.warmup:g}s · Reduction = σ_raw/σ_filt = {reduction:.2f}×{verdict}",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # ── Console + JSON sidecar ────────────────────────────────────────────────────
    result = {
        "recording": os.path.abspath(args.csv),
        "model": args.model, "q": args.q, "q_per_step": args.q_per_step, "R": args.R,
        "peak_mode": args.peak, "zone": args.zone if args.peak == "zone" else None,
        "effective_hz": eff_hz, "n_frames": len(secs), "n_valid": n_valid, "n_dropped": n_drop,
        "warmup_s": args.warmup, "mean_distance_mm": mean_lvl,
        "sigma_raw_mm": sigma_raw, "sigma_filtered_mm": sigma_flt, "reduction_factor": reduction,
        "sigma_raw_detrended_mm": sigma_raw_dt, "sigma_filtered_detrended_mm": sigma_flt_dt,
        "reduction_factor_detrended": reduction_dt,
        "steady_state_K": k_report if np.isfinite(k_report) else None,
        "accept_mm": args.accept,
        "pass": (None if args.accept is None else bool(sigma_flt <= args.accept)),
    }
    sidecar = os.path.splitext(out)[0] + ".json"
    with open(sidecar, "w") as f:
        json.dump(result, f, indent=2)

    print("── A2 Kalman characterization ────────────────────────────────")
    print(f"  recording        : {args.csv}")
    print(f"  peak-distance    : {peak_label}   (mean {mean_lvl:.2f} mm)")
    print(f"  effective rate   : {eff_hz:.2f} Hz   ({n_valid} valid / {len(secs)} frames"
          + (f", {n_drop} dropped" if n_drop else "") + ")")
    print(f"  model / tuning   : {args.model}   q={args.q:g}"
          f"{' per-step' if args.q_per_step else ''}   R={args.R:g} mm²"
          + (f"   steady-state K≈{k_report:.3f}" if np.isfinite(k_report) else ""))
    print(f"  jitter (σ)       : raw {sigma_raw:.3f} mm  →  filtered {sigma_flt:.3f} mm")
    print(f"  REDUCTION FACTOR : {reduction:.2f}×")
    print(f"  (detrended σ)    : raw {sigma_raw_dt:.3f} → filtered {sigma_flt_dt:.3f} mm  "
          f"({reduction_dt:.2f}× — slow drift removed)")
    if args.accept is not None:
        print(f"  acceptance       : filtered {sigma_flt:.3f} mm "
              f"{'≤' if sigma_flt <= args.accept else '>'} {args.accept:.2f} mm  →  "
              f"{'PASS' if sigma_flt <= args.accept else 'FAIL'}")
    print(f"  figure           : {out}")
    print(f"  metrics (json)   : {sidecar}")
    if not (9.0 <= eff_hz <= 16.0):
        print(f"  NOTE: {eff_hz:.1f} Hz — the spec's 60 Hz is unreachable at 8x8 (64 zones); "
              "15 Hz is the ceiling. The filter used dt from timestamps, so the σ ratio is valid.")


if __name__ == "__main__":
    main()
