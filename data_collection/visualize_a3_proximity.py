#!/usr/bin/env python3
"""
A3 — Proximity ranging accuracy
-------------------------------
Validates ToF distance-estimation accuracy across the working range against a
matte white target, per the A3 test spec:

    Metric : range error and noise (sigma) vs. true distance
    Error  : Reported - True
    RMSE   : sqrt(mean(Error^2))

Data: test_results/tof_mounted/a3/  (49 steps, 163.7 mm UR5 start, 10 mm increments,
ascending + descending). ``ur5_a3_*`` = UR5 z (m); ``vl53l5cx_a3_*`` = 64 zones (mm).

True distance
=============
The UR5 z is the end-effector height, which equals the sensor-to-target distance
plus a fixed mounting standoff. We estimate that standoff as the mean of
(UR5_z - reported) over every dwell step of BOTH directions, then define
    True = UR5_z - standoff
so any direction- or distance-dependent deviation surfaces as error (the
constant mounting offset is not a sensor error). The residual ~1.5% slope of
Reported-vs-True is a real sensor scale error and is reported separately.

Reported distance & noise per step
==================================
At each dwell step we pool the filtered (gated + MAD) zone readings over the held
frames:
    Reported  = mean of all valid zone readings
    Noise (sigma) = mean over zones of each zone's temporal std across the dwell
NOTE: the intern logged ~12 frames/step, below the spec's >=100-for-noise
recommendation, so sigma is a coarse estimate (shown with that caveat).

Run:
    .venv/bin/python data_collection/visualize_a3_proximity.py
"""

import os
import glob
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tof_common import load_readings, load_ground_truth, detect_steps, FILTER_LABEL

COL = {"ascending": "#1f77b4", "descending": "#d62728"}
MK = {"ascending": "o", "descending": "s"}

HERE = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(HERE, "test_results", "tof_mounted", "a3")


def per_step_stats(secs, grid, steps):
    """Return arrays (ur5_level, reported, noise_sigma, n_frames) per dwell step."""
    lvl, rep, sig, nfr = [], [], [], []
    for t0, t1, level in steps:
        m = (secs >= t0) & (secs <= t1)
        if not m.any():
            continue
        sub = grid[m]                                   # (F, 64) NaN-filtered
        if not np.isfinite(sub).any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reported = np.nanmean(sub)                  # pooled mean distance
            zone_temporal_std = np.nanstd(sub, axis=0)  # (64,) temporal noise/zone
            noise = np.nanmean(zone_temporal_std)
        lvl.append(level)
        rep.append(reported)
        sig.append(noise)
        nfr.append(int(m.sum()))
    return (np.array(lvl), np.array(rep), np.array(sig), np.array(nfr))


def load_direction(tag):
    uf = glob.glob(os.path.join(A3_DIR, f"ur5_a3_{tag}*.csv"))
    sf = glob.glob(os.path.join(A3_DIR, f"vl53l5cx_a3_{tag}*.csv"))
    if not uf or not sf:
        return None
    secs, grid, t0 = load_readings(sf[0])
    g_t, g_z = load_ground_truth(uf[0], t0)
    steps = detect_steps(g_t, g_z)
    lvl, rep, sig, nfr = per_step_stats(secs, grid, steps)
    med = np.nanmedian(grid, axis=1)                    # per-frame ToF reading (mm)
    return {"tag": tag, "ur5": lvl, "reported": rep, "sigma": sig, "nframes": nfr,
            "secs": secs, "med": med, "g_t": g_t, "g_z": g_z}


def make_overlay_error_noise(dirs, standoff, rmse):
    """One picture: a UR5/ToF time-series overlay per direction, then a single
    error-and-noise graph below."""
    n = len(dirs)
    fig = plt.figure(figsize=(11, 3.4 * n + 4.0))
    gs = GridSpec(n + 1, 1, height_ratios=[1.3] * n + [1.7], hspace=0.45)
    fig.suptitle("A3 — UR5 vs ToF overlay, with range error & noise",
                 fontsize=14, fontweight="bold")

    # Overlay panels (one per direction): UR5 position with ToF laid on top.
    for i, d in enumerate(dirs):
        ax = fig.add_subplot(gs[i])
        ax.plot(d["g_t"], d["g_z"], color="#1f77b4", lw=2.4,
                label="UR5 API position (z)")
        ax.plot(d["secs"], d["med"] + standoff, color="#ff7f0e", lw=1.2,
                label=f"ToF reading + {standoff:.0f} mm standoff")
        ax.set_title(f"Overlay — {d['tag']}", fontsize=10)
        ax.set_ylabel("Distance (mm)")
        ax.set_xlabel("Time since logging start (s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=9, framealpha=0.9)

    # Error & noise (single graph, twin axes) vs true distance.
    axe = fig.add_subplot(gs[n])
    axn = axe.twinx()
    axe.axhline(0, color="k", lw=0.8)
    axe.axhspan(-rmse, rmse, color="orange", alpha=0.10, zorder=0)
    handles = []
    for d in dirs:
        c = COL[d["tag"]]; mk = MK[d["tag"]]
        he = axe.plot(d["true"], d["error"], mk + "-", color=c, ms=4, lw=1.3,
                      label=f"{d['tag']} — range error")[0]
        hn = axn.plot(d["true"], d["sigma"], mk + "--", color=c, ms=3, lw=1.0,
                      alpha=0.45, label=f"{d['tag']} — noise σ")[0]
        handles += [he, hn]
    axe.set_xlabel("True distance (mm)  [UR5 z − standoff]")
    axe.set_ylabel("Range error = Reported − True (mm)")
    axn.set_ylabel("Noise σ (mm)", color="#555")
    axn.set_ylim(bottom=0)
    axe.grid(True, alpha=0.3)
    axe.set_title(f"Range error (solid, left)  &  noise σ (dashed, right)  vs distance "
                  f"·  ±RMSE band = {rmse:.1f} mm", fontsize=10)
    axe.legend(handles=handles, fontsize=8, ncol=2, loc="upper center")

    fig.text(0.5, 0.005, f"Filter: {FILTER_LABEL}.  ToF shifted by a shared "
             f"{standoff:.1f} mm standoff to overlay the UR5 position.",
             ha="center", fontsize=8, color="#555")
    out = os.path.join(A3_DIR, "a3_overlay_error_noise.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay figure written to {out}")


def main():
    dirs = [d for d in (load_direction("ascending"), load_direction("descending"))
            if d is not None]
    if not dirs:
        print("No A3 data found under", A3_DIR)
        return

    # Shared standoff from all dwell steps of both directions -> True distance.
    all_off = np.concatenate([d["ur5"] - d["reported"] for d in dirs])
    standoff = float(np.mean(all_off))
    for d in dirs:
        d["true"] = d["ur5"] - standoff
        d["error"] = d["reported"] - d["true"]          # == reported - ur5 + standoff

    # Combined fit Reported vs True (slope = scale error).
    true_all = np.concatenate([d["true"] for d in dirs])
    rep_all = np.concatenate([d["reported"] for d in dirs])
    err_all = np.concatenate([d["error"] for d in dirs])
    slope, intercept = np.polyfit(true_all, rep_all, 1)
    rmse = float(np.sqrt(np.mean(err_all ** 2)))
    mae = float(np.mean(np.abs(err_all)))
    max_abs = float(np.max(np.abs(err_all)))

    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(11, 13))
    fig.suptitle("A3 — Proximity ranging accuracy (VL53L5CX vs matte white target)",
                 fontsize=14, fontweight="bold")

    # Panel A: Reported vs True with ideal y=x.
    lo = min(true_all.min(), rep_all.min())
    hi = max(true_all.max(), rep_all.max())
    axA.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1, label="ideal (y = x)")
    for d in dirs:
        axA.plot(d["true"], d["reported"], MK[d["tag"]], ms=4,
                 color=COL[d["tag"]], label=f"{d['tag']}")
    axA.set_xlabel("True distance (mm)  [UR5 z − standoff]")
    axA.set_ylabel("Reported distance (mm)")
    axA.set_title(f"Reported vs True   ·   fit slope = {slope:.4f} "
                  f"({(slope-1)*100:+.2f}% scale)   ·   standoff = {standoff:.1f} mm")
    axA.legend(fontsize=9); axA.grid(True, alpha=0.3)

    # Panel B: Range error vs True with +/- sigma noise bars (THE core curve).
    axB.axhline(0, color="k", lw=0.8)
    for d in dirs:
        axB.errorbar(d["true"], d["error"], yerr=d["sigma"], fmt=MK[d["tag"]],
                     ms=4, capsize=2, lw=1, color=COL[d["tag"]],
                     label=f"{d['tag']} (error ± noise σ)")
    axB.axhspan(-rmse, rmse, color="orange", alpha=0.12, label=f"±RMSE ({rmse:.1f} mm)")
    axB.set_xlabel("True distance (mm)")
    axB.set_ylabel("Range error = Reported − True (mm)")
    axB.set_title(f"Range error vs distance   ·   RMSE = {rmse:.1f} mm   ·   "
                  f"MAE = {mae:.1f} mm   ·   max|err| = {max_abs:.1f} mm")
    axB.legend(fontsize=9); axB.grid(True, alpha=0.3)

    # Panel C: Noise sigma vs True.
    for d in dirs:
        axC.plot(d["true"], d["sigma"], MK[d["tag"]] + "-", ms=4,
                 color=COL[d["tag"]], label=f"{d['tag']}")
    nmin = min(int(d["nframes"].min()) for d in dirs)
    nmax = max(int(d["nframes"].max()) for d in dirs)
    axC.set_xlabel("True distance (mm)")
    axC.set_ylabel("Noise σ (mm)")
    axC.set_title(f"Ranging noise vs distance   ·   per-zone temporal σ "
                  f"({nmin}–{nmax} frames/step — below spec's ≥100)")
    axC.legend(fontsize=9); axC.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.text(0.5, 0.005, f"Filter: {FILTER_LABEL}.  True distance uses a shared "
             f"standoff of {standoff:.1f} mm estimated from both directions.",
             ha="center", fontsize=8, color="#555")
    out = os.path.join(A3_DIR, "a3_proximity_accuracy.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # Second figure: UR5/ToF overlay + a combined error & noise graph.
    make_overlay_error_noise(dirs, standoff, rmse)

    # Console summary table.
    print(f"A3 proximity ranging accuracy")
    print(f"  standoff (shared)     : {standoff:.1f} mm")
    print(f"  scale (fit slope)     : {slope:.4f}  ({(slope-1)*100:+.2f}%)")
    print(f"  overall RMSE / MAE    : {rmse:.2f} / {mae:.2f} mm   max|err| {max_abs:.1f} mm")
    for d in dirs:
        de = d["error"]
        print(f"  {d['tag']:11s}: steps={len(d['true'])}  "
              f"true {d['true'].min():.0f}–{d['true'].max():.0f} mm  "
              f"RMSE={np.sqrt(np.mean(de**2)):.2f} mm  "
              f"σ {d['sigma'].min():.2f}–{d['sigma'].max():.2f} mm")
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
