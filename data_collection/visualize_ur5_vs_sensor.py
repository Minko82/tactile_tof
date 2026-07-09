#!/usr/bin/env python3
"""
UR5 vs ToF sensor overlay / error / per-step heatmap visualizer
---------------------------------------------------------------
For each flat-mount test under test_results/flat_ur5_mount/ (a folder with a
``readings_*.csv`` and a ``*_ground_truth.csv``) this produces:

  * ``ur5_vs_sensor.png`` — UR5 API position overlaid with the filtered, offset-
    corrected ToF reading, plus the error over time.
  * ``step_heatmaps.png`` — an 8x8 distance heatmap for every dwell step,
    averaged over the frames held at that position. Each panel is autoscaled to
    its own range so within-step structure (surface flatness / tilt) is visible.

Run:
    .venv/bin/python data_collection/visualize_ur5_vs_sensor.py
"""

import os
import glob
import math
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tof_common import (
    GRID, MAX_RANGE_MM, MAD_K, MIN_VALID_ZONES, FILTER_LABEL,
    load_readings, load_ground_truth, detect_steps, dwell_mask,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "test_results", "flat_ur5_mount")

FIXED_OFFSET_MM = 132.0        # mounting standoff for this flat-mount rig


def pretty_name(folder: str) -> str:
    name = os.path.basename(folder.rstrip("/"))
    return name.replace("vl53l5cx_", "").replace("_", " ")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: overlay + error
# ─────────────────────────────────────────────────────────────────────────────
def make_overlay(folder, secs, grid, g_t, g_z, steps):
    med = np.nanmedian(grid, axis=1)
    nvalid = np.sum(np.isfinite(grid), axis=1)
    corrected = med + FIXED_OFFSET_MM

    gt_start, gt_end = g_t[0], g_t[-1]
    in_win = (secs >= gt_start) & (secs <= gt_end)
    gt_on_sensor = np.interp(secs, g_t, g_z)
    err = corrected - gt_on_sensor

    # Headline stats over DWELL frames only (exclude robot-motion transients).
    dwell = dwell_mask(secs, steps)
    good = in_win & dwell & np.isfinite(med) & (nvalid >= MIN_VALID_ZONES)
    residual_bias = np.nanmedian(err[good]) if good.any() else np.nan
    rmse = np.sqrt(np.nanmean(err[good] ** 2)) if good.any() else np.nan
    mae = np.nanmean(np.abs(err[good])) if good.any() else np.nan

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]})
    fig.suptitle(f"UR5 position vs ToF reading — {pretty_name(folder)}",
                 fontsize=14, fontweight="bold")

    # Top: the two lines overlaid (sensor shifted by the fixed offset).
    filt_label = (f"Sensor median  ·  filter: {FILTER_LABEL}  ·  "
                  f"+{FIXED_OFFSET_MM:.0f} mm offset")
    ax1.plot(g_t, g_z, color="#1f77b4", lw=2.4, label="UR5 API position (z)")
    ax1.plot(secs, corrected, color="#ff7f0e", lw=1.3, label=filt_label)
    ax1.axvspan(gt_start, gt_end, color="gray", alpha=0.06)
    ax1.set_ylabel("Distance (mm)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Bottom: error over time (full trace; transition spikes are visible but
    # excluded from the headline stats).
    ax2.axhline(0, color="k", lw=0.8)
    ax2.plot(secs[in_win], err[in_win], color="#d62728", lw=1.2,
             label=f"Error (sensor + {FIXED_OFFSET_MM:.0f} mm − UR5)")
    ax2.axvspan(gt_start, gt_end, color="gray", alpha=0.06)
    ax2.set_ylabel("Error (mm)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    finite = err[in_win][np.isfinite(err[in_win])]
    if finite.size:
        lim = max(20, np.nanpercentile(np.abs(finite), 99) * 1.5)
        ax2.set_ylim(-lim, lim)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time since logging start (s)")

    stats = (f"dwell stats — offset = +{FIXED_OFFSET_MM:.0f} mm    "
             f"residual bias = {residual_bias:+.1f} mm    "
             f"RMSE = {rmse:.1f} mm    MAE = {mae:.1f} mm")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9, title=stats)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(folder, "ur5_vs_sensor.png"), dpi=130)
    plt.close(fig)
    return residual_bias, rmse, mae


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: per-step heatmaps (each panel autoscaled to reveal within-step detail)
# ─────────────────────────────────────────────────────────────────────────────
def make_heatmaps(folder, secs, grid, steps, offset_mm):
    panels = []
    for k, (t0, t1, level) in enumerate(steps, start=1):
        m = (secs >= t0) & (secs <= t1)
        if not m.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            zone_mean = np.nanmean(grid[m], axis=0)        # (64,) NaN if zone never valid
        n_valid_zones = int(np.sum(np.isfinite(zone_mean)))
        panels.append({
            "k": k,
            "true_d": level - offset_mm,                   # sensor-to-target distance
            "map": zone_mean.reshape(GRID, GRID),
            "sigma": float(np.nanstd(zone_mean)),
            "nzones": n_valid_zones,
        })
    if not panels:
        return 0

    n = len(panels)
    ncol = math.ceil(math.sqrt(n))
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(2.9 * ncol + 0.5, 3.0 * nrow + 0.8))
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")            # zones with no valid reading

    for ax, p in zip(axes, panels):
        hmap = p["map"]
        finite = hmap[np.isfinite(hmap)]
        # autoscale each panel to its own robust range -> within-step detail shows
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, 5), np.nanpercentile(finite, 95)
            if vmax - vmin < 1.0:        # near-uniform: pad so it isn't one flat color
                mid = (vmax + vmin) / 2
                vmin, vmax = mid - 1.0, mid + 1.0
        else:
            vmin, vmax = 0, 1
        im = ax.imshow(hmap, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest")
        ax.set_title(f"step {p['k']} • d≈{p['true_d']:.0f} mm\n"
                     f"σ={p['sigma']:.1f} mm · {p['nzones']}/64 zones", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Per-step ToF distance heatmaps (8×8 zones, each panel autoscaled) — "
        f"{pretty_name(folder)}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(folder, "step_heatmaps.png"), dpi=130)
    plt.close(fig)
    return n


def process(folder):
    rpath = glob.glob(os.path.join(folder, "readings_*.csv"))
    gpath = glob.glob(os.path.join(folder, "*ground_truth.csv"))
    if not rpath or not gpath:
        return None

    secs, grid, t0 = load_readings(rpath[0])
    g_t, g_z = load_ground_truth(gpath[0], t0)
    steps = detect_steps(g_t, g_z)

    residual_bias, rmse, mae = make_overlay(folder, secs, grid, g_t, g_z, steps)
    n_steps = make_heatmaps(folder, secs, grid, steps, FIXED_OFFSET_MM)

    return {"test": pretty_name(folder), "residual_bias_mm": residual_bias,
            "rmse_mm": rmse, "mae_mm": mae, "n_steps": n_steps}


def main():
    folders = sorted(
        d for d in glob.glob(os.path.join(ROOT, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*ground_truth.csv")))
    results = []
    for folder in folders:
        r = process(folder)
        if r:
            results.append(r)
            print(f"✓ {r['test']:<55}  dwell residual_bias={r['residual_bias_mm']:+6.1f}mm  "
                  f"RMSE={r['rmse_mm']:6.1f}mm  MAE={r['mae_mm']:6.1f}mm  steps={r['n_steps']}")

    if not results:
        print("No test folders with ground truth found under", ROOT)
        return

    fig, ax = plt.subplots(figsize=(11, 0.5 * len(results) + 2))
    names = [r["test"] for r in results]
    rmses = [r["rmse_mm"] for r in results]
    y = np.arange(len(results))
    ax.barh(y, rmses, color="#d62728", alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Dwell RMSE after +{FIXED_OFFSET_MM:.0f} mm offset (mm)")
    ax.set_title("ToF dwell error across flat-mount tests (filtered)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    for yi, v in zip(y, rmses):
        ax.text(v, yi, f" {v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    summary = os.path.join(ROOT, "summary_tracking_rmse.png")
    fig.savefig(summary, dpi=130)
    plt.close(fig)
    print(f"\nSummary written to {summary}")


if __name__ == "__main__":
    main()
