"""
step_heatmaps.py — Average 8×8 sensor reading during each step plateau.

For each of the 4 arm steps, averages the 8×8 zone readings across the
stable plateau region of all 50 recordings.

Usage:
    python3 step_heatmaps.py

Output:
    test_results/step_heatmaps.csv   — 4 rows (steps) × 64 zone columns
    test_results/step_heatmaps.png   — 4 heatmaps, one per step
"""

import glob
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
N_STEPS       = 4
SMOOTH_WINDOW = 5
MIN_STEP_GAP  = 8
SETTLE        = 8    # frames after onset to skip before measuring plateau

ZONE_COLS = [f"zone_{i:02d}" for i in range(64)]

# ── Scholarly style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "figure.dpi":        300,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def mean_signal(df: pd.DataFrame) -> np.ndarray:
    return df[ZONE_COLS].replace(0, np.nan).mean(axis=1).values

def smooth(signal: np.ndarray) -> np.ndarray:
    k = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    return np.convolve(signal, k, mode="same")

def detect_step_onsets(signal: np.ndarray) -> list[int]:
    sm    = smooth(signal)
    deriv = np.diff(sm, prepend=sm[0])
    neg   = -deriv
    peaks, props = find_peaks(neg, height=1.5, distance=MIN_STEP_GAP)

    if len(peaks) >= N_STEPS:
        top   = np.argsort(props["peak_heights"])[-N_STEPS:]
        peaks = np.sort(peaks[top])
    else:
        peaks = np.sort(np.argsort(neg)[-N_STEPS:])

    onsets = []
    for peak in peaks:
        onset = peak
        for i in range(int(peak) - 1, max(0, int(peak) - 20), -1):
            if deriv[i] >= -1.0:
                onset = i + 1
                break
        onsets.append(int(onset))
    return onsets

# ── Collect per-step plateau averages ─────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "readings_*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No readings_*.csv found in {DATA_DIR}")
print(f"Found {len(csv_files)} files.")

# accum[step] = list of shape-(64,) arrays, one per valid file
accum = [[] for _ in range(N_STEPS)]
skipped = 0

for path in csv_files:
    df = pd.read_csv(path)
    if not all(c in df.columns for c in ZONE_COLS):
        skipped += 1
        continue

    sig    = mean_signal(df)
    onsets = detect_step_onsets(sig)

    if len(onsets) != N_STEPS:
        print(f"  Skipping {os.path.basename(path)}: "
              f"detected {len(onsets)} steps (expected {N_STEPS})")
        skipped += 1
        continue

    zones = df[ZONE_COLS].values.astype(float)
    zones[zones == 0] = np.nan

    for step_idx in range(N_STEPS):
        p_start = onsets[step_idx] + SETTLE
        p_end   = onsets[step_idx + 1] - 2 if step_idx < N_STEPS - 1 else len(df)
        if p_start >= p_end:
            continue
        plateau_slice = zones[p_start:p_end]           # (frames, 64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            zone_avg = np.nanmean(plateau_slice, axis=0)  # (64,)
        accum[step_idx].append(zone_avg)

n_valid = len(accum[0])
print(f"Collected plateau data from {n_valid} recordings ({skipped} skipped).")

# ── Average across recordings ─────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    step_mean = np.array([np.nanmean(accum[s], axis=0) for s in range(N_STEPS)])  # (4, 64)
    step_std  = np.array([np.nanstd( accum[s], axis=0) for s in range(N_STEPS)])  # (4, 64)

grids_mean = step_mean.reshape(N_STEPS, 8, 8)
grids_std  = step_std.reshape(N_STEPS, 8, 8)

# ── Save CSV ──────────────────────────────────────────────────────────────────
rows = []
for step_idx in range(N_STEPS):
    row = {"step": step_idx + 1}
    for z in range(64):
        row[f"zone_{z:02d}_mean"] = round(float(step_mean[step_idx, z]), 2)
        row[f"zone_{z:02d}_std"]  = round(float(step_std[step_idx, z]),  2)
    rows.append(row)

out_csv = os.path.join(DATA_DIR, "step_heatmaps.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"Saved data  → {out_csv}")

# ── Plot ──────────────────────────────────────────────────────────────────────
# Shared colour scale across all 4 steps so they're directly comparable
vmin = np.nanmin(grids_mean)
vmax = np.nanmax(grids_mean)

fig, axes_grid = plt.subplots(
    2, 2,
    figsize=(7.16, 6.0),
    gridspec_kw={"hspace": 0.18, "wspace": 0.12},
)
axes = axes_grid.flatten()

for step_idx, ax in enumerate(axes):
    grid = grids_mean[step_idx]

    # pcolormesh renders clean, hair-thin cell borders — the standard
    # approach in publication-quality annotated heatmaps
    im = ax.pcolormesh(
        grid, cmap="viridis_r",
        vmin=vmin, vmax=vmax,
        edgecolors="white", linewidth=0.4,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Annotate each cell at its centre.
    # Pick text colour from the actual rendered RGBA → WCAG relative luminance,
    # so contrast is correct regardless of where in the colormap we land.
    cmap_fn = plt.cm.viridis_r
    norm    = plt.Normalize(vmin=vmin, vmax=vmax)

    for r in range(8):
        for c in range(8):
            val = grid[r, c]
            if np.isnan(val):
                continue
            rgba = cmap_fn(norm(val))
            # WCAG relative luminance
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "black" if luminance > 0.35 else "white"
            ax.text(c + 0.5, r + 0.5, f"{val:.0f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)

    ax.set_title(f"Step {step_idx + 1}", pad=6)

    # Ticks at cell centres
    ax.set_xticks(np.arange(0.5, 8.5))
    ax.set_yticks(np.arange(0.5, 8.5))
    ax.set_xticklabels(np.arange(8), fontsize=7)
    ax.set_yticklabels(np.arange(8), fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlim(0, 8)
    ax.set_ylim(8, 0)

    # Only label the outer edges of the 2×2 grid
    if step_idx >= 2:   # bottom row
        ax.set_xlabel("Column", labelpad=3)
    if step_idx % 2 == 0:  # left column
        ax.set_ylabel("Row", labelpad=3)

# Shared colourbar on the right
cbar = fig.colorbar(
    im, ax=axes.tolist(),
    fraction=0.02, pad=0.03, aspect=30,
)
cbar.set_label("Mean distance (mm)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.suptitle(
    f"Average 8$\\times$8 sensor reading per step plateau "
    f"($n = {n_valid}$ recordings)",
    fontsize=9, y=1.02,
)

out_png = os.path.join(DATA_DIR, "step_heatmaps.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved plot  → {out_png}")
plt.show()
