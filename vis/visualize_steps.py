"""
visualize_steps.py — Overlaid staircase plot for all recordings.

Detects the 4 step onsets in each CSV, aligns them so that x=0 is the
start of the first step, and overlays all 50 signals on one plot.
The trace ends a fixed number of frames after the 4th step onset.

Usage:
    python3 visualize_steps.py

Output:
    test_results/staircase_overlay.png
"""

import glob
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
N_STEPS        = 4    # expected arm step-downs per recording
TAIL_FRAMES    = 20   # frames to include after the last step onset
SMOOTH_WINDOW  = 5    # rolling-average window for onset detection
MIN_STEP_GAP   = 8    # minimum frames between two step onsets
RANGING_HZ     = 10   # sensor framerate — used to convert frames → seconds

# UR5 arm parameters (all in mm)
ARM_START_MM   = 310    # initial arm distance above sensor (mm)
ARM_STEP_MM    = 50     # decrement per step (mm)
ARM_WAIT_S     = 2.0    # commanded dwell time at each plateau (s)
# Commanded arm position during each plateau: [260, 210, 160, 110] mm
ARM_PLATEAU_MM = [ARM_START_MM - (i + 1) * ARM_STEP_MM for i in range(N_STEPS)]

ZONE_COLS   = [f"zone_{i:02d}" for i in range(64)]
STEP_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]  # ColorBrewer Set1

# ── Scholarly figure style (ICRA / IEEE) ──────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#d0d0d0",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
})


# ── Helpers (shared with analyze.py) ─────────────────────────────────────────
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


# ── Load and align every recording ───────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "readings_*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No readings_*.csv found in {DATA_DIR}")
print(f"Found {len(csv_files)} files.")

traces        = []   # list of (x_offsets, y_signal) tuples
step_offsets  = []   # list of [s1_off, s2_off, s3_off, s4_off] per file
skipped       = 0

for path in csv_files:
    df = pd.read_csv(path)
    if not all(c in df.columns for c in ZONE_COLS):
        skipped += 1
        continue

    sig    = mean_signal(df)
    onsets = detect_step_onsets(sig)

    if len(onsets) != N_STEPS:
        print(f"  Skipping {os.path.basename(path)}: "
              f"detected {len(onsets)} steps, expected {N_STEPS}")
        skipped += 1
        continue

    first = onsets[0]
    last  = onsets[-1]
    end   = min(last + TAIL_FRAMES, len(sig))

    y = sig[first:end]
    x = np.arange(len(y))   # x=0 is the first step onset

    traces.append((x, y))
    step_offsets.append([o - first for o in onsets])   # relative to first onset

print(f"Aligned {len(traces)} recordings ({skipped} skipped).")

# ── Compute average step onset positions ─────────────────────────────────────
mean_offsets = np.mean(step_offsets, axis=0).astype(int)   # [0, s2, s3, s4]

# ── Interpolate all traces to a common x grid for averaging ──────────────────
max_x = max(x[-1] for x, _ in traces)
common_x = np.arange(max_x + 1)
interp_traces = []
for x, y in traces:
    if x[-1] < max_x:
        # Pad with the final value so all traces reach max_x
        y_pad = np.pad(y, (0, max_x - x[-1]), mode="edge")
        interp_traces.append(y_pad)
    else:
        interp_traces.append(y[:max_x + 1])

stack   = np.array(interp_traces)   # (n_files, max_x+1)
avg_y   = np.nanmean(stack, axis=0)
std_y   = np.nanstd(stack,  axis=0)

# ── Convert frames → seconds for plotting ────────────────────────────────────
common_s      = common_x / RANGING_HZ
mean_offsets_s = mean_offsets / RANGING_HZ

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.16, 3.5))

# Individual trials — light grey, behind everything
for x, y in traces:
    ax.plot(x / RANGING_HZ, y, color="#b0b0b0", linewidth=0.5, alpha=0.35, zorder=1)

# Mean ± 1 std band
ax.fill_between(common_s, avg_y - std_y, avg_y + std_y,
                color="#2166ac", alpha=0.12, zorder=2)
ax.plot(common_s, avg_y, color="#2166ac", linewidth=1.6, zorder=3)


# Step onset verticals — labelled inline at the top of each line
for step_idx in range(N_STEPS):
    offset_s = mean_offsets_s[step_idx]
    ax.axvline(offset_s, color=STEP_COLORS[step_idx], linewidth=0.9,
               linestyle="--", alpha=0.9)
    ax.text(offset_s + common_s[-1] * 0.012, 0.97,
            f"Step {step_idx + 1}",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=7.5, fontweight="bold",
            color=STEP_COLORS[step_idx])

# ── Plateau distance annotations ──────────────────────────────────────────────
SETTLE = 8   # frames after onset before measuring the plateau

plateau_bounds = []
for step_idx in range(N_STEPS):
    p_start = int(mean_offsets[step_idx]) + SETTLE
    p_end   = int(mean_offsets[step_idx + 1]) - 2 if step_idx < N_STEPS - 1 else max_x
    plateau_bounds.append((p_start, p_end))

for step_idx, (p_start, p_end) in enumerate(plateau_bounds):
    if p_start >= p_end or p_end > len(avg_y):
        continue

    p_start_s   = p_start / RANGING_HZ
    p_end_s     = p_end   / RANGING_HZ
    mid_s       = (p_start_s + p_end_s) / 2
    plateau_val = float(np.nanmean(avg_y[p_start:p_end]))
    col         = STEP_COLORS[step_idx]

    ax.hlines(plateau_val, p_start_s, p_end_s,
              colors=col, linewidth=1.2, linestyle="-", alpha=1.0, zorder=4)

    # Sensor plateau label — centered on plateau span
    ax.annotate(
        f"{plateau_val:.0f} mm",
        xy=(mid_s, plateau_val),
        xytext=(0, 5),
        textcoords="offset points",
        color=col, fontsize=7, fontstyle="italic",
        va="bottom", ha="center",
    )

    # UR5 commanded position line
    ur5_mm = ARM_PLATEAU_MM[step_idx]
    ax.hlines(ur5_mm, p_start_s, p_end_s,
              colors="#444444", linewidth=1.0, linestyle=":",
              alpha=0.85, zorder=4)

    # UR5 label — centered on plateau span
    ax.annotate(
        f"{ur5_mm} mm",
        xy=(mid_s, ur5_mm),
        xytext=(0, 5),
        textcoords="offset points",
        color="#444444", fontsize=7, fontstyle="italic",
        va="bottom", ha="center",
    )

# Initial resting distance — labeled in the flat region before the first drop
initial_val = float(np.nanmean(avg_y[:mean_offsets[0] + 1]))
pre_mid_s   = (mean_offsets_s[0]) / 2
ax.annotate(
    f"{initial_val:.0f} mm",
    xy=(pre_mid_s, initial_val),
    xytext=(0, 5),
    textcoords="offset points",
    color="#333333", fontsize=7, fontstyle="italic",
    va="bottom", ha="center",
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mlines.Line2D([], [], color="#2166ac", linewidth=1.6,
                  label=f"Sensor mean ($n={len(traces)}$)"),
    mpatches.Patch(facecolor="#2166ac", alpha=0.3, edgecolor="none",
                   label="$\\pm$1 s.d."),
    mlines.Line2D([], [], color="#b0b0b0", linewidth=0.8, alpha=0.6,
                  label="Individual trials"),
    mlines.Line2D([], [], color="#444444", linewidth=1.0, linestyle=":",
                  label="UR5 commanded pos."),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
          handlelength=2.0, labelspacing=0.4)

# ── Axes styling ──────────────────────────────────────────────────────────────
ax.set_xlabel("Time from first step onset (s)")
ax.set_ylabel("Mean distance across zones (mm)")
ax.set_xlim(common_s[0], common_s[-1])

ax.tick_params(direction="in", length=3)

plt.tight_layout(pad=0.5)

out_path = os.path.join(DATA_DIR, "staircase_overlay.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
