#!/usr/bin/env python3
"""
Object-shape ToF visualizer (tof_mounted dataset)
-------------------------------------------------
The intern placed small black objects (cup, spoon, stair, star, torus) on the
flat cutting mat and swept the UR5 up/down. Each object shows up two ways in the
8x8 ToF frame:
  * a DISTANCE anomaly (raised objects read nearer; the mat reads flat), and
  * an IR-ABSORPTION anomaly (black surfaces drop out -> fewer valid returns).

So unlike the flat-surface ranging analysis, here we:
  * filter GATE-ONLY (no MAD) — the MAD outlier step would discard the very
    object zones we want to see;
  * robustly plane-DETREND each step (fit + refit on inliers) to remove the mat's
    slight tilt, leaving the object as a residual;
  * also map per-zone valid-fraction, where black objects appear as dropouts.

Outputs per object/direction: ``step_heatmaps.png`` (object residual at every
step). Plus a top-level ``shapes_montage.png`` pairing each setup photo with the
ToF depth-residual and dropout map at the closest step.

Run:
    .venv/bin/python data_collection/visualize_shapes.py
"""

import os
import glob
import math
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tof_common import (
    load_readings, load_ground_truth, detect_steps, GRID, GATE_ONLY_LABEL,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "test_results", "tof_mounted")
IMG_DIR = os.path.join(ROOT, "images")
SHAPES = ["cup", "spoon", "stair", "star", "torus"]

_YS, _XS = np.mgrid[0:GRID, 0:GRID]


def robust_plane_detrend(zmap: np.ndarray) -> np.ndarray:
    """Subtract a robust best-fit plane (the tilted mat) -> object residual (mm).

    Fits a plane to all valid zones, then refits on inliers (rejecting the object)
    so the residual isolates the object instead of being dragged toward it.
    """
    finite = np.isfinite(zmap)
    if finite.sum() < 4:
        return zmap - np.nanmedian(zmap)

    def fit(mask):
        A = np.column_stack([_XS[mask], _YS[mask], np.ones(mask.sum())])
        coef, *_ = np.linalg.lstsq(A, zmap[mask], rcond=None)
        return coef[0] * _XS + coef[1] * _YS + coef[2]

    plane = fit(finite)
    resid = zmap - plane
    r = resid[finite]
    mad = np.median(np.abs(r - np.median(r)))
    if mad > 0:
        inliers = finite & (np.abs(resid - np.median(r)) <= 2.5 * 1.4826 * mad)
        if inliers.sum() >= 4:
            plane = fit(inliers)
            resid = zmap - plane
    return resid


def step_panels(shape, direction):
    sf = glob.glob(os.path.join(ROOT, f"{shape}_{direction}",
                                f"vl53l5cx_{shape}_{direction}*.csv"))
    uf = glob.glob(os.path.join(ROOT, f"{shape}_{direction}",
                                f"ur5_{shape}_{direction}*.csv"))
    if not sf or not uf:
        return None, None
    secs, grid, t0 = load_readings(sf[0], apply_mad=False)   # gate-only
    g_t, g_z = load_ground_truth(uf[0], t0)
    steps = detect_steps(g_t, g_z)

    panels = []
    for k, (a, b, level) in enumerate(steps, start=1):
        m = (secs >= a) & (secs <= b)
        if not m.any():
            continue
        sub = grid[m]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            zmap = np.nanmean(sub, axis=0).reshape(GRID, GRID)
        vf = np.isfinite(sub).mean(axis=0).reshape(GRID, GRID)
        panels.append({"k": k, "level": level,
                       "resid": robust_plane_detrend(zmap), "vf": vf})
    folder = os.path.join(ROOT, f"{shape}_{direction}")
    return panels, folder


def object_mask(p):
    """8x8 'is-object' strength in [0,1]: 1 = black object (IR dropout), 0 = table."""
    return 1.0 - p["vf"]


def object_score(p):
    """How strongly an object is visible at this step (dropouts dominate, depth helps)."""
    drop = int((p["vf"] < 0.5).sum())                       # black-object dropouts
    resid = p["resid"]
    dep = int(np.sum(np.isfinite(resid) & (np.abs(resid) > 5.0)))  # raised/sunken zones
    return drop * 2 + dep


def best_panel(panels):
    return max(panels, key=object_score)


def make_step_grid(shape, direction):
    """Per-step object silhouette (black object on the white table) across the approach."""
    panels, folder = step_panels(shape, direction)
    if not panels:
        return False

    n = len(panels)
    ncol = math.ceil(math.sqrt(n))
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol + 1.0, 2.6 * nrow + 0.9))
    axes = np.atleast_1d(axes).ravel()

    best = best_panel(panels)
    im = None
    for ax, p in zip(axes, panels):
        # gray: object (low valid-fraction) -> black, table -> white, like the photo
        im = ax.imshow(p["vf"], cmap="gray", vmin=0, vmax=1,
                       origin="upper", interpolation="nearest")
        tag = "  ◀ clearest" if p is best else ""
        ax.set_title(f"step {p['k']} • UR5 z={p['level']:.0f} mm{tag}",
                     fontsize=8, fontweight=("bold" if p is best else "normal"))
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{shape.capitalize()} ({direction}) — object silhouette as the sensor "
                 f"approaches\n(black = IR dropout = black object; white = table)",
                 fontsize=12, fontweight="bold")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("valid-return fraction")
    fig.text(0.5, 0.01, f"Filter: {GATE_ONLY_LABEL}.", ha="center", fontsize=8, color="#555")
    fig.savefig(os.path.join(folder, "step_heatmaps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def make_montage(direction="descending"):
    """One row per shape: setup photo | object silhouette | depth residual (best step)."""
    rows = []
    for shape in SHAPES:
        panels, _ = step_panels(shape, direction)
        if not panels:
            continue
        p = best_panel(panels)
        img = os.path.join(IMG_DIR, f"{shape}_{direction}.JPEG")
        rows.append({"shape": shape, "panel": p,
                     "img": img if os.path.exists(img) else None})
    if not rows:
        return

    vlim = max(8.0, np.nanpercentile(np.concatenate(
        [r["panel"]["resid"][np.isfinite(r["panel"]["resid"])] for r in rows]), 98))

    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.2 * len(rows)))
    axes = np.atleast_2d(axes)
    dcmap = plt.cm.RdBu_r.copy(); dcmap.set_bad("#000000")   # dropout -> black

    for i, r in enumerate(rows):
        axp, axs, axd = axes[i]
        p = r["panel"]
        if r["img"]:
            axp.imshow(mpimg.imread(r["img"]))
        axp.set_ylabel(f"{r['shape'].capitalize()}\nz={p['level']:.0f} mm",
                       fontsize=11, fontweight="bold")
        axp.set_xticks([]); axp.set_yticks([])
        if i == 0:
            axp.set_title("setup photo", fontsize=10)

        # object silhouette — bilinear-smoothed for legibility at 8x8
        axs.imshow(p["vf"], cmap="gray", vmin=0, vmax=1,
                   origin="upper", interpolation="bilinear")
        axs.set_xticks([]); axs.set_yticks([])
        if i == 0:
            axs.set_title("ToF object silhouette\n(black = object, smoothed 8×8)", fontsize=10)

        im = axd.imshow(p["resid"], cmap=dcmap, vmin=-vlim, vmax=vlim,
                        origin="upper", interpolation="nearest")
        axd.set_xticks([]); axd.set_yticks([])
        if i == 0:
            axd.set_title("depth residual (mm)\n(blue=nearer, black=dropout)", fontsize=10)
        fig.colorbar(im, ax=axd, fraction=0.046, pad=0.04)

    fig.suptitle(f"ToF object shapes — clearest step per object ({direction})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.text(0.5, 0.005, f"Filter: {GATE_ONLY_LABEL}.  Silhouette from IR dropouts "
             f"(black objects absorb IR on the white table); depth residual = distance − "
             f"robust table-plane fit.", ha="center", fontsize=8, color="#555")
    out = os.path.join(ROOT, "shapes_montage.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Montage written to {out}")


def main():
    for shape in SHAPES:
        for direction in ("ascending", "descending"):
            ok = make_step_grid(shape, direction)
            print(f"  {'✓' if ok else '—'} {shape}_{direction}")
    make_montage("descending")


if __name__ == "__main__":
    main()
