#!/usr/bin/env python3
"""
analysis.py — translated ToF vs UR5, plus repeatability + per-round error.

    python3 analysis.py smooth
    python3 analysis.py steps

Averages every round in the folder, translates the raw central-zone ToF onto the
UR5 scale (offset+scale fit from the data), and writes to <folder>/figs/:

  average_overlay.png     translated ToF vs UR5 ground truth, residual error overlaid
  repeatability.png       all sweeps overlaid + run-to-run spread
  per_round_error.png     per-round MAE / bias   (+ per_round_error.csv)
  error_vs_position.png   error vs UR5 position (descending vs ascending)

Self-contained: numpy + matplotlib only.  (For the raw-vs-Kalman-vs-UR5 filter
view, see A2_data_filter/compare_filter.py.)
"""
import os, sys
try:                                        # numpy/matplotlib live in ~/ur5-env; re-exec there
    import numpy  # noqa: F401             # if launched with a bare python3
except ModuleNotFoundError:
    _v = os.path.expanduser("~/ur5-env/bin/python3")
    if os.path.exists(_v) and os.path.realpath(_v) != os.path.realpath(sys.executable):
        os.execv(_v, [_v, os.path.abspath(__file__), *sys.argv[1:]])
    sys.exit("analysis needs numpy + matplotlib — run:\n"
             "  ~/ur5-env/bin/python3 analysis.py <folder>")

import csv, glob, math, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*tight_layout.*")
HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_Z_MM = -143.30                        # UR5 table height (robot.py `table`)
CENTRAL = (27, 28, 35, 36)                  # central 2x2 zones -> straight down
N_GRID = 760
DPI = 200
C_GT, C_SN, C_ERR = "#e8871a", "#2f7fd6", "#d1354a"
C_DOWN, C_UP, C_ACC = "#2f7fd6", "#e8871a", "#12263a"
STYLE = {"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11,
         "axes.edgecolor": "#c8ced6", "axes.linewidth": 1.0,
         "axes.grid": True, "grid.color": "#eceff3", "grid.linewidth": 1.0}


def _natkey(p):
    d = "".join(c for c in os.path.basename(p.rstrip("/")) if c.isdigit())
    return int(d) if d else 1 << 30


def load_round(folder):
    """-> (t, central-2x2 median ToF, UR5 height above table), gt interpolated onto ToF time."""
    with open(os.path.join(folder, "robot_log.csv")) as f:
        r = csv.reader(f); next(r); rob = [x for x in r if x]
    with open(os.path.join(folder, "tof_log.csv")) as f:
        r = csv.reader(f); next(r); tof = [x for x in r if x]
    tr = np.array([float(x[0]) for x in rob])
    gr = np.array([float(x[3]) - TABLE_Z_MM for x in rob])         # z_mm - table
    ts = np.array([float(x[0]) for x in tof])
    sens = []
    for row in tof:
        vals = [int(row[1 + i]) for i in CENTRAL]                  # tof cols: time, z0, z1, ...
        vals = [v for v in vals if v > 0]                          # -1 = no target
        sens.append(np.median(vals) if vals else np.nan)
    return ts, np.array(sens), np.interp(ts, tr, gr)


def _fill_nan(y):
    ok = np.isfinite(y)
    if ok.all() or not ok.any():
        return y
    i = np.arange(len(y))
    return np.interp(i, i[ok], y[ok])


def build(folder):
    dirs = sorted((os.path.join(folder, d) for d in os.listdir(folder)
                   if os.path.isdir(os.path.join(folder, d))
                   and os.path.isfile(os.path.join(folder, d, "robot_log.csv"))), key=_natkey)
    if not dirs:
        sys.exit(f"analysis: no round subfolders in {folder}")
    rounds = [load_round(d) for d in dirs]
    names = [os.path.basename(d) for d in dirs]
    t_end = min(t[-1] for (t, s, g) in rounds)
    grid = np.linspace(0, t_end, N_GRID)
    GT = np.vstack([np.interp(grid, t, g) for (t, s, g) in rounds])
    SN = np.vstack([np.interp(grid, t, _fill_nan(s)) for (t, s, g) in rounds])
    m = np.isfinite(GT.ravel()) & np.isfinite(SN.ravel())
    a, b = np.polyfit(GT.ravel()[m], SN.ravel()[m], 1)             # raw ToF = a·height + b
    return dict(names=names, grid=grid, t_end=t_end, GT=GT, SN=SN,
                SNc=(SN - b) / a, a=float(a), b=float(b), R=len(rounds))


def _save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_overlay(D, outdir):
    grid, GT, SN = D["grid"], D["GT"], D["SNc"]
    ERR = SN - GT
    gt_m, gt_s, sn_m, sn_s, er_m = GT.mean(0), GT.std(0), SN.mean(0), SN.std(0), ERR.mean(0)
    mae, rmse, bias = np.abs(ERR).mean(), math.sqrt((ERR ** 2).mean()), ERR.mean()

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.fill_between(grid, gt_m - gt_s, gt_m + gt_s, color=C_GT, alpha=.15, lw=0)
    ax.fill_between(grid, sn_m - sn_s, sn_m + sn_s, color=C_SN, alpha=.15, lw=0)
    l1, = ax.plot(grid, gt_m, color=C_GT, lw=2.6, label="UR5 ground truth (height above table)")
    l2, = ax.plot(grid, sn_m, color=C_SN, lw=2.6, label="ToF, translated (offset + scale)")
    ax.set_xlabel("time  (s)"); ax.set_ylabel("distance above table  (mm)")
    ax.set_xlim(0, D["t_end"]); ax.set_ylim(-20, max(sn_m.max(), gt_m.max()) * 1.06)

    axe = ax.twinx()
    l3, = axe.plot(grid, er_m, color=C_ERR, lw=2.0, label="residual error  ToF − UR5")
    l4 = axe.axhline(0, color="#12263a", ls="--", lw=1.2, alpha=.7, label="zero error")
    axe.set_ylim(-7, 7); axe.set_ylabel("residual error  ToF − UR5  (mm)", color=C_ERR)
    axe.tick_params(axis="y", colors=C_ERR); axe.spines["right"].set_color(C_ERR); axe.grid(False)

    ax.set_title(f"ToF vs. UR5 — {D['tag']}  (n = {D['R']})", fontsize=14, fontweight="bold", pad=12)
    ax.legend(handles=[l1, l2, l3, l4], loc="upper center", ncol=2, framealpha=.95, fontsize=10)
    ax.text(.985, .03,
            f"ToF' = (ToF − {D['b']:.2f}) / {D['a']:.4f}\n"
            f"MAE   {mae:5.2f} mm\nRMSE  {rmse:5.2f} mm\nbias  {bias:+5.2f} mm\n"
            f"n = {D['R']} rounds × {N_GRID} samples",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            family="DejaVu Sans Mono", bbox=dict(boxstyle="round,pad=0.5", fc="#f7f9fb", ec="#d5dae1"))
    _save(fig, outdir, "average_overlay.png")
    return f"average_overlay.png  (MAE {mae:.2f}, RMSE {rmse:.2f}, bias {bias:+.2f} mm)"


def fig_repeatability(D, outdir):
    if D["R"] < 2:
        return "repeatability.png  — skipped (need ≥2 rounds)"
    grid, SN = D["grid"], D["SNc"]
    sn_m, sn_s = SN.mean(0), SN.std(0)
    p95 = np.percentile(np.abs(SN - sn_m), 95)

    fig, ax = plt.subplots(figsize=(11, 6.0))
    for i in range(D["R"]):
        ax.plot(grid, SN[i], color=C_SN, alpha=.12, lw=.6)
    ax.plot(grid, sn_m, color="#12263a", lw=2.0, label=f"mean of {D['R']} sweeps")
    ax.plot([], [], color=C_SN, lw=1.8, alpha=.6, label=f"all {D['R']} samples overlaid")
    ax.set_xlabel("time  (s)"); ax.set_ylabel("ToF distance, translated  (mm)")
    ax.set_xlim(0, D["t_end"]); ax.set_ylim(-20, sn_m.max() * 1.06)
    ax.set_title(f"All {D['R']} sweeps overlaid — run-to-run difference — {D['tag']}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower left", framealpha=.95, fontsize=10)
    ax.text(.5, .60,
            f"Run-to-run difference across {D['R']} samples\n"
            f"±{sn_s.mean():.2f} mm  (1σ)        ±{p95:.2f} mm  (95%)",
            transform=ax.transAxes, ha="center", va="center", fontsize=14, fontweight="bold",
            color="#12263a", bbox=dict(boxstyle="round,pad=0.7", fc="#f7f9fb", ec="#c8ced6"))
    _save(fig, outdir, "repeatability.png")
    return f"repeatability.png  (1σ {sn_s.mean():.2f} mm, 95% ±{p95:.2f} mm)"


def fig_per_round(D, outdir):
    err = D["SNc"] - D["GT"]
    mae_i, bias_i, rmse_i = np.abs(err).mean(1), err.mean(1), np.sqrt((err ** 2).mean(1))
    with open(os.path.join(outdir, "per_round_error.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["round", "mae_mm", "bias_mm", "rmse_mm"])
        for nm, a, b, c in zip(D["names"], mae_i, bias_i, rmse_i):
            w.writerow([nm, f"{a:.2f}", f"{b:+.2f}", f"{c:.2f}"])

    xr = np.arange(1, D["R"] + 1)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(xr, mae_i, color=C_SN, lw=1.8, marker="o", ms=3.5, label="MAE per round  (|error|)")
    ax.plot(xr, bias_i, color=C_GT, lw=1.5, alpha=.9, label="bias per round  (signed)")
    ax.axhline(mae_i.mean(), color=C_SN, ls="--", lw=1.3, alpha=.8, label=f"overall MAE = {mae_i.mean():.2f} mm")
    ax.axhline(0, color="#12263a", lw=1.0, alpha=.5)
    ax.set_xlabel("round #"); ax.set_ylabel("error vs. UR5 position  (mm)"); ax.set_xlim(.5, D["R"] + .5)
    ax.set_title(f"Average ToF error per round — {D['tag']}  (n = {D['R']})",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=.95, fontsize=9.5)
    _save(fig, outdir, "per_round_error.png")
    return (f"per_round_error.png (+csv)  (overall MAE {mae_i.mean():.2f} mm, "
            f"range {mae_i.min():.2f}..{mae_i.max():.2f})")


def fig_error_vs_position(D, outdir):
    pos = D["GT"].mean(0)
    err = (D["SNc"] - D["GT"]).mean(0); ers = (D["SNc"] - D["GT"]).std(0)
    a, b = D["a"], D["b"]
    jmin = int(np.argmin(pos)); dn = slice(0, jmin + 1); up = slice(jmin, None)

    fig, ax = plt.subplots(figsize=(11, 6.3))
    ax.axhline(0, color=C_ACC, lw=1.0, alpha=.6)
    ax.fill_between(pos[dn], err[dn] - ers[dn], err[dn] + ers[dn], color=C_DOWN, alpha=.13, lw=0)
    ax.fill_between(pos[up], err[up] - ers[up], err[up] + ers[up], color=C_UP, alpha=.13, lw=0)
    ax.plot(pos[dn], err[dn], color=C_DOWN, lw=2.4, label="descending  (top → table)")
    ax.plot(pos[up], err[up], color=C_UP, lw=2.4, label="ascending  (table → top)")
    ax.set_xlabel("UR5 position — height above table  (mm)")
    ax.set_ylabel("mean ToF error   ToF − UR5   (mm)")
    ax.set_title(f"ToF error vs. UR5 position — {D['tag']}  (n = {D['R']})",
                 fontsize=14, fontweight="bold", pad=28)
    ax.legend(loc="upper right", framealpha=.95, fontsize=10)
    ax.secondary_xaxis("top", functions=(lambda p: a * p + b, lambda s: (s - b) / a)) \
      .set_xlabel("corresponding raw ToF reading  (mm)", labelpad=6)
    ax.text(.015, .03, f"raw offset  b = {b:.1f} mm   scale  a = {a:.4f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
            family="DejaVu Sans Mono", bbox=dict(boxstyle="round,pad=0.5", fc="#f7f9fb", ec="#d5dae1"))
    _save(fig, outdir, "error_vs_position.png")
    return f"error_vs_position.png  (|error| up to {np.abs(err).max():.2f} mm)"


def main(argv):
    if len(argv) != 1:
        sys.exit("usage: analysis.py <folder>   e.g.  smooth   or   steps")
    folder = argv[0] if os.path.isdir(argv[0]) else os.path.join(HERE, argv[0])
    if not os.path.isdir(folder):
        sys.exit(f"analysis: folder not found: {argv[0]}")
    D = build(folder); D["tag"] = os.path.basename(folder.rstrip("/"))
    outdir = os.path.join(folder, "figs"); os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update(STYLE)
    print(f"analysis: {D['tag']}  ({D['R']} rounds)  ->  {os.path.relpath(outdir, os.getcwd())}")
    print(f"  translation  ToF' = (ToF − {D['b']:.2f}) / {D['a']:.4f}")
    for fn in (fig_overlay, fig_repeatability, fig_per_round, fig_error_vs_position):
        try:
            print("  •", fn(D, outdir))
        except Exception as e:
            print(f"  ! {fn.__name__} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
