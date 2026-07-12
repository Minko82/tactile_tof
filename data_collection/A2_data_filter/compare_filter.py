#!/usr/bin/env python3
"""
compare_filter.py — raw ToF  vs  Kalman filter  vs  UR5, for whatever you select.

    python3 compare_filter.py raw_data/raw_1.csv       # static  -> raw vs Kalman (no UR5)
    python3 compare_filter.py test_data_A3/round_1      # tracked -> raw vs Kalman vs UR5
    python3 compare_filter.py test_data_A3              # every round in the folder

Takes the central 2x2 ToF zones as the distance signal and runs the live
constant-velocity Kalman filter (live_filter.py) over it. When the selection has
a robot_log.csv (UR5 ground truth), the raw signal is first translated onto the
UR5 scale (offset+scale) so all three overlay — calibrate BEFORE the Kalman — and
the residual error is drawn on a twin axis. Figures go to <selection>/figs/.
"""
import os, sys
try:                                        # numpy/matplotlib live in ~/ur5-env; re-exec there
    import numpy  # noqa: F401             # if launched with a bare python3
except ModuleNotFoundError:
    _v = os.path.expanduser("~/ur5-env/bin/python3")
    if os.path.exists(_v) and os.path.realpath(_v) != os.path.realpath(sys.executable):
        os.execv(_v, [_v, os.path.abspath(__file__), *sys.argv[1:]])
    sys.exit("compare_filter needs numpy + matplotlib — run:\n"
             "  ~/ur5-env/bin/python3 compare_filter.py <file-or-folder>")

import glob, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tof_sensor import load_round, load_raw, CENTRAL
from live_filter import LiveFilter

warnings.filterwarnings("ignore", message=".*tight_layout.*")
HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 200
C_RAW, C_KAL, C_GT, C_ERR = "#aeb6c0", "#3b4cc0", "#2a9d5c", "#d1354a"
STYLE = {"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11,
         "axes.edgecolor": "#c8ced6", "axes.linewidth": 1.0,
         "axes.grid": True, "grid.color": "#eceff3", "grid.linewidth": 1.0}


def _natkey(p):
    d = "".join(c for c in os.path.basename(p.rstrip("/")) if c.isdigit())
    return int(d) if d else 1 << 30


def resolve(arg):
    for base in ("", HERE, os.path.dirname(HERE)):        # cwd, A2 folder, dataset/
        c = arg if base == "" else os.path.join(base, arg)
        if os.path.exists(c):
            return os.path.abspath(c)
    sys.exit(f"compare_filter: not found: {arg}")


def _is_round(d):
    return (os.path.isfile(os.path.join(d, "robot_log.csv")) and
            os.path.isfile(os.path.join(d, "tof_log.csv")))


def targets(path):
    """-> list of (kind, name, path) where kind is 'round' (has UR5) or 'static'."""
    if os.path.isfile(path):
        return [("static", os.path.splitext(os.path.basename(path))[0], path)]
    if _is_round(path):
        return [("round", os.path.basename(path.rstrip("/")), path)]
    subs = sorted((os.path.join(path, d) for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d))), key=_natkey)
    rounds = [("round", os.path.basename(d), d) for d in subs if _is_round(d)]
    if rounds:
        return rounds
    csvs = [c for c in sorted(glob.glob(os.path.join(path, "*.csv")), key=_natkey)
            if os.path.basename(c) not in ("robot_log.csv", "tof_log.csv", "force_log.csv")]
    if csvs:
        return [("static", os.path.splitext(os.path.basename(c))[0], c) for c in csvs]
    sys.exit(f"compare_filter: no ToF data in {path}")


def central(F):
    z = np.nanmedian(F[:, CENTRAL], axis=1)
    ok = np.isfinite(z)
    if ok.any() and not ok.all():
        z = np.interp(np.arange(len(z)), np.where(ok)[0], z[ok])
    return z


def kalman(t, z):
    f = LiveFilter()
    return np.array([f.update(zi, ti) for zi, ti in zip(z, t)])


def _save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_round(name, folder, outdir):
    t, F, gt = load_round(folder)
    raw = central(F)
    a, b = np.polyfit(raw, gt, 1)                 # translate raw onto UR5 scale (gt ≈ a·raw + b)
    raw_t = a * raw + b                           # calibrate BEFORE the Kalman
    kal = kalman(t, raw_t)
    raw_err, kal_err = raw_t - gt, kal - gt
    rmse = float(np.sqrt(np.nanmean(kal_err ** 2)))
    # denoising = high-frequency (sample-to-sample) jitter removed, independent of the
    # sweep motion and the sensor's slow systematic error (which no filter can remove)
    jr, jk = float(np.nanstd(np.diff(raw_t))), float(np.nanstd(np.diff(kal)))

    # LEFT axis: tracking (UR5 vs Kalman overlap tightly).  RIGHT axis: the raw signal's
    # ±2 mm noise is invisible against a 600 mm sweep, so the denoising is shown as error.
    fig, ax = plt.subplots(figsize=(11, 6.0))
    l1, = ax.plot(t, gt, color=C_GT, lw=2.4, label="UR5 ground truth")
    l2, = ax.plot(t, kal, color=C_KAL, lw=1.7, label="Kalman filtered")
    ax.set_xlabel("time (s)"); ax.set_ylabel("distance above table (mm)"); ax.set_xlim(t[0], t[-1])
    ax.set_ylim(min(-30, np.nanmin(gt) - 20), max(np.nanmax(gt), np.nanmax(raw_t)) * 1.04)

    axe = ax.twinx()                              # error overlaid on its own zoomed scale
    l3, = axe.plot(t, raw_err, color=C_RAW, lw=.6, alpha=.9, label=f"raw ToF error  (σ {np.nanstd(raw_err):.2f} mm)")
    l4, = axe.plot(t, kal_err, color=C_ERR, lw=1.4, label=f"Kalman error  (σ {np.nanstd(kal_err):.2f} mm)")
    axe.axhline(0, color="#12263a", ls="--", lw=1.0, alpha=.55)
    cap = max(5.0, float(np.nanpercentile(np.abs(raw_err), 99)))
    axe.set_ylim(-cap, cap); axe.set_ylabel("error  ToF − UR5  (mm)", color=C_ERR)
    axe.tick_params(axis="y", colors=C_ERR); axe.spines["right"].set_color(C_ERR); axe.grid(False)

    ax.set_title(f"Raw vs Kalman vs UR5 — {name}    ·    tracks UR5 to {rmse:.2f} mm RMSE    ·    "
                 f"{jr / jk:.1f}× smoother (jitter {jr:.2f}→{jk:.2f} mm)",
                 fontweight="bold", loc="left", fontsize=11.5)
    ax.legend(handles=[l1, l2, l3, l4], loc="upper right", framealpha=.95, fontsize=9)
    _save(fig, outdir, f"compare_{name}.png")
    return (f"compare_{name}.png  (RMSE {rmse:.2f} mm vs UR5, jitter {jr:.2f}→{jk:.2f} mm, "
            f"{jr / jk:.1f}× smoother)")


def fig_static(name, path, outdir):
    t, F = load_raw(path)
    raw = central(F)
    kal = kalman(t, raw)
    jr, jf = np.nanstd(raw), np.nanstd(kal)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(t, raw, ".", ms=2, alpha=.35, color="#3b7dd8", label=f"raw ToF  (σ {jr:.2f} mm)")
    ax.plot(t, kal, "-", lw=1.8, color=C_ERR, label=f"Kalman filtered  (σ {jf:.2f} mm, {jr / jf:.1f}× less)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("central distance (mm)"); ax.set_xlim(t[0], t[-1])
    ax.set_title(f"Raw vs Kalman — {name}  (static, no UR5)", fontweight="bold", loc="left")
    ax.legend(loc="upper right", framealpha=.95)
    _save(fig, outdir, f"compare_{name}.png")
    return f"compare_{name}.png  ({jr / jf:.1f}× less jitter, {jr:.2f}→{jf:.2f} mm)"


def main(argv):
    if len(argv) != 1:
        sys.exit("usage: compare_filter.py <file-or-folder>   e.g.  test_data_A3/round_1")
    path = resolve(argv[0])
    tgts = targets(path)
    root = path if os.path.isdir(path) else os.path.dirname(path)
    outdir = os.path.join(root, "figs"); os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update(STYLE)
    print(f"compare_filter: {os.path.relpath(path, HERE)}  ({len(tgts)} item/s)  ->  "
          f"{os.path.relpath(outdir, os.getcwd())}")
    for kind, name, p in tgts:
        try:
            print("  •", (fig_round if kind == "round" else fig_static)(name, p, outdir))
        except Exception as e:
            print(f"  ! {name} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
