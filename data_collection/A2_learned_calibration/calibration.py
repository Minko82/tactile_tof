"""
calibration.py — learned, MEMORYLESS raw->mm calibration for the central-2x2 ToF
median, fitted against UR5 ground truth. Stage 1 of the two-stage architecture
(see A2_data_filter/PROS_CONS.md): because the correction is a per-sample
function of the current reading only, it removes the range-dependent bias with
exactly zero lag.

Fit (offline, from existing A3 filtertest runs; uses the RAW tof_log.csv +
robot_log.csv pairs, so it is independent of whatever mount offset each run
used in filter_log.csv):

    python3 calibration.py fit <surface>         all runs pooled under <surface>/
                                                 (e.g. wood/, white/); also saves
                                                 calibration_<surface>.json
    python3 calibration.py fit [run_dir ...]     explicit run dirs
    python3 calibration.py fit-generic           surface-independent model from
                                                 ALL pools in surfaces.json
                                                 -> calibration_generic.json
    python3 calibration.py show                  print the stored calibration

Only samples where the robot moved slowly (|v| <= FIT_MAX_SPEED_MM_S) are used:
the pose/frame pairing is worth a few mm of sync slop during fast motion, and a
fit that ingests fast-phase data learns that artifact as "bias".

Apply (live): Calibration.load(...).apply(median_mm) -> corrected mm. Inputs
outside the fitted raw span are clamped to the span's edge correction (the
polynomial is never extrapolated).
"""
import csv
import json
import math
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(os.path.dirname(HERE), "A3_proximity")
sys.path.insert(0, A3_DIR)
from robot import FT_CENTRAL, TABLE_Z_MM              # single source of truth

CALIB_PATH = os.path.join(HERE, "calibration.json")
DEGREE = 3                    # cubic: enough for ToF range walk, tame extrapolation
FIT_MAX_SPEED_MM_S = 100.0    # exclude fast motion (ground-truth sync slop)


class Calibration:
    """Polynomial raw->mm map with clamped (never-extrapolated) evaluation."""

    def __init__(self, coeffs, raw_min, raw_max, meta=None):
        self.coeffs = [float(c) for c in coeffs]      # np.polyval order (high first)
        self.raw_min, self.raw_max = float(raw_min), float(raw_max)
        self.meta = meta or {}

    def apply(self, raw_mm: float, signal=None) -> float:
        """Corrected distance. Outside the fitted span, the edge correction is
        carried along at slope 1 instead of extrapolating the polynomial.
        `signal` is accepted (and ignored) so per-surface and generic models
        share one call signature."""
        x = min(max(raw_mm, self.raw_min), self.raw_max)
        return float(np.polyval(self.coeffs, x)) + (raw_mm - x)

    def save(self, path=CALIB_PATH):
        with open(path, "w") as f:
            json.dump({"coeffs": self.coeffs, "raw_min": self.raw_min,
                       "raw_max": self.raw_max, "meta": self.meta}, f, indent=2)

    @classmethod
    def load(cls, path=CALIB_PATH):
        with open(path) as f:
            d = json.load(f)
        return cls(d["coeffs"], d["raw_min"], d["raw_max"], d.get("meta"))


GENERIC_PATH = os.path.join(HERE, "calibration_generic.json")

# 2026-07-17 remount shifted readings by a constant (+9.1/+9.5/+10.6 measured
# on white / black_shiny / matte_black). Runs recorded BEFORE the remount
# (auto-detected: they have no pose_log.csv) get their raw values shifted by
# this to express them in the current mount geometry. Wood is excluded from the
# estimate (its sheet also moved, delta +16.4).
EPOCH_OFFSET_MM = 9.7


def frame_tilt(z64):
    """Dimensionless tilt feature from one 8x8 distance frame: the plane-fit
    gradient magnitude normalized by mean distance (~= tan(incidence) x zone
    pitch). Monotone in true tilt (validated 0/5/10/15 deg); the regression
    learns its own scale. None if too few valid zones or too close."""
    z = np.asarray(z64, float)
    m = z > 0
    if m.sum() < 40:
        return None
    iz = np.arange(64)
    A = np.c_[np.ones(m.sum()), (iz % 8)[m] - 3.5, (iz // 8)[m] - 3.5]
    w, *_ = np.linalg.lstsq(A, z[m], rcond=None)
    if w[0] <= 60:                       # very close: mount geometry dominates
        return None
    return float(math.hypot(w[1], w[2]) / w[0])


class GenericCalibration:
    """Surface-independent (distance, signal, tilt) -> surface-distance model,
    trained across ALL material pools (thickness-corrected truth, epoch-aligned,
    angle sweeps included). Same .apply() signature family as the per-surface
    Calibration; tilt defaults to 0 (perpendicular) when not supplied."""

    @staticmethod
    def feats(r, ls, g):
        return [1.0, r, r * r, r ** 3, ls, r * ls, ls * ls, g, g * r, g * ls]

    def __init__(self, weights, raw_min, raw_max, ls_min, ls_max,
                 ls_default, g_max=0.25, meta=None):
        self.w = [float(x) for x in weights]
        self.raw_min, self.raw_max = float(raw_min), float(raw_max)
        self.ls_min, self.ls_max = float(ls_min), float(ls_max)
        self.ls_default = float(ls_default)      # used if a sample lacks signal
        self.g_max = float(g_max)
        self.meta = meta or {}

    def apply(self, raw_mm: float, signal=None, tilt=None) -> float:
        x = min(max(raw_mm, self.raw_min), self.raw_max)
        ls = (math.log(signal) if signal and signal > 0 else self.ls_default)
        ls = min(max(ls, self.ls_min), self.ls_max)
        g = min(max(tilt or 0.0, 0.0), self.g_max)
        return float(np.dot(self.w, self.feats(x, ls, g))) + (raw_mm - x)

    def save(self, path=GENERIC_PATH):
        with open(path, "w") as f:
            json.dump({"type": "generic", "weights": self.w,
                       "raw_min": self.raw_min, "raw_max": self.raw_max,
                       "ls_min": self.ls_min, "ls_max": self.ls_max,
                       "ls_default": self.ls_default, "g_max": self.g_max,
                       "meta": self.meta}, f, indent=2)

    @classmethod
    def load(cls, path=GENERIC_PATH):
        with open(path) as f:
            d = json.load(f)
        return cls(d["weights"], d["raw_min"], d["raw_max"], d["ls_min"],
                   d["ls_max"], d["ls_default"], d.get("g_max", 0.25),
                   d.get("meta"))


def _load_pool_with_signal(pool, thickness_mm):
    """Slow (raw_median, truth_to_surface_face, signal_median, tilt_feature)
    samples of a pool. Old-epoch runs (no pose_log.csv) get EPOCH_OFFSET_MM
    added to raw so all samples share the current mount geometry."""
    out = []
    for d in sorted(x for x in os.listdir(os.path.join(HERE, pool)) if x.isdigit()):
        run = os.path.join(HERE, pool, d)
        if not os.path.exists(os.path.join(run, "tof_log.csv")):
            continue
        with open(os.path.join(run, "tof_log.csv")) as f:
            header = f.readline().strip().split(",")
            if "s0" not in header:
                continue                                  # pre-signal firmware
            zi, si = header.index("z0"), header.index("s0")
            rows = list(csv.reader(f))
        epoch = 0.0 if os.path.exists(os.path.join(run, "pose_log.csv")) \
            else EPOCH_OFFSET_MM
        truth_z = _truth_z_fn(run)
        samples = []
        for r in rows:
            zc = [float(r[zi + i]) for i in FT_CENTRAL if float(r[zi + i]) > 0]
            sc = [float(r[si + i]) for i in FT_CENTRAL
                  if r[si + i] and float(r[si + i]) > 0]
            if len(zc) == 4 and sc:
                t = float(r[0])
                g = frame_tilt([float(r[zi + i]) for i in range(64)])
                samples.append((statistics.median(zc) + epoch,
                                truth_z(t) - TABLE_Z_MM - thickness_mm,
                                statistics.median(sc),
                                0.0 if g is None else g, t))
        for i in range(1, len(samples) - 1):
            dt = samples[i + 1][4] - samples[i - 1][4]
            if dt > 0 and abs((samples[i + 1][1] - samples[i - 1][1]) / dt) \
                    <= FIT_MAX_SPEED_MM_S:
                out.append(samples[i][:4])
    return out


def _generic_pools():
    """(pool_dir, thickness, base_surface) for every base pool in surfaces.json
    plus its angle sweeps (<base>_angle<deg>), which share the base thickness."""
    surf = json.load(open(os.path.join(HERE, "surfaces.json")))["surfaces"]
    out = []
    for base, props in surf.items():
        for d in sorted(os.listdir(HERE)):
            if d == base or (d.startswith(base + "_angle") and
                             os.path.isdir(os.path.join(HERE, d))):
                if os.path.isdir(os.path.join(HERE, d)):
                    out.append((d, props["thickness_mm"], base))
    return out


def fit_generic():
    """Fit the surface-independent model: all material pools + angle sweeps,
    epoch-aligned, with the 8x8-derived tilt feature."""
    X, y, used = [], [], set()
    for pool, th, base in _generic_pools():
        s = _load_pool_with_signal(pool, th)
        if not s:
            print(f"  {pool}: no signal-bearing runs, skipped")
            continue
        used.add(base)
        for r, t, sig, g in s:
            X.append((r, math.log(max(sig, 1.0)), g)); y.append(t)
        print(f"  {pool}: {len(s)} samples")
    raws = [r for r, _, _ in X]; lss = [ls for _, ls, _ in X]
    gs = [g for _, _, g in X]
    A = np.array([GenericCalibration.feats(r, ls, g) for r, ls, g in X])
    w, *_ = np.linalg.lstsq(A, np.array(y), rcond=None)
    resid = A @ w - np.array(y)
    gen = GenericCalibration(w, min(raws), max(raws), min(lss), max(lss),
                             statistics.median(lss), max(gs), meta={
        "n": len(y), "pools": sorted(used),
        "rmse_mm": float(np.sqrt(np.mean(resid ** 2))),
        "epoch_offset_mm": EPOCH_OFFSET_MM,
        "features": "raw, log(signal), tilt(8x8 plane fit)",
    })
    print(f"\ngeneric fit: {len(used)} materials (+angle sweeps), n={len(y)}, "
          f"pooled residual RMSE {gen.meta['rmse_mm']:.2f} mm")
    return gen


def load_for_surface(surface):
    """calibration_<surface>.json if fitted; else the GENERALIZED model (unknown
    materials get ~5 mm expected accuracy instead of an unbounded wrong-surface
    bias); else the old active-file fallback."""
    path = os.path.join(HERE, f"calibration_{surface}.json")
    if os.path.exists(path):
        return Calibration.load(path)
    if os.path.exists(GENERIC_PATH):
        gen = GenericCalibration.load()
        print(f"NOTE: no calibration_{surface}.json — using the GENERALIZED model "
              f"({len(gen.meta.get('pools', []))} materials, ~4-5 mm expected on "
              f"unseen surfaces). For mm-level accuracy collect 2-3 runs and:  "
              f"python3 calibration.py fit {surface}")
        return gen
    cal = Calibration.load()
    print(f"WARNING: no calibration_{surface}.json — falling back to the active "
          f"calibration.json (fitted for "
          f"'{cal.meta.get('surface', 'unknown')}'), so expect a constant bias.\n"
          f"  Collect 2-3 runs on '{surface}', then:  python3 calibration.py fit {surface}")
    return cal


def _truth_z_fn(run_dir):
    """Callable t -> robot z (mm), interpolated to the exact frame timestamp.

    Uses the full-rate pose_log.csv when the run has one (removes the
    latest-pose staleness jitter, worth up to ~8 mm during fast motion).
    Older runs fall back to robot_log.csv — whose rows share the frame
    timestamps exactly, so interpolation reproduces the old pairing bit-for-bit."""
    path = os.path.join(run_dir, "pose_log.csv")
    if not os.path.exists(path):
        path = os.path.join(run_dir, "robot_log.csv")
    rows = list(csv.reader(open(path)))[1:]
    ts = np.array([float(r[0]) for r in rows])
    zs = np.array([float(r[3]) for r in rows])
    order = np.argsort(ts)
    ts, zs = ts[order], zs[order]
    return lambda t: float(np.interp(t, ts, zs))


def _load_run(run_dir):
    """Frame-locked (raw_central_median_mm, truth_mm, time_s) rows of one run."""
    truth_z = _truth_z_fn(run_dir)
    tof = list(csv.reader(open(os.path.join(run_dir, "tof_log.csv"))))[1:]
    out = []
    for trow in tof:
        vals = [float(trow[1 + i]) for i in FT_CENTRAL if float(trow[1 + i]) > 0]
        if not vals:
            continue
        t = float(trow[0])
        out.append((statistics.median(vals),                 # raw median (no offset)
                    truth_z(t) - TABLE_Z_MM,                 # truth at frame time
                    t))
    return out


def _slow_samples(rows):
    """Keep samples where ground truth moved slowly (central-difference speed)."""
    kept = []
    for i in range(1, len(rows) - 1):
        dt = rows[i + 1][2] - rows[i - 1][2]
        if dt <= 0:
            continue
        v = (rows[i + 1][1] - rows[i - 1][1]) / dt
        if abs(v) <= FIT_MAX_SPEED_MM_S:
            kept.append(rows[i])
    return kept


def fit(run_dirs, degree=DEGREE):
    """Fit raw->truth over the slow samples of the given runs; returns Calibration."""
    samples = []
    for d in run_dirs:
        rows = _load_run(d)
        slow = _slow_samples(rows)
        samples += slow
        print(f"  {d}: {len(slow)}/{len(rows)} slow samples")
    if len(samples) < 50:
        raise SystemExit(f"only {len(samples)} usable samples — need more runs")
    raw = np.array([s[0] for s in samples])
    tru = np.array([s[1] for s in samples])
    coeffs = np.polyfit(raw, tru, degree)
    resid = np.polyval(coeffs, raw) - tru
    cal = Calibration(coeffs, raw.min(), raw.max(), meta={
        "degree": degree, "n": len(samples),
        "rmse_mm": float(np.sqrt(np.mean(resid ** 2))),
        "max_abs_resid_mm": float(np.max(np.abs(resid))),
        "fitted_from": [os.path.relpath(d, HERE) for d in run_dirs],
        "fit_max_speed_mm_s": FIT_MAX_SPEED_MM_S,
    })
    print(f"\nfit: degree {degree}, n={len(samples)}, span "
          f"[{cal.raw_min:.0f}, {cal.raw_max:.0f}] mm raw")
    print(f"residual RMSE {cal.meta['rmse_mm']:.2f} mm, "
          f"max |resid| {cal.meta['max_abs_resid_mm']:.2f} mm")
    return cal


def _pool_dirs(base):
    """Numbered run dirs under a surface pool folder (e.g. wood/1, wood/2 ...)."""
    return [os.path.join(base, d)
            for d in sorted((x for x in os.listdir(base) if x.isdigit()), key=int)
            if os.path.exists(os.path.join(base, d, "tof_log.csv"))]


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "show":
        cal = Calibration.load()
        print(json.dumps({"coeffs": cal.coeffs, "raw_min": cal.raw_min,
                          "raw_max": cal.raw_max, "meta": cal.meta}, indent=2))
    elif a and a[0] == "fit-generic":
        gen = fit_generic()
        gen.save()
        print(f"saved -> {GENERIC_PATH}")
    elif a and a[0] == "fit" and len(a) > 1:
        surface = None
        if len(a) == 2 and os.path.isdir(os.path.join(HERE, a[1])):
            surface = a[1]                       # pool shortcut: fit wood
            dirs = _pool_dirs(os.path.join(HERE, surface))
        else:
            dirs = a[1:]
        cal = fit(dirs)
        if surface:
            cal.meta["surface"] = surface
            cal.save(os.path.join(HERE, f"calibration_{surface}.json"))
            print(f"saved -> calibration_{surface}.json")
        cal.save()
        print(f"saved -> {CALIB_PATH}  (active)")
    else:
        print(__doc__)
