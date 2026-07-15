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


class GenericCalibration:
    """Surface-independent (distance, signal) -> surface-distance model, trained
    across ALL material pools with thickness-corrected ground truth. Expected
    accuracy on unseen materials: ~4-5 mm RMSE typical (LOSO), worst observed
    ~13 mm on extreme subsurface scatterers. Same .apply() signature as the
    per-surface Calibration."""

    @staticmethod
    def feats(r, ls):
        return [1.0, r, r * r, r ** 3, ls, r * ls, ls * ls]

    def __init__(self, weights, raw_min, raw_max, ls_min, ls_max,
                 ls_default, meta=None):
        self.w = [float(x) for x in weights]
        self.raw_min, self.raw_max = float(raw_min), float(raw_max)
        self.ls_min, self.ls_max = float(ls_min), float(ls_max)
        self.ls_default = float(ls_default)      # used if a sample lacks signal
        self.meta = meta or {}

    def apply(self, raw_mm: float, signal=None) -> float:
        x = min(max(raw_mm, self.raw_min), self.raw_max)
        ls = (math.log(signal) if signal and signal > 0 else self.ls_default)
        ls = min(max(ls, self.ls_min), self.ls_max)
        return float(np.dot(self.w, self.feats(x, ls))) + (raw_mm - x)

    def save(self, path=GENERIC_PATH):
        with open(path, "w") as f:
            json.dump({"type": "generic", "weights": self.w,
                       "raw_min": self.raw_min, "raw_max": self.raw_max,
                       "ls_min": self.ls_min, "ls_max": self.ls_max,
                       "ls_default": self.ls_default, "meta": self.meta}, f, indent=2)

    @classmethod
    def load(cls, path=GENERIC_PATH):
        with open(path) as f:
            d = json.load(f)
        return cls(d["weights"], d["raw_min"], d["raw_max"], d["ls_min"],
                   d["ls_max"], d["ls_default"], d.get("meta"))


def _load_pool_with_signal(pool, thickness_mm):
    """Slow (raw_median, truth_to_surface_face, signal_median) samples of a pool."""
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
        rob = list(csv.reader(open(os.path.join(run, "robot_log.csv"))))[1:]
        samples = []
        for r, rr in zip(rows, rob):
            zc = [float(r[zi + i]) for i in FT_CENTRAL if float(r[zi + i]) > 0]
            sc = [float(r[si + i]) for i in FT_CENTRAL
                  if r[si + i] and float(r[si + i]) > 0]
            if len(zc) == 4 and sc:
                samples.append((statistics.median(zc),
                                float(rr[3]) - TABLE_Z_MM - thickness_mm,
                                statistics.median(sc), float(r[0])))
        for i in range(1, len(samples) - 1):
            dt = samples[i + 1][3] - samples[i - 1][3]
            if dt > 0 and abs((samples[i + 1][1] - samples[i - 1][1]) / dt) \
                    <= FIT_MAX_SPEED_MM_S:
                out.append(samples[i][:3])
    return out


def fit_generic():
    """Fit the surface-independent model from every pool in surfaces.json."""
    surf = json.load(open(os.path.join(HERE, "surfaces.json")))["surfaces"]
    X, y, used = [], [], []
    for pool, props in surf.items():
        s = _load_pool_with_signal(pool, props["thickness_mm"])
        if not s:
            print(f"  {pool}: no signal-bearing runs, skipped")
            continue
        used.append(pool)
        for r, t, sig in s:
            X.append((r, math.log(max(sig, 1.0)))); y.append(t)
        print(f"  {pool}: {len(s)} samples")
    raws = [r for r, _ in X]; lss = [ls for _, ls in X]
    A = np.array([GenericCalibration.feats(r, ls) for r, ls in X])
    w, *_ = np.linalg.lstsq(A, np.array(y), rcond=None)
    resid = A @ w - np.array(y)
    gen = GenericCalibration(w, min(raws), max(raws), min(lss), max(lss),
                             statistics.median(lss), meta={
        "n": len(y), "pools": used,
        "rmse_mm": float(np.sqrt(np.mean(resid ** 2))),
        "loso_note": "expect ~4-5 mm RMSE typical on unseen materials",
    })
    print(f"\ngeneric fit: {len(used)} materials, n={len(y)}, "
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


def _load_run(run_dir):
    """Frame-locked (raw_central_median_mm, truth_mm, time_s) rows of one run."""
    tof = list(csv.reader(open(os.path.join(run_dir, "tof_log.csv"))))[1:]
    rob = list(csv.reader(open(os.path.join(run_dir, "robot_log.csv"))))[1:]
    out = []
    for trow, rrow in zip(tof, rob):
        vals = [float(trow[1 + i]) for i in FT_CENTRAL if float(trow[1 + i]) > 0]
        if not vals:
            continue
        out.append((statistics.median(vals),                 # raw median (no offset)
                    float(rrow[3]) - TABLE_Z_MM,             # truth: robot z above table
                    float(trow[0])))
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
