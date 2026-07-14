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

    def apply(self, raw_mm: float) -> float:
        """Corrected distance. Outside the fitted span, the edge correction is
        carried along at slope 1 instead of extrapolating the polynomial."""
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


def load_for_surface(surface):
    """Load calibration_<surface>.json; fall back to the active calibration.json
    (with a loud warning) if that surface has never been fitted."""
    path = os.path.join(HERE, f"calibration_{surface}.json")
    if os.path.exists(path):
        return Calibration.load(path)
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
