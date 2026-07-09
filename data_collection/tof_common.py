#!/usr/bin/env python3
"""
Shared helpers for the VL53L5CX ToF analysis scripts.

Data conventions
================
* Sensor readings CSV: ``timestamp,zone_00..zone_63`` — 64 zone distances in mm,
  ``0`` = no valid measurement. The VL53L5CX is an 8x8 zone array.
* UR5 / ground-truth CSV: ``,z`` — end-effector height in METERS, logged densely.

Noise filter
============
At far range the sensor returns spurious large values from background reflections
instead of the surface. Measured across the flat-mount and A3 datasets, true
surface returns stay below ~600 mm while background noise lands above ~800 mm,
with an empty gap between. Each frame is cleaned in two stages:
  1. Far-range gate — drop zones reading 0 or above ``MAX_RANGE_MM``.
  2. Per-frame MAD outlier rejection — drop zones more than ``MAD_K`` robust
     deviations from the frame median (with an absolute floor).
"""

import csv
import os
import numpy as np
from datetime import datetime

N_ZONES = 64
GRID = 8

# Noise-filter parameters (shared across all ToF analyses).
MAX_RANGE_MM = 800.0
MAD_K = 3.5
MAD_FLOOR_MM = 25.0
MIN_VALID_ZONES = 8

# Step (dwell-plateau) detection on the UR5 staircase.
DWELL_WIN_S = 0.15
DWELL_EPS_MM = 4.0
DWELL_MIN_DUR_S = 0.4

FILTER_LABEL = (f"≤{MAX_RANGE_MM:.0f} mm gate + {MAD_K:.1f}·MAD per-frame")
GATE_ONLY_LABEL = (f"≤{MAX_RANGE_MM:.0f} mm gate only (no MAD — keeps object zones)")


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def filter_frame(vals: np.ndarray, apply_mad: bool = True) -> np.ndarray:
    """Return a 64-vector of cleaned zone distances (NaN = rejected/no-reading).

    apply_mad=False keeps only the far-range gate (stage 1). Use it for object /
    shape data, where a small target reads at a biased distance that the MAD
    outlier rejection would wrongly discard — for flat-surface ranging keep it on.
    """
    g = vals.astype(float).copy()
    g[(g <= 0) | (g > MAX_RANGE_MM)] = np.nan           # stage 1: far-range gate
    if apply_mad:
        finite = g[np.isfinite(g)]
        if finite.size >= MIN_VALID_ZONES:              # stage 2: MAD rejection
            med = np.median(finite)
            mad = np.median(np.abs(finite - med))
            spread = max(MAD_K * 1.4826 * mad, MAD_FLOOR_MM)
            g[np.abs(g - med) > spread] = np.nan        # NaN-safe (NaN compares False)
    return g


def load_readings(path: str, apply_mad: bool = True):
    """Return (secs, grid[N,64] cleaned mm with NaN, t0)."""
    ts, grids = [], []
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)  # header
        for row in rd:
            if not row or len(row) < 1 + N_ZONES:
                continue
            ts.append(parse_ts(row[0]))
            grids.append(filter_frame(
                np.array([float(x) for x in row[1:1 + N_ZONES]]), apply_mad))
    t0 = ts[0]
    secs = np.array([(t - t0).total_seconds() for t in ts])
    return secs, np.array(grids), t0


def load_ground_truth(path: str, t0: datetime):
    """Return (secs relative to t0, z_mm) sorted by time."""
    ts, z = [], []
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)  # header: ",z"
        for row in rd:
            if len(row) < 2 or row[1] == "":
                continue
            ts.append(parse_ts(row[0]))
            z.append(float(row[1]) * 1000.0)  # m -> mm
    secs = np.array([(t - t0).total_seconds() for t in ts])
    order = np.argsort(secs)
    return secs[order], np.array(z)[order]


def detect_steps(g_t: np.ndarray, g_z: np.ndarray):
    """Find dwell plateaus in the UR5 staircase.

    A sample is "dwelling" when UR5 z stays within DWELL_EPS_MM over a
    +/- DWELL_WIN_S window (robust to the sub-mm jitter that defeats a naive
    velocity threshold). Returns [(t_start, t_end, level_mm), ...] in time order.
    NOTE: commanded steps smaller than DWELL_EPS_MM would merge — fine for the
    >=10 mm increments used here.
    """
    n = len(g_t)
    dwelling = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = np.searchsorted(g_t, g_t[i] - DWELL_WIN_S)
        hi = np.searchsorted(g_t, g_t[i] + DWELL_WIN_S)
        seg = g_z[lo:hi]
        dwelling[i] = (seg.max() - seg.min()) < DWELL_EPS_MM

    steps, i = [], 0
    while i < n:
        if dwelling[i]:
            j = i
            while j < n and dwelling[j]:
                j += 1
            if g_t[j - 1] - g_t[i] >= DWELL_MIN_DUR_S:
                steps.append((g_t[i], g_t[j - 1], float(np.median(g_z[i:j]))))
            i = j
        else:
            i += 1
    return steps


def dwell_mask(secs: np.ndarray, steps) -> np.ndarray:
    """Boolean mask over `secs` that is True only during a detected dwell step."""
    m = np.zeros(len(secs), dtype=bool)
    for t0, t1, _ in steps:
        m |= (secs >= t0) & (secs <= t1)
    return m
