# Kalman Filter: Current Implementation vs. Proposed Architecture

Comparison of the current live-filter setup against the proposed two-stage
architecture, grounded in the UR5 `filtertest` results (see
`A3_proximity/filtertest/<n>/filter_log.csv`; phase RMSE figures below are from
run 8/9, sensor at ~15 Hz over a flat table).

**Measured context that drives every trade-off here:**

- Static sensor scatter is **0.35 mm** — the sensor is extremely clean, so
  temporal smoothing has almost nothing to win.
- Raw bias vs. robot ground truth is **range-dependent** (about −1 mm near the
  table, +7 mm at ~600 mm) — a constant offset cannot zero it everywhere.
- Ground truth itself has a few mm of sync slop during fast motion (60 Hz pose
  stream paired with 15 Hz sensor frames).

---

## 1. Current implementation

`live_filter.LiveFilter` (constant-velocity Kalman, `process_accel_psd` ≈
500–3000, adaptive measurement variance, 5σ outlier gating, maneuver relock,
dropout coasting) fed the central-2×2 median minus a constant mount offset.

### Pros

- **Excellent on steady motion and at rest.** Linear phase: 3.2 mm RMSE with
  no steady-state lag at 60 mm/s. Static phase: sub-mm after bias removal.
- **Velocity estimate for free.** Recovered the commanded −60 mm/s descent
  almost exactly from noisy distances alone; raw readings cannot provide this.
- **Robust to real-stream pathologies.** Dropped/invalid frames coast on the
  prediction; isolated spikes are softly downweighted; sustained jumps trigger
  a maneuver relock. Deploy-ready against sensor glitches.
- **Simple, causal, fast, dependency-free.** One file, pure Python, ~µs per
  update, fully deterministic and unit-tested (30 tests).
- **One interpretable knob.** `process_accel_psd` maps directly to a physical
  assumption ("how hard can the target accelerate").

### Cons

- **Net loss during fast motion.** Random phase: raw 10 mm RMSE vs. filtered
  41 mm — the smoothing lag costs far more than the 0.35 mm of noise it
  removes. The filter made accurate data less accurate whenever the target
  maneuvered.
- **One tuning cannot fit all regimes.** Low `q` = clean static/linear but
  laggy maneuvers; high `q` = tight tracking but noisier velocity and weaker
  outlier gating. Any single value is a compromise.
- **Constant offset can't fix range-dependent bias.** 23.5 mm zeros the
  *average* error only; residual ±4–5 mm systematic error remains at the range
  extremes, and the filter cannot remove bias — it faithfully tracks it.
- **Adaptive R learns whatever it is fed.** With the sensor this clean it
  adapts toward the floor and then treats real motion onsets as outliers until
  the maneuver logic confirms them (the downweighted cluster at the start of
  the fast phase).

---

## 2. Proposed architecture

**Stage 1 — learned, memoryless calibration:** per-zone (or central-median)
polynomial `raw → mm` fitted against robot ground truth
(`tof_sensor.fit_zone_calibration` already implements this), replacing the
constant mount offset.
**Stage 2 — near-pass-through Kalman:** same `LiveFilter`, `process_accel_psd`
high (≈10 000+), kept for velocity estimation, outlier gating, and dropout
coasting rather than for smoothing.
**No temporal neural denoiser** — with 0.35 mm scatter there is nothing for it
to remove, and it would inherit the same smoothing-vs-lag physics plus new
failure modes.

### Pros

- **Kills the range-dependent bias with exactly zero lag.** Calibration is a
  per-sample function of the current reading only — no temporal window, so
  correcting bias costs no latency at any speed.
- **Filter stops fighting real motion.** Near-pass-through tuning means the
  fast phase tracks close to raw (~10 mm) instead of 41 mm, while keeping the
  glitch/dropout machinery.
- **Training data is free.** Every `filtertest` / `record` run produces
  frame-locked raw + ground truth; fitting the poly is one offline script.
- **Each stage stays interpretable and testable.** A polynomial and a Kalman
  filter, not a black box; both are inspectable and unit-testable, and the
  existing per-phase RMSE table measures any change directly.
- **Right tool per error type.** Systematic error → learned static map;
  stochastic/temporal effects → recursive filter. Clean separation of concerns.

### Cons

- **Calibration is only valid for the world it was trained in.** ToF bias
  shifts with target reflectivity, incidence angle, ambient IR, and
  temperature. A fit made against this table can be *worse* than raw+constant
  on dark, glossy, or oblique targets. Biggest single risk.
- **Extrapolation danger.** A degree-4 poly outside its fitted span
  (~12–618 mm) can swing wildly; inputs must be clamped or fall back to
  raw+constant, and forgetting that is a silent bug.
- **Recalibration burden.** Any mount, sensor, or firmware change invalidates
  the fit, and recalibrating requires the robot rig.
- **Ground-truth sync slop caps the fit.** Pose/frame pairing is worth a few
  mm during motion — calibrate from static/slow data only, or the fit learns
  the sync artifact as "bias".
- **Noisy velocity estimate.** High `q` barely smooths, and velocity is
  effectively a derivative — the clean −60 mm/s trace degrades. If both
  responsive position *and* clean velocity are needed, run two filter tunings
  in parallel (cheap) or move to an IMM.
- **Weaker outlier gating.** High `q` widens the 5σ gate, so real glitch
  spikes pass more easily. Currently cheap (near-zero spike rate on the
  table); re-check on less friendly surfaces.
- **Two coupled stages, two suspects.** A stale calibration poisons everything
  downstream — including the filter's adaptive R — and can masquerade as a
  filter bug. Diagnosis and validation cost roughly doubles, and mm-level
  comparisons run into the rig's own measurement floor.

---

## Bottom line

| | Current (CV Kalman + constant offset) | Proposed (calibration + pass-through Kalman) |
|---|---|---|
| Static / linear accuracy | Excellent | Excellent (bias also removed) |
| Fast-motion accuracy | Poor (4× worse than raw) | Near-raw |
| Bias handling | Average-only constant | Zero-lag learned map (surface-specific) |
| Velocity estimate | Clean | Noisy (needs 2nd tuning or IMM if required) |
| Robustness to spikes/dropouts | Strong | Kept, but gate is wider |
| Generalization to new surfaces | Insensitive | Fragile — recalibrate per material |
| Complexity / failure modes | One stage | Two coupled stages |

The proposed architecture is the better fit for the current lab setup (flat
table, known material, robot available for calibration). Its main liability —
calibration fragility across surfaces — becomes the dominant concern the
moment varied or unknown materials enter the picture; at that point consider
per-material calibration, reflectance-aware correction (the VL53L5CX reports
per-zone signal strength), or falling back to raw + constant with the IMM
pipeline from `tof_sensor.py`.
