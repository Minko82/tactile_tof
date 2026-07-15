# Tactile force capture & calibration protocol

Turns the 8×8 ToF + deformable membrane into **calibrated normal (pressure) and
shear force**, reusing this folder's proven front-end: **per-zone degree-4
polynomial calibration + leave-one-round-out (LORO) validation**
(`tof_sensor.fit_zone_calibration`). Nothing here is self-fit — every model is tested on
held-out rounds.

The chain is three calibrations, in order. Each writes the **same logging
format** so `plot_force.py` consumes them directly.

```
raw ToF  ─①per-zone distance cal→  accurate mm  ─②baseline→  indentation field
         indentation  ─③normal fit→  F_normal (N)     (from depth / volume)
                       ─③shear  fit→  F_shear (N,dir)  (from centroid shift)
```

---

## Rig

- **Sensor + membrane.** Record membrane material, thickness, durometer (Shore
  A), and rest gap (sensor→membrane underside, target ~15–30 mm so a press stays
  in valid range as long as possible).
- **Distance-cal target.** A flat matte plate parallel to the sensor on a
  micrometer/shim stage — OR the UR5 (as the A3 proximity captures already do)
  driven to close range.
- **Normal-force tool.** A flat indenter of **known contact area** + known loads
  (calibrated weights 0–500 g, or an inline load cell). Log **force**, not mass.
- **Shear-force tool.** A probe coupled to the membrane, tangential load via a
  pulley+weights or a lateral force gauge, applied at a **held normal preload**.

---

## ① Distance calibration  (per-zone raw → mm)

Makes indentation *depth* accurate. Same engine as `tof_sensor.fit_zone_calibration`.

- Present the flat plate filling the FoV at known gaps
  `g ∈ {5, 7, 10, 15, 20, 30, 50} mm` (bracket the touch band closely; extend up
  if you also want long proximity range).
- Hold each gap ≥ 3 s (≥ 200 samples/zone). **Repeat 5 rounds** (LORO).
- Below the VL53L5CX ~20 mm rated minimum, expect `status≠5`/no-target — **record
  it anyway**; the fit models only the valid samples and the detector treats
  below-range blanking as contact.
- Output: per-zone degree-4 coeffs; drop zones with residual σ > 6 mm.

## ② Baseline  (membrane at rest)

- No contact, ≥ 5 s. Per-zone baseline = filtered resting distance (median).
- **Re-capture** whenever the rig changes and periodically (elastomer creep +
  temperature drift move the baseline by tenths of a mm).

## ③a Normal-force calibration  (depth → N)

- Flat indenter at the membrane **center**.
- Load steps `F_n ∈ {0, 0.5, 1, 2, 3, 5} N` — **ascending then descending**
  (captures hysteresis).
- Hold each step ~2 s. Repeat at **≥ 3 positions** (center + 2 offsets, to check
  spatial uniformity of the map) and **≥ 3 rounds** (LORO).
- Feature fit: indentation **volume** `V = Σ depth · zone_area` over the contact
  (primary; ∝ displaced material) and **peak depth**. Fit `F_n = f(V, peak)`;
  expect near-linear, allow degree 2. Report LORO RMSE in **N** and in **% FS**.

## ③b Shear-force calibration  (centroid shift → N)

- Hold a fixed **normal preload** (e.g. 2 N). Apply tangential force
  `F_s ∈ {0, 0.25, 0.5, 1 N}` in **≥ 4 directions** (±x, ±y), ascending/descending.
- Hold ~2 s each; repeat rounds. Redo at **2–3 preloads** (shear stiffness
  depends on normal load / contact area).
- Feature fit: deformation-**centroid shift** `Δc` relative to the no-shear
  centroid at that preload → `F_s = k_s·|Δc|`, direction = angle of `Δc`.
- ⚠ **Risk item.** Depth-only ToF senses shear only if the membrane couples
  tangential force into a measurable depth *skew*. If `Δc` vs `F_s` comes back
  flat/noisy, a flat membrane is insufficient — you need an anchored/structured
  surface (ridges, internal features, or bonded edges). Capture this first with a
  quick sweep before committing to the full grid.

---

## Logging format

Every session is a folder tree; each round has two time-synchronized CSVs on the
**same clock** as the existing `tof_logger.py`:

```
<session>/
  distance_cal/round_1..5/   tof_log.csv                 (+ gap_log.csv)
  baseline/                  tof_log.csv
  normal_cal/round_1..3/     tof_log.csv  force_log.csv
  shear_cal/round_1..3/      tof_log.csv  force_log.csv
```

- **`tof_log.csv`** — unchanged: `time_s, z0..z63` (add `status0..63`,
  `sigma0..63` if the firmware exposes them — the fusion can use `range_sigma`).
- **`gap_log.csv`** — `time_s, gap_mm` (the known plate distance).
- **`force_log.csv`** — `time_s, phase, F_n_N, F_s_N, shear_dir_deg, position,
  indenter_area_mm2, note`. One row per label change is enough (values hold until
  the next row); `capture_logger.py` stamps these against the same clock.

`plot_force.py` reads this tree, applies ①, computes indentation, and fits ③a/③b
with LORO (currently on synthetic data until real captures exist).
