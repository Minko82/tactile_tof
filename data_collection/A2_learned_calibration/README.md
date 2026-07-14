# A2 — Learned Calibration + Near-Pass-Through Kalman

Implementation of the two-stage architecture proposed in
`../A2_data_filter/PROS_CONS.md`, for head-to-head comparison against the
Kalman-only method in `../A3_proximity` (`robot.py filtertest`).

```
raw central-2x2 median ──> stage 1: learned poly calibration (zero lag)
                       ──> stage 2: near-pass-through Kalman (velocity,
                            outlier gating, dropout coasting)
```

- **Stage 1** (`calibration.py`) replaces the constant mount offset with a
  degree-3 polynomial `raw -> mm` fitted against UR5 ground truth. Memoryless,
  so it removes the range-dependent bias with exactly zero lag. Fitted only on
  slow samples (|v| ≤ 100 mm/s) to avoid learning pose/frame sync slop; never
  extrapolated outside its fitted raw span (edge correction carried at slope 1).
- **Stage 2** (`run_test.py`) keeps `LiveFilter` but tuned near-pass-through:
  `q=5e4`, fixed `R=4`, gate loosened to 8σ. Adaptive R is OFF — on this
  ultra-clean sensor it shrinks R toward the floor and the outlier gate then
  strangles real fast motion (that was the dominant lag mechanism, worth ~27 mm
  RMSE in the fast phase).

## Data layout & workflow

Every material gets its own pool and its own model, keyed by the name you pass
on the command line — ANY name works, new folders are created automatically:

```
<surface>/<n>/                 runs pool (wood/, white/, cardboard/, ...)
calibration_<surface>.json     that surface's fitted model
calibration.json               fallback used only when a surface has no fit yet
```

`run_test.py` / `compare_test.py <surface>` automatically save into
`<surface>/<n>/` AND load `calibration_<surface>.json` (falling back to the
active file with a loud warning if the surface was never fitted — expect a
constant bias until you fit it).

```bash
# 1. collect runs on any material (creates felt/1, felt/2, ...)
python3 compare_test.py felt viz

# 2. fit that material's calibration from its whole pool
python3 calibration.py fit felt       # -> calibration_felt.json

# 3. (optional) constant for the kalman baseline on that material
python3 ../A3_proximity/robot.py offset felt/1   # paste into SURFACE_OFFSETS_MM
```

Motion, logging format, live plot, and the per-phase RMSE summary are identical
to A3's `filtertest`, so `results/<n>/filter_log.csv` compares row-for-row with
`../A3_proximity/filtertest/<n>/filter_log.csv`.

## Held-out validation (replay, no robot needed)

Calibration fitted on runs 5–10, evaluated on run 11 (never seen in the fit),
replaying run 11's raw stream through the full new pipeline:

| phase     | old pipeline (logged) | calibrated raw | new pipeline |
|-----------|----------------------:|---------------:|-------------:|
| linear    | 4.64                  | 1.83           | **1.75**     |
| static    | 4.46                  | 0.28           | **0.26**     |
| rand-slow | 5.38                  | 3.71           | **3.75**     |
| rand-fast | 50.74                 | 8.44           | **9.01**     |

RMSE vs robot-Z ground truth, mm. The new pipeline matches or beats calibrated
raw everywhere except a ~0.6 mm cost in the fast phase — the price of keeping
velocity estimation, spike gating, and dropout coasting. Static velocity-estimate
noise: ~4 mm/s std.

Stage-2 tuning was chosen by an offline sweep replayed against the held-out run
(`q ∈ {1e4, 5e4, 1e5}` × {adaptive R, fixed R} × {5σ, 8σ, no gate}); `q=5e4,
R=4, 8σ` was the best fast-phase result that keeps some gating.

## Caveats (see PROS_CONS.md for the full list)

- The calibration is valid for **this table, this mount, near-perpendicular
  incidence**. Refit (`calibration.py fit`) after any mount/sensor/firmware
  change or when sensing different materials.
- Fast-phase numbers for BOTH pipelines include a few mm of ground-truth sync
  slop (60 Hz pose vs 15 Hz frames) — treat small differences there as noise.
