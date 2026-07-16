# ToF + Silicone Interface Simulator

This phase keeps the VL53L8CX-style 8x8 Time-of-Flight sensor backend and adds a first silicone mold exploration layer. The simulator can still run the original raw RTX Lidar scenes, but it can now place a static silicone shape in front of the sensor and post-process the 8x8 frame with an approximate optical response model.

The silicone layer is intentionally approximate: it estimates surface incidence angle, Snell-style refraction through silicone, absorption/scattering loss, and a measured ToF shift. Full optical path tracing, multipath, and Newton/FEM deformation are still separate later phases.

The same entrypoint also supports recorded shape experiments. `shape-replay` loads an STL without modifying its vertices, moves a simple kinematic sensor carriage through the measured robot Z trajectory, and compares raw RTX frames with the corresponding real VL53L5CX frames.

## Current Behavior

The main entrypoint is `sim/scripts/run_vl53l8cx_isaac_tof.py`.

By default it:

* opens Isaac Sim in GUI mode;
* creates a gray sandbox with sunlight;
* shows a visible sensor marker and a 39.89 mm x 40.15 mm x 39.91 mm cube target;
* enables RTX Lidar debug rays;
* prints clean accepted 8x8 distance frames;
* skips startup-empty RTX frames;
* writes the legacy matrix CSV to `sim/output/vl53l8cx_isaac_tof.csv`;
* writes the flat live CSV to `sim/output/live_readings.csv`.

The flat CSV starts with:

```text
timestamp,frame_index,sim_tick,valid_zones,zone_00,...,zone_63
```

The same rows reserve `intensity_00...intensity_63` and `material_00...material_63` after the distance zones. Those fields are blank when Isaac does not provide auxiliary return data.

When a silicone shape is enabled, `zone_00...zone_63` hold the final `measured_tof` matrix. Additional debug columns are appended after the legacy columns:

```text
shape_id
raw_distance_00 ... raw_distance_63
refracted_distance_00 ... refracted_distance_63
optical_loss_00 ... optical_loss_63
surface_angle_00 ... surface_angle_63
ray_deviation_00 ... ray_deviation_63
```

## Commands

Run a GUI validation scene from Isaac Sim's Python directory:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --frames 120 --scene white-full
```

Run headless data generation:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --frames 300 --scene moving --quiet_arrays --record_test_results
```

Launch the live visualizer from the simulator:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --frames 10000 --scene white-full --launch_visualizer --visualizer_python py --keep_visualizer_open
```

Launch the tabletop sensor/cube scene:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --frames 10000 --scene table-cube --launch_visualizer --visualizer_python py --keep_visualizer_open
```

Run a static silicone mold experiment with approximate light paths:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --frames 120 --scene white-full --silicone_shape convex --show_light_paths
```

Load a reusable silicone profile:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --frames 120 --scene white-full --silicone_profile "C:\PathTo\tactile_tof\sim\config\silicone_shapes\convex_soft.json"
```

Run the same capture across all built-in silicone shapes:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --frames 120 --scene white-full --target_distance_m 1.0 --compare_silicone_shapes --quiet_arrays --no_debug_draw
```

## Recorded Cup Experiment

The cup preset is `sim/config/shape_experiments/cup.json`. It references the separated `cup.stl`, both real robot/ToF recordings, a 10 Hz 8x8 replay profile matching the recorded CSV cadence, the timestamp lag, the calibrated TCP-to-sensor offset, and one rigid cup pose shared by both directions.

Shape replay defaults to `--distance-calibration-mode strict`. The
`cup_spoon_ascending_v1` support preflight currently fails 64-zone coverage, so
no strict shared artifact is published. Until a separately versioned flat-target
regime is collected, use `off` to produce honest raw and projected outputs:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --scene shape-replay --experiment-profile "C:\PathTo\tactile_tof\sim\config\shape_experiments\cup.json" --experiment-direction ascending --distance-calibration-mode off
```

Run both directions headlessly. `both` starts a fresh Isaac process for each direction:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --quiet_arrays --no_debug_draw --scene shape-replay --experiment-profile "C:\PathTo\tactile_tof\sim\config\shape_experiments\cup.json" --experiment-direction both --distance-calibration-mode off
```

Render only the initial descending setup for photographic comparison, without replacing any CSV outputs:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --quiet_arrays --no_debug_draw --scene shape-replay --experiment-profile "C:\PathTo\tactile_tof\sim\config\shape_experiments\cup.json" --experiment-direction descending --experiment-setup-image-only --distance-calibration-mode off
```

This writes `setup_beginning.png` in the direction's output directory. The comparison-camera position, target, and image resolution can be overridden with the corresponding `--experiment-setup-camera-*` and `--experiment-setup-image-resolution` options.

The replay automatically captures the 205 ToF frames that overlap the robot log after applying the configured 65 ms lag. `--frames` is intentionally replaced by that reference count. The moving parent rig represents the UR tool; the cup and table never move.

For a quick runtime check, add `--experiment-max-frames 5`; leaving it at the default `0` always runs the complete overlap.

Re-run the dependency-light rigid calibration without starting Isaac Sim:

```powershell
python sim/scripts/run_vl53l8cx_isaac_tof.py --calibrate-experiment-pose --experiment-profile sim/config/shape_experiments/cup.json
```

Calibration writes `sim/output/shape_experiments/cup/calibration.json`. It searches only cup XY translation, yaw, zone-grid orientation, and the fixed TCP-to-sensor offset. It does not alter the STL or synthesize sensor dropout. Copy calibrated values into a profile only after reviewing the reported silhouette score.

Each direction uses output schema `shape-replay-v2` and writes under
`sim/output/shape_experiments/cup/<direction>/`:

```text
sim_matrix.csv             legacy v1 raw RTX time_stamp,data matrices
sim_projected_matrix.csv   projected axial matrices
sim_comparison_matrix.csv  projection-plus-residual matrices (strict/diagnostic only)
sim_flat.csv               explicit validity, unrounded rtx_range_m, modes, and selected-return metadata
comparison.csv             complete-scene raw/projected/comparison frame metrics
summary.json               nested metrics; legacy top-level fields explicitly alias raw
comparison_graph*.png      real/simulation trends labelled by distance mode
step_heatmaps*.png         real/simulation plateau averages labelled by distance mode
```

Create a synchronized 1920x1080 MP4 with the real sensor and simulated 8x8
frames playing side by side, plus a per-zone error view:

```powershell
py "C:\PathTo\tactile_tof\sim\scripts\record_shape_comparison.py" --direction descending
```

The recorder aligns frames by their exact reference timestamps and writes
`real_vs_sim.mp4` beside `sim_flat.csv`. The default is 10 FPS, matching the
recorded dataset. Pink error zones mean the simulation returned a distance when
the real sensor did not; blue zones mean the opposite. To rerun Isaac before
recording the video, use one command:

```powershell
py "C:\PathTo\tactile_tof\sim\scripts\record_shape_comparison.py" --direction descending --sim-distance-mode projected --rerun-simulation --distance-calibration-mode off --isaac-python "C:\isaacsim\python.bat"
```

`--sim-distance-mode raw|projected|comparison` selects the MP4/visualizer data.
Requesting `comparison` from an `off` capture fails instead of silently using a
different mode.

## Shared Ascending Distance Calibration

The explicit training manifest is
`sim/config/shape_experiments/cup_spoon_ascending_v1.json`. It can reference
only ascending inputs; the fitter rejects and never opens descending paths.
Fresh immutable raw/projected captures are created with:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --quiet_arrays --no_debug_draw --scene shape-replay --experiment-profile "C:\PathTo\tactile_tof\sim\config\shape_experiments\cup.json" --experiment-direction ascending --experiment-output-dir "C:\PathTo\tactile_tof\sim\output\shape_calibration\cup_spoon_ascending_v1" --distance-calibration-mode off --experiment-immutable-output
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --quiet_arrays --no_debug_draw --scene shape-replay --experiment-profile "C:\PathTo\tactile_tof\sim\config\shape_experiments\spoon.json" --experiment-direction ascending --experiment-output-dir "C:\PathTo\tactile_tof\sim\output\shape_calibration\cup_spoon_ascending_v1" --distance-calibration-mode off --experiment-immutable-output
```

Run the mandatory support-only preflight:

```powershell
python sim/scripts/fit_shape_distance_calibration.py --training-manifest sim/config/shape_experiments/cup_spoon_ascending_v1.json --mode strict --preflight-only
```

The current capture fails zones `19, 20, 26-29, 33-37, 43, 44`, so the workflow
terminates at `sim/output/shape_calibration/cup_spoon_ascending_v1/coverage.json`
and does not create a calibration artifact. A flat-target capture must start a
new training-regime ID; it must never be pooled into this manifest.

## Recorded Spoon Experiment

The spoon preset is `sim/config/shape_experiments/spoon.json`. It preserves all
21,256 source triangles and applies millimetres-to-metres only through the same
USD transform used by the cup. The resulting dimensions remain exactly
165.0935 x 34.9250 x 9.5250 mm. One pose and one 90.132 mm TCP-to-sensor offset
are fitted jointly from the ascending and descending recordings.

Run both spoon directions in fresh Isaac processes:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --quiet_arrays --no_debug_draw --scene shape-replay --experiment-profile "C:\PathTo\tactile_tof\sim\config\shape_experiments\spoon.json" --experiment-direction both --distance-calibration-mode off
```

Create the synchronized real-versus-simulation videos:

```powershell
py "C:\PathTo\tactile_tof\sim\scripts\record_shape_comparison.py" --experiment-profile "C:\PathTo\tactile_tof\sim\config\shape_experiments\spoon.json" --direction ascending
py "C:\PathTo\tactile_tof\sim\scripts\record_shape_comparison.py" --experiment-profile "C:\PathTo\tactile_tof\sim\config\shape_experiments\spoon.json" --direction descending
```

The timestamp overlap contains 204 ascending frames and 205 descending frames;
the recorder preserves these native counts.

The checked-in cup mesh remains in source units. Isaac applies `0.001` on a parent USD transform, preserving its 72,382 triangles and 124.8843 x 102.1417 x 75 mm bounds. To add another shape, copy the cup profile and change the STL, source origin, reference CSVs, material, and calibrated rigid pose.

Comparison mode writes one flat CSV per shape:

```text
sim/output/shape_tests/flat.csv
sim/output/shape_tests/convex.csv
sim/output/shape_tests/concave.csv
sim/output/shape_tests/half_dome.csv
sim/output/shape_tests/cube.csv
sim/output/shape_tests/fingertip.csv
sim/output/shape_tests/summary.csv
```

`summary.csv` includes:

```text
shape_id
mean_distance
mean_intensity
valid_zones
spatial_variance
edge_distortion
center_distortion
estimated_optical_loss
```

Or run the visualizer separately against the live CSV:

```powershell
py "C:\PathTo\tactile_tof\sim\scripts\visualizer.py" --source csv-tail --input_csv "D:\Machines virtueles\tactile_tof\sim\output\live_readings.csv"
```

## Scenes

`--target_distance_m` is the distance from the sensor to the target's front surface, not the target center. This matches the ToF distance reported by the sensor and avoids placing thick targets inside the near-range boundary.

* `cube`: default 39.89 mm x 40.15 mm x 39.91 mm calibration cube.
* `table-cube`: tabletop scene with a red ToF housing facing a blue cube. If no target distance is supplied, the cube front face is 50 mm from the sensor face.
* `white`: flat white panel at the requested target distance; useful for checking partial field-of-view hits.
* `white-full`: larger white panel sized from the configured FOV and target distance; useful for confirming all 64 zones map correctly.
* `oblique`: tilted panel that should create a row/column distance gradient.
* `moving`: white panel moving sinusoidally along the sensor axis.
* `materials`: vertical material strips for checking intensity and material IDs.
* `no-target`: no target object; accepted frames should be all zero.
* `shape-replay`: fixed STL/table scene with a downward-facing sensor rig replaying recorded robot Z samples.

Common options:

```powershell
--target_distance_m 1.0  # front-surface distance, meters; table-cube defaults to 0.05
--target_center_z 0.35
--sensor_xyz 0,0,0.35
--sensor_quat_wxyz 1,0,0,0
--max_sim_ticks 5000
--no_debug_draw
--print_payload_debug
```

Silicone mold options:

```powershell
--silicone_shape flat|convex|concave|half_dome|cube|fingertip
--silicone_width_m 0.04
--silicone_height_m 0.04
--silicone_thickness_m 0.006
--silicone_radius_m 0.025
--silicone_curvature 1.0
--silicone_offset_from_sensor_m 0.02
--silicone_refractive_index 1.41
--silicone_transparency 0.86
--silicone_scattering_strength 0.2
--silicone_absorption_strength 0.05
--silicone_surface_roughness 0.15
--reflective_inner_coating
--show_light_paths
```

`--silicone_profile` accepts the same fields as JSON. CLI values override profile values when both are provided.

Comparison options:

```powershell
--compare_silicone_shapes
--shape_tests_dir sim/output/shape_tests
--shape_summary_csv sim/output/shape_tests/summary.csv
```

During comparison, the selected scene, target distance, sensor pose, frame count, material settings, and noise/material parameters are reused for every shape. The `flat` result is used as the baseline for edge and center distortion metrics.

## Tests

The dependency-free helper tests do not require Isaac Sim:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Recommended Isaac smoke checks when Isaac Python is available:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --frames 5 --scene white-full
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --frames 5 --scene no-target
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --frames 10 --scene moving
```
