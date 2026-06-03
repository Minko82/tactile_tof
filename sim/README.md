# ToF + Silicone Interface Simulator

This phase keeps the VL53L8CX-style 8x8 Time-of-Flight sensor backend and adds a first silicone mold exploration layer. The simulator can still run the original raw RTX Lidar scenes, but it can now place a static silicone shape in front of the sensor and post-process the 8x8 frame with an approximate optical response model.

The silicone layer is intentionally approximate: it estimates surface incidence angle, Snell-style refraction through silicone, absorption/scattering loss, and a measured ToF shift. Full optical path tracing, multipath, and Newton/FEM deformation are still separate later phases.

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
