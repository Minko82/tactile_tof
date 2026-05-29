# ToF Sensor Simulator

This phase validates the VL53L8CX-style Time-of-Flight sensor by itself. The simulator models an 8x8 ray-based proximity sensor in Isaac Sim and writes distance frames that can be inspected in the console, tailed by the live visualizer, or analyzed from CSV.

The silicone dome, deformable fingertip mesh, scattering, multipath, and touch physics are intentionally excluded from this phase. Those features belong to the next simulator phase after the ToF sensing pipeline is stable.

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

## Tests

The dependency-free helper tests do not require Isaac Sim:

```powershell
python -m unittest tests.test_vl53l8cx_isaac_tof
```

Recommended Isaac smoke checks when Isaac Python is available:

```powershell
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --frames 5 --scene white-full
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --frames 5 --scene no-target
.\python.bat "C:\PathTo\tactile_tof\sim\scripts\run_vl53l8cx_isaac_tof.py" --headless --frames 10 --scene moving
```
