from __future__ import annotations

import csv
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "sim" / "scripts" / "run_vl53l8cx_isaac_tof.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_vl53l8cx_isaac_tof", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tof = load_module()


class VL53L8CXCoreTests(unittest.TestCase):
    def test_default_config_and_json_profile(self):
        default = tof.VL53L8CXConfig()
        config = tof.VL53L8CXConfig.from_json(REPO_ROOT / "sim" / "config" / "vl53l8cx_8x8.json")
        self.assertEqual(default.rows, config.rows)
        self.assertEqual(default.cols, config.cols)
        self.assertEqual(default.fov_h_deg, config.fov_h_deg)
        self.assertEqual(default.fov_v_deg, config.fov_v_deg)
        self.assertEqual(default.min_mm, config.min_mm)
        self.assertEqual(default.max_mm, config.max_mm)
        self.assertEqual(default.invalid_mm, config.invalid_mm)
        self.assertEqual(config.rows, 8)
        self.assertEqual(config.cols, 8)
        self.assertEqual(config.zones, 64)
        self.assertEqual(config.min_range_m, 0.02)
        self.assertEqual(config.max_range_m, 4.0)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            tof.VL53L8CXConfig(rows=0)
        with self.assertRaises(ValueError):
            tof.VL53L8CXConfig(min_mm=100, max_mm=50)
        with self.assertRaises(ValueError):
            tof.VL53L8CXConfig(frame_rate_hz=0)

    def test_zone_row_column_mapping(self):
        config = tof.VL53L8CXConfig()
        self.assertEqual(tof.zone_index_to_row_col(0, config), (0, 0))
        self.assertEqual(tof.zone_index_to_row_col(7, config), (0, 7))
        self.assertEqual(tof.zone_index_to_row_col(8, config), (1, 0))
        self.assertEqual(tof.zone_index_to_row_col(63, config), (7, 7))
        self.assertEqual(tof.row_col_to_zone_index(7, 7, config), 63)

    def test_emitter_id_normalization_accepts_zero_or_one_based_ids(self):
        config = tof.VL53L8CXConfig()
        self.assertEqual(tof.emitter_ids_to_zone_indices([0, 1, 63], config), [0, 1, 63])
        self.assertEqual(tof.emitter_ids_to_zone_indices([1, 2, 64], config), [0, 1, 63])
        self.assertEqual(tof.emitter_ids_to_zone_indices([999], config), [None])

    def test_distance_conversion_clips_and_handles_no_return(self):
        config = tof.VL53L8CXConfig()
        self.assertEqual(tof.distance_m_to_mm(None, config), 0)
        self.assertEqual(tof.distance_m_to_mm(float("nan"), config), 0)
        self.assertEqual(tof.distance_m_to_mm(-1, config), 0)
        self.assertEqual(tof.distance_m_to_mm(0.001, config), 20)
        self.assertEqual(tof.distance_m_to_mm(1.234, config), 1234)
        self.assertEqual(tof.distance_m_to_mm(5.0, config), 4000)

    def test_sensor_attributes_match_vl53l8cx_profile(self):
        config = tof.VL53L8CXConfig()
        attrs = tof._make_sensor_attributes(config)
        self.assertEqual(attrs["omni:sensor:Core:scanRateBaseHz"], 15.0)
        self.assertEqual(attrs["OmniSensorGenericLidarCoreEmitterStateAPI:s001:beamCountHoriz"], 8)
        self.assertEqual(attrs["OmniSensorGenericLidarCoreEmitterStateAPI:s001:beamCountVert"], 8)
        self.assertEqual(attrs["OmniSensorGenericLidarCoreEmitterStateAPI:s001:azimuthStartDeg"], -22.5)
        self.assertEqual(attrs["OmniSensorGenericLidarCoreEmitterStateAPI:s001:azimuthEndDeg"], 22.5)
        self.assertEqual(attrs["OmniSensorGenericLidarCoreEmitterStateAPI:s001:minRangeM"], 0.02)
        self.assertEqual(attrs["OmniSensorGenericLidarCoreEmitterStateAPI:s001:maxRangeM"], 4.0)
        self.assertEqual(attrs["omni:sensor:Core:minDistBetweenEchosM"], 0.02)

    def test_target_distance_is_front_surface_distance(self):
        self.assertAlmostEqual(tof._center_x_from_front_distance(0.03, tof.TARGET_CUBE_SIZE_M[0]), 0.049945)
        parser = tof.build_arg_parser()
        self.assertEqual(tof._scene_target_distance_m(parser.parse_args(["--scene", "cube"])), 1.0)
        self.assertEqual(tof._scene_target_distance_m(parser.parse_args(["--scene", "table-cube"])), 0.05)
        self.assertEqual(tof._scene_target_distance_m(parser.parse_args(["--scene", "table-cube", "--target_distance", "0.03"])), 0.03)

    def test_build_distance_matrix_from_returns(self):
        config = tof.VL53L8CXConfig()
        matrix, intensities, material_ids = tof.build_distance_matrix_from_returns(
            [0.10, 0.20, 0.15, 5.0, math.nan],
            config,
            emitter_ids=[0, 1, 0, 63, 2],
            intensities=[0.5, 0.6, 0.9, 0.1, 1.0],
            material_ids=[10, 11, 12, 13, 14],
        )
        self.assertEqual(matrix[0][0], 100)
        self.assertEqual(matrix[0][1], 200)
        self.assertEqual(matrix[0][2], 0)
        self.assertEqual(matrix[7][7], 4000)
        self.assertEqual(intensities[0][0], 0.5)
        self.assertEqual(material_ids[0][0], 10)

    def test_csv_matrix_format_and_round_trip(self):
        matrix = [[row * 8 + col for col in range(8)] for row in range(8)]
        text = tof.format_matrix_for_csv(matrix)
        self.assertTrue(text.startswith("[[0 1 2 3 4 5 6 7], [8 9"))
        self.assertEqual(tof.parse_matrix_text(text), matrix)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["time_stamp", "data"])
        writer.writerow(tof.VL53L8CXFrame("12:00:00.000000", matrix).csv_row())
        output.seek(0)
        rows = list(csv.DictReader(output))
        self.assertEqual(tof.parse_matrix_text(rows[0]["data"]), matrix)

    def test_existing_press_example_matrix_is_compatible(self):
        with (REPO_ROOT / "examples" / "press_example.csv").open("r", encoding="utf-8", newline="") as handle:
            first_row = next(csv.DictReader(handle))
        parsed = tof.parse_matrix_text(first_row["data"])
        self.assertEqual(len(parsed), 8)
        self.assertTrue(all(len(row) == 8 for row in parsed))

    def test_write_frames_csv_creates_expected_header(self):
        frame = tof.VL53L8CXFrame("12:00:00.000000", [[1 for _ in range(8)] for _ in range(8)])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frames.csv"
            tof.write_frames_csv([frame], path)
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["time_stamp", "data"])
        self.assertEqual(rows[1][0], "12:00:00.000000")
        self.assertEqual(tof.parse_matrix_text(rows[1][1]), frame.distances_mm)

    def test_flat_csv_writer_creates_zone_columns(self):
        config = tof.VL53L8CXConfig()
        matrix = [[row * 8 + col for col in range(8)] for row in range(8)]
        frame = tof.VL53L8CXFrame("12:00:00.000000", matrix)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "live.csv"
            with tof.VL53L8CXFlatCsvWriter(path, config) as writer:
                writer.write_frame(2, 9, frame)
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames
                rows = list(reader)
        self.assertIsNotNone(header)
        self.assertEqual(header[4], "zone_00")
        self.assertEqual(header[67], "zone_63")
        self.assertEqual(header[68], "intensity_00")
        self.assertEqual(header[131], "intensity_63")
        self.assertEqual(header[132], "material_00")
        self.assertEqual(header[195], "material_63")
        self.assertIn("zone_00", rows[0])
        self.assertIn("zone_63", rows[0])
        self.assertEqual(rows[0]["frame_index"], "2")
        self.assertEqual(rows[0]["sim_tick"], "9")
        self.assertEqual(rows[0]["valid_zones"], "63")
        self.assertEqual(rows[0]["zone_00"], "0")
        self.assertEqual(rows[0]["zone_63"], "63")
        self.assertEqual(rows[0]["intensity_00"], "")
        self.assertEqual(rows[0]["material_63"], "")

    def test_flat_csv_writer_writes_auxiliary_columns(self):
        config = tof.VL53L8CXConfig()
        matrix = [[100 for _ in range(8)] for _ in range(8)]
        intensities = [[None for _ in range(8)] for _ in range(8)]
        material_ids = [[None for _ in range(8)] for _ in range(8)]
        intensities[0][0] = 0.5
        intensities[7][7] = 1.25
        material_ids[0][0] = 10
        material_ids[7][7] = 99
        frame = tof.VL53L8CXFrame("12:00:00.000000", matrix, intensities=intensities, material_ids=material_ids)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "live.csv"
            with tof.VL53L8CXFlatCsvWriter(path, config) as writer:
                writer.write_frame(0, 4, frame)
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["valid_zones"], "64")
        self.assertEqual(rows[0]["intensity_00"], "0.5")
        self.assertEqual(rows[0]["intensity_63"], "1.25")
        self.assertEqual(rows[0]["material_00"], "10")
        self.assertEqual(rows[0]["material_63"], "99")

    def test_empty_startup_frame_skip_and_no_target_zero_frame(self):
        config = tof.VL53L8CXConfig()
        self.assertIsNone(tof.frame_from_sensor_frame({}, config))
        self.assertIsNone(tof.frame_from_sensor_frame({"IsaacCreateRTXLidarScanBuffer": {"distance": []}}, config))

        frame = tof.frame_from_sensor_frame({}, config, allow_empty_no_return=True)
        self.assertIsNotNone(frame)
        self.assertEqual(tof.flatten_matrix(frame.distances_mm), [0 for _ in range(config.zones)])
        self.assertIsNone(frame.intensities)
        self.assertIsNone(frame.material_ids)

    def test_parser_defaults_debug_draw_and_array_printing_on(self):
        parser = tof.build_arg_parser()
        args = parser.parse_args([])
        self.assertTrue(args.debug_draw)
        self.assertTrue(args.print_arrays)
        self.assertFalse(args.print_payload_debug)
        self.assertFalse(args.headless)
        self.assertEqual(args.max_sim_ticks, 0)
        self.assertIsNone(args.target_distance_m)

        quiet = parser.parse_args(["--no_debug_draw", "--quiet_arrays", "--scene", "white-full", "--max_sim_ticks", "123"])
        self.assertFalse(quiet.debug_draw)
        self.assertFalse(quiet.print_arrays)
        self.assertEqual(quiet.scene, "white-full")
        self.assertEqual(quiet.max_sim_ticks, 123)

        alias = parser.parse_args(["--target_distance", "0.03"])
        self.assertEqual(alias.target_distance_m, 0.03)


if __name__ == "__main__":
    unittest.main()
