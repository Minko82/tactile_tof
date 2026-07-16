from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "sim" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import shape_replay  # noqa: E402 - tests import the sibling script module directly


def load_main_module():
    path = SCRIPT_DIR / "run_vl53l8cx_isaac_tof.py"
    spec = importlib.util.spec_from_file_location("shape_replay_main", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


main_module = load_main_module()
PROFILE_PATH = REPO_ROOT / "sim" / "config" / "shape_experiments" / "cup.json"


@dataclass
class FakeFrame:
    distances_mm: list[list[int]]
    intensities: object | None = None
    material_ids: object | None = None


class ShapeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = shape_replay.ShapeExperimentProfile.from_json(PROFILE_PATH)

    def test_cup_profile_and_stl_preserve_source_geometry(self):
        mesh = shape_replay.load_stl(self.profile.stl_path)
        self.assertEqual(mesh.triangle_count, 72_382)
        expected_min = (220.31185913085938, 39.00712966918945, 0.0)
        expected_max = (345.1961669921875, 141.14881896972656, 75.0)
        for actual, expected in zip(mesh.bounds_min, expected_min):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(mesh.bounds_max, expected_max):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(mesh.extents[0] * self.profile.stl_units_to_m, 0.12488430786132812)
        self.assertAlmostEqual(mesh.extents[1] * self.profile.stl_units_to_m, 0.10214168930053711)
        self.assertAlmostEqual(mesh.extents[2] * self.profile.stl_units_to_m, 0.075)
        sensor = main_module.VL53L8CXConfig.from_json(self.profile.sensor_profile)
        self.assertEqual(sensor.frame_rate_hz, 10.0)

    def test_reference_alignment_has_205_frames_per_direction(self):
        ascending = shape_replay.load_replay_samples(self.profile, "ascending")
        descending = shape_replay.load_replay_samples(self.profile, "descending")
        self.assertEqual(len(ascending), 205)
        self.assertEqual(len(descending), 205)
        self.assertLess(ascending[0].tcp_z_m, ascending[-1].tcp_z_m)
        self.assertGreater(descending[0].tcp_z_m, descending[-1].tcp_z_m)
        self.assertAlmostEqual(ascending[0].elapsed_s, 0.0)
        self.assertAlmostEqual(descending[0].elapsed_s, 0.0)
        self.assertAlmostEqual(
            ascending[50].tcp_z_m - ascending[50].sensor_z_m,
            self.profile.tcp_to_sensor_z_m,
        )

    def test_downward_quaternion_and_zone_transforms(self):
        direction = shape_replay.quaternion_rotate_vector(self.profile.sensor_quat_wxyz, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(direction[0], 0.0, places=12)
        self.assertAlmostEqual(direction[1], 0.0, places=12)
        self.assertAlmostEqual(direction[2], -1.0, places=12)
        values = list(range(64))
        self.assertEqual(shape_replay.transform_zones(values, "identity"), values)
        self.assertEqual(shape_replay.transform_zones(values, "rot180")[0], 63)
        self.assertEqual(shape_replay.transform_zones(values, "mirror")[:8], list(reversed(range(8))))

    def test_calibration_is_deterministic_and_beats_centered_baseline(self):
        samples = {
            direction: shape_replay.load_replay_samples(self.profile, direction)
            for direction in ("ascending", "descending")
        }
        result = shape_replay.calibrate_rigid_pose(self.profile, samples)
        self.assertGreater(result.score, result.centered_baseline_score)
        self.assertAlmostEqual(result.mesh_pose.x_m, 0.0012494615598963913, places=6)
        self.assertAlmostEqual(result.mesh_pose.y_m, -0.011395834617274564, places=6)
        self.assertAlmostEqual(result.mesh_pose.yaw_deg, 90.0, places=6)
        self.assertEqual(result.zone_transform, "identity")
        self.assertAlmostEqual(result.tcp_to_sensor_z_m, 0.09005696180259326, places=6)

    def test_prepare_replay_and_both_direction_child_commands(self):
        parser = main_module.build_arg_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(
                [
                    "--scene",
                    "shape-replay",
                    "--experiment-profile",
                    str(PROFILE_PATH),
                    "--experiment-direction",
                    "ascending",
                    "--experiment-output-dir",
                    tmpdir,
                ]
            )
            run = main_module.prepare_shape_replay(args)
            self.assertEqual(args.frames, 205)
            self.assertEqual(args.profile, self.profile.sensor_profile)
            self.assertTrue(args.disable_flat_csv)
            self.assertEqual(args.output_csv.name, "sim_matrix.csv")
            self.assertEqual(run.output_dir, Path(tmpdir) / "cup" / "ascending")

        command = main_module.build_shape_replay_direction_command(
            ["--scene", "shape-replay", "--experiment-direction", "both"],
            "descending",
            executable="isaac-python",
            script_path="runner.py",
        )
        self.assertEqual(command[:2], ["isaac-python", "runner.py"])
        direction_index = command.index("--experiment-direction")
        self.assertEqual(command[direction_index + 1], "descending")
        self.assertNotIn("both", command)

    def test_comparison_and_output_schemas_keep_raw_simulation_separate(self):
        samples = shape_replay.load_replay_samples(self.profile, "ascending")[:4]
        frames = [FakeFrame([list(sample.real_zones_mm[row * 8 : (row + 1) * 8]) for row in range(8)]) for sample in samples]
        rows, summary = shape_replay.build_comparison(samples, frames, "identity")
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary["distance_mae_mm"], 0.0)
        self.assertEqual(summary["distance_bias_mm"], 0.0)
        self.assertEqual(summary["mean_no_return_iou"], 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            run = shape_replay.ShapeReplayRun(self.profile, "ascending", list(samples), Path(tmpdir), [2, 8, 14, 20])
            output_summary = shape_replay.write_shape_replay_outputs(run, frames)
            self.assertTrue((Path(tmpdir) / "sim_flat.csv").is_file())
            self.assertTrue((Path(tmpdir) / "comparison.csv").is_file())
            self.assertTrue((Path(tmpdir) / "comparison_graph.png").is_file())
            self.assertTrue((Path(tmpdir) / "summary.json").is_file())
            with (Path(tmpdir) / "sim_flat.csv").open("r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(header[:7], ["reference_timestamp", "frame_index", "sim_tick", "elapsed_s", "tcp_z_m", "sensor_z_m", "valid_zones"])
            with (Path(tmpdir) / "summary.json").open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertTrue(saved["raw_rtx"])
            self.assertTrue(saved["comparison_graph_written"])
            self.assertEqual(output_summary["frames"], 4)


if __name__ == "__main__":
    unittest.main()
