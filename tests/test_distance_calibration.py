from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "sim" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import distance_calibration as calibration  # noqa: E402
import run_vl53l8cx_isaac_tof as simulator  # noqa: E402
import shape_replay  # noqa: E402


@dataclass
class Frame:
    distances_mm: object
    rtx_ranges_m: object
    validity_mask: object
    projected_distances_mm: object | None = None
    comparison_distances_mm: object | None = None


class DistanceCalibrationTests(unittest.TestCase):
    def test_authored_angles_and_spherical_rays_are_canonical(self):
        config = simulator.VL53L8CXConfig()
        emitters = shape_replay.canonical_emitter_angles(8, 8, 45.0, 45.0)
        attrs = simulator._make_sensor_attributes(config)
        self.assertEqual([value.azimuth_deg for value in emitters], attrs["omni:sensor:Core:emitterState:s001:azimuthDeg"])
        self.assertEqual([value.elevation_deg for value in emitters], attrs["omni:sensor:Core:emitterState:s001:elevationDeg"])
        corner = simulator.zone_ray_direction(0, 0, config)
        expected = (
            math.cos(math.radians(22.5)) * math.cos(math.radians(-22.5)),
            math.cos(math.radians(22.5)) * math.sin(math.radians(-22.5)),
            math.sin(math.radians(22.5)),
        )
        for actual, wanted in zip(corner, expected):
            self.assertAlmostEqual(actual, wanted, places=14)
        tangent = simulator._normalize3((1.0, math.tan(math.radians(-22.5)), math.tan(math.radians(22.5))))
        self.assertNotAlmostEqual(corner[0], tangent[0], places=6)

    def test_world_ray_origin_rotation_and_stl_transform(self):
        profile = shape_replay.ShapeExperimentProfile.from_json(
            REPO_ROOT / "sim/config/shape_experiments/cup.json", require_files=False
        )
        self.assertEqual(
            shape_replay.sensor_world_position(profile, 0.4),
            (profile.sensor_xy_m[0], profile.sensor_xy_m[1], 0.4 - profile.tcp_to_sensor_z_m),
        )
        world_forward = shape_replay.quaternion_rotate_vector(profile.sensor_quat_wxyz, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(world_forward[2], -1.0, places=12)
        transformed = shape_replay.transform_stl_vertex_to_world(profile.mesh_origin_source_units, profile)
        self.assertAlmostEqual(transformed[0], profile.mesh_pose.x_m)
        self.assertAlmostEqual(transformed[1], profile.mesh_pose.y_m)
        self.assertAlmostEqual(transformed[2], profile.table_top_z_m)

    def test_duplicate_return_selection_precedes_rounding_and_keeps_metadata(self):
        config = simulator.VL53L8CXConfig()
        selected = simulator.select_returns_by_emitter(
            [0.10049, 0.10040],
            config,
            emitter_ids=[0, 0],
            intensities=[1.0, 2.0],
            material_ids=[10, 20],
        )
        self.assertAlmostEqual(selected.ranges_m[0][0], 0.10040)
        self.assertEqual(selected.distances_mm[0][0], 100)
        self.assertEqual(selected.intensities[0][0], 2.0)
        self.assertEqual(selected.material_ids[0][0], 20)
        self.assertEqual(selected.return_indices[0][0], 1)

    def test_modes_derive_independently_and_keep_validity(self):
        ranges = [[None for _ in range(8)] for _ in range(8)]
        validity = [[False for _ in range(8)] for _ in range(8)]
        ranges[0][0] = 0.1006
        validity[0][0] = True
        frame = Frame([[0] * 8 for _ in range(8)], ranges, validity)
        shape_replay.apply_frame_distance_modes(
            frame,
            rows=8,
            cols=8,
            min_mm=20,
            max_mm=4000,
            invalid_mm=0,
            projection=[0.5] * 64,
            mode="off",
        )
        self.assertEqual(frame.distances_mm[0][0], 101)
        self.assertEqual(frame.projected_distances_mm[0][0], 50)
        self.assertIsNone(frame.comparison_distances_mm)
        self.assertEqual(frame.distances_mm[0][1], 0)
        self.assertEqual(frame.projected_distances_mm[0][1], 0)

    def test_plateau_segmentation_rejects_slow_transition_without_subdivision(self):
        samples = []
        for index, z_mm in enumerate([100, 100.1, 100.0, 100.1, 101, 102, 103, 104, 130, 130.1, 130.0, 130.1]):
            samples.append(
                shape_replay.ReplaySample(str(index), index * 0.1, z_mm / 1000.0, 0.0, tuple([100] * 64))
            )
        settings = calibration.TrainingSettings(max_adjacent_tcp_delta_mm=1.1)
        plateaus = calibration.segment_plateaus("cup", samples, settings)
        self.assertEqual(len(plateaus), 1)
        self.assertAlmostEqual(plateaus[0].median_tcp_z_mm, 130.05, places=2)

    def test_current_profiles_pass_declared_mount_compatibility(self):
        settings = calibration.TrainingSettings()
        datasets = []
        for name in ("cup", "spoon"):
            profile_path = REPO_ROOT / f"sim/config/shape_experiments/{name}.json"
            profile = shape_replay.ShapeExperimentProfile.from_json(profile_path, require_files=False)
            with profile.sensor_profile.open(encoding="utf-8") as handle:
                sensor = json.load(handle)
            entry = calibration.ManifestEntry(name, "ascending", profile_path, Path(), Path(), Path())
            datasets.append(calibration.TrainingDataset(entry, profile, sensor, (), (), ()))
        result = calibration.validate_shared_compatibility(datasets)
        self.assertLess(result["profiles"]["spoon"]["difference_from_reference_mm"], 0.10)
        self.assertEqual(result["profiles"]["spoon"]["orientation_difference_from_reference_deg"], 0.0)

        changed_xy = replace(datasets[1].profile, sensor_xy_m=(99.0, -99.0))
        calibration.validate_shared_compatibility([datasets[0], replace(datasets[1], profile=changed_xy)])

    def test_manifest_rejects_descending_without_accessing_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": calibration.MANIFEST_SCHEMA,
                        "training_regime_id": calibration.TRAINING_REGIME,
                        "entries": [
                            {"name": "cup", "direction": "descending", "profile": "cup.json", "robot_csv": "x", "real_tof_csv": "y", "simulation_csv": "z"}
                        ],
                        "output_artifact": "artifact.json",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "direction must be exactly 'ascending'"):
                calibration.TrainingManifest.from_json(manifest)

    def test_zone_residuals_recenter_and_global_receives_common_component(self):
        settings = calibration.TrainingSettings(min_heights_per_shape=1, min_total_height_groups=2)
        cells = {}
        projected = {}
        for shape in ("cup", "spoon"):
            for zone in range(64):
                key = (shape, 0, zone)
                cells[key] = [10.0 + zone / 10.0] * 4
                projected[key] = [200.0] * 4
        fit = calibration._fit_from_cells(cells, projected, settings, 1.0, strict=True)
        self.assertAlmostEqual(sum(fit.residuals_mm), 0.0, places=10)
        self.assertAlmostEqual(fit.final_global_mm, fit.provisional_global_mm + fit.common_component_mm)

    def test_cross_validation_includes_zero_and_uses_larger_lambda_tie_break(self):
        settings = calibration.TrainingSettings(min_heights_per_shape=1, min_total_height_groups=2)
        cells = {}
        projected = {}
        for shape in ("cup", "spoon"):
            for plateau in range(3):
                for zone in range(64):
                    key = (shape, plateau, zone)
                    cells[key] = [0.0] * 4
                    projected[key] = [200.0] * 4
        selected, report = calibration._cross_validate(cells, projected, settings)
        self.assertIn(0.0, report["lambda_grid"])
        self.assertEqual(selected, 16.0)

    def test_real_zone_transform_happens_at_ingestion_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "real.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp"] + [f"zone_{zone:02d}" for zone in range(64)])
                writer.writerow(["2026-01-01T00:00:00"] + list(range(64)))
            loaded = shape_replay.load_tof_csv(path, zone_transform="mirror")
        self.assertEqual(loaded[0].zones_mm[:8], tuple(reversed(range(8))))
        sample = shape_replay.ReplaySample("t", 0.0, 0.3, 0.2, loaded[0].zones_mm)
        frame = Frame(
            [list(loaded[0].zones_mm[start : start + 8]) for start in range(0, 64, 8)],
            [[0.1] * 8 for _ in range(8)],
            [[True] * 8 for _ in range(8)],
        )
        frame.projected_distances_mm = frame.distances_mm
        rows, _summary = shape_replay.build_comparison([sample], [frame], "mirror")
        self.assertEqual(rows[0]["mae_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
