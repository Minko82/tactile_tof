from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "sim" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_video_module():
    path = SCRIPT_DIR / "record_shape_comparison.py"
    spec = importlib.util.spec_from_file_location("shape_comparison_video", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


video = load_video_module()


class ShapeComparisonVideoTests(unittest.TestCase):
    def write_flat_csv(self, path: Path, timestamp_column: str, rows: list[tuple[str, list[int]]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([timestamp_column, "frame_index", "elapsed_s", "tcp_z_m"] + video.ZONE_COLUMNS)
            for index, (timestamp, zones) in enumerate(rows):
                writer.writerow([timestamp, index, index * 0.1, 0.4 - index * 0.03] + zones)

    def test_load_align_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            real_path = root / "real.csv"
            sim_path = root / "sim.csv"
            real_zones = [100] * 64
            real_zones[0] = 0
            sim_zones = [110] * 64
            sim_zones[1] = 0
            self.write_flat_csv(real_path, "timestamp", [("2026-01-01T00:00:00.000", real_zones)])
            self.write_flat_csv(sim_path, "reference_timestamp", [("2026-01-01T00:00:00.000", sim_zones)])

            aligned = video.align_frames(video.load_zone_frames(real_path), video.load_zone_frames(sim_path))
            self.assertEqual(len(aligned), 1)
            self.assertEqual(aligned[0].sim.tcp_z_m, 0.4)
            metrics = video.frame_metrics(aligned[0].real.zones_mm, aligned[0].sim.zones_mm)
            self.assertEqual(metrics["real_valid"], 63)
            self.assertEqual(metrics["sim_valid"], 63)
            self.assertEqual(metrics["paired_valid"], 62)
            self.assertEqual(metrics["sim_only_returns"], 1)
            self.assertEqual(metrics["real_only_returns"], 1)
            self.assertEqual(metrics["mae_mm"], 10.0)
            self.assertEqual(metrics["bias_mm"], 10.0)
            self.assertEqual(metrics["no_return_iou"], 0.0)

    def test_alignment_requires_every_sim_timestamp(self):
        real = [video.ZoneFrame("real-only", tuple([100] * 64))]
        sim = [video.ZoneFrame("sim-only", tuple([100] * 64))]
        with self.assertRaisesRegex(ValueError, "no exact real frame"):
            video.align_frames(real, sim)

    def test_zone_transforms_match_expected_orientation(self):
        values = tuple(range(64))
        self.assertEqual(video.transform_zones(values, "identity"), values)
        self.assertEqual(video.transform_zones(values, "rot180")[0], 63)
        self.assertEqual(video.transform_zones(values, "mirror")[:8], tuple(reversed(range(8))))

    def test_distance_mode_columns_and_one_time_real_ingestion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            real_path = root / "real.csv"
            sim_path = root / "sim.csv"
            values = list(range(64))
            self.write_flat_csv(real_path, "timestamp", [("t", values)])
            with sim_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["reference_timestamp"] + [f"projected_zone_{zone:02d}" for zone in range(64)])
                writer.writerow(["t"] + list(reversed(values)))
            real = video.load_zone_frames(real_path, zone_transform="mirror")
            sim = video.load_zone_frames(sim_path, sim_distance_mode="projected")
            aligned = video.align_frames(real, sim)
            self.assertEqual(aligned[0].real.zones_mm[:8], tuple(reversed(range(8))))
            with self.assertRaisesRegex(ValueError, "applied exactly once"):
                video.align_frames(real, sim, "mirror")
            with self.assertRaisesRegex(ValueError, "does not provide"):
                video.load_zone_frames(sim_path, sim_distance_mode="comparison")


if __name__ == "__main__":
    unittest.main()
