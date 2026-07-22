import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim.mechanics.exporter import MechanicalDataExporter
from sim.mechanics.trajectory import DeterministicTrajectory


CONFIG = {
    "clearance_m": 0.001,
    "indentation_m": 0.002,
    "easing": "smoothstep",
    "durations_s": {
        "approach": 0.2,
        "press": 0.4,
        "hold": 0.1,
        "release": 0.5,
        "recovery": 0.2,
    },
    "lateral_slip": {
        "enabled": True,
        "duration_s": 0.3,
        "distance_m": 0.001,
        "direction": [1.0, 0.0, 0.0],
    },
}


class RepeatabilityTests(unittest.TestCase):
    def test_identical_trajectory_configs_are_equivalent(self):
        first = DeterministicTrajectory(CONFIG)
        second = DeterministicTrajectory(CONFIG)
        times = np.linspace(0.0, first.total_duration_s, 1001)
        a = np.asarray(
            [
                np.r_[
                    sample.normal_travel_m,
                    sample.normal_velocity_m_s,
                    sample.lateral_offset_m,
                ]
                for sample in map(first.sample, times)
            ]
        )
        b = np.asarray(
            [
                np.r_[
                    sample.normal_travel_m,
                    sample.normal_velocity_m_s,
                    sample.lateral_offset_m,
                ]
                for sample in map(second.sample, times)
            ]
        )
        np.testing.assert_array_equal(a, b)

    def test_exported_arrays_round_trip_without_randomness(self):
        with tempfile.TemporaryDirectory() as temporary:
            exporter = MechanicalDataExporter(Path(temporary), {"seedless": True})
            exporter.append(
                {
                    "timestamp_s": 0.0,
                    "trajectory_phase": "approach",
                    "tet_particle_positions_m": np.arange(12, dtype=float).reshape(
                        4, 3
                    ),
                },
                {"timestamp_s": 0.0, "trajectory_phase": "approach"},
            )
            exporter.finalize()
            with np.load(Path(temporary) / "frames.npz") as loaded:
                np.testing.assert_array_equal(
                    loaded["tet_particle_positions_m"],
                    np.arange(12, dtype=float).reshape(1, 4, 3),
                )


if __name__ == "__main__":
    unittest.main()
