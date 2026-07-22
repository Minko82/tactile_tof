import unittest

import numpy as np

from sim.mechanics.trajectory import PrescribedTrajectory


def trajectory():
    return PrescribedTrajectory(
        {
            "clearance_m": 0.0005,
            "indentation_m": 0.00075,
            "easing": "linear",
            "durations_s": {
                "approach": 0.25,
                "press": 0.375,
                "hold": 0.5,
                "release": 0.625,
                "recovery": 0.5,
            },
            "lateral_slip": {"enabled": False},
        }
    )


class PressHoldReleaseTests(unittest.TestCase):
    def test_state_machine_reaches_hold_and_recovers(self):
        motion = trajectory()
        samples = [
            motion.sample(time)
            for time in np.linspace(0.0, motion.total_duration_s, 501)
        ]
        phases = {sample.phase for sample in samples}
        self.assertEqual(phases, {"approach", "press", "hold", "release", "recovery"})
        hold = motion.sample(motion.starts["hold"] + 0.25)
        self.assertAlmostEqual(motion.nominal_indentation_m(hold), 0.00075)
        final = motion.sample(motion.total_duration_s)
        self.assertEqual(final.phase, "recovery")
        self.assertEqual(final.normal_travel_m, 0.0)
        self.assertEqual(final.normal_velocity_m_s, 0.0)

    def test_contact_starts_in_press_and_ends_in_release(self):
        motion = trajectory()
        times = np.linspace(0.0, motion.total_duration_s, 2001)
        samples = [motion.sample(time) for time in times]
        indentation = np.asarray(
            [motion.nominal_indentation_m(sample) for sample in samples]
        )
        normal_force = 10000.0 * indentation
        active = np.flatnonzero(normal_force > 1.0e-9)
        self.assertEqual(samples[int(active[0])].phase, "press")
        self.assertEqual(samples[int(active[-1])].phase, "release")
        press_force = normal_force[[sample.phase == "press" for sample in samples]]
        self.assertTrue(np.all(np.diff(press_force) >= -1.0e-12))
        recovery_force = normal_force[
            [sample.phase == "recovery" for sample in samples]
        ]
        np.testing.assert_allclose(recovery_force, 0.0)


if __name__ == "__main__":
    unittest.main()
