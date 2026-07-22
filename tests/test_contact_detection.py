import unittest

import numpy as np

from sim.mechanics.contact import contact_face_mask, estimate_contact_summary


def summary(position_z, radius=0.001):
    positions = np.asarray([[0.0, 0.0, position_z]])
    return estimate_contact_summary(
        particle_positions_m=positions,
        previous_particle_positions_m=positions.copy(),
        particle_radii_m=np.asarray([radius]),
        contact_particles=np.asarray([0], dtype=np.int32),
        contact_normals=np.asarray([[0.0, 0.0, -1.0]]),
        contact_body_positions_m=np.asarray([[0.0, 0.0, 0.0]]),
        penalty_stiffness_n_m=np.asarray([10000.0]),
        damping_ratio=np.asarray([0.0]),
        friction_coefficients=np.asarray([0.0]),
        body_position_m=np.zeros(3),
        body_quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        body_linear_velocity_m_s=np.zeros(3),
        body_angular_velocity_rad_s=np.zeros(3),
        loading_direction=np.asarray([0.0, 0.0, -1.0]),
        dt_s=0.001,
        force_threshold_n=1.0e-6,
        friction_epsilon_m_s=1.0e-5,
    )


class ContactDetectionTests(unittest.TestCase):
    def test_contact_flag_uses_reaction_not_candidate_count(self):
        active = summary(-0.0001)
        margin_only = summary(-0.002)
        self.assertTrue(active.contact_flag)
        self.assertGreater(active.estimated_axial_reaction_n, 0.0)
        self.assertFalse(margin_only.contact_flag)
        self.assertEqual(margin_only.estimated_axial_reaction_n, 0.0)

    def test_contact_face_mask_tracks_active_particle(self):
        vertices = np.asarray(
            [[-0.001, -0.001, 0.0], [0.001, -0.001, 0.0], [0.0, 0.001, 0.0]]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)
        mask = contact_face_mask(
            vertices,
            faces,
            np.asarray([0]),
            np.asarray([[0.0, 0.0, -0.0001]]),
            distance_m=0.0004,
        )
        self.assertEqual(mask.tolist(), [True])


if __name__ == "__main__":
    unittest.main()
