import unittest

import numpy as np

from sim.mechanics.mesh import (
    AssetValidationError,
    boundary_faces,
    validate_surface,
    validate_tets,
)


class AssetValidationTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.asarray(
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]]
        )
        self.tets = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        self.faces = boundary_faces(self.tets)

    def test_valid_surface_and_positive_tet(self):
        surface = validate_surface(self.vertices, self.faces)
        volume = validate_tets(
            self.vertices,
            self.tets,
            min_volume_m3=1.0e-15,
            min_quality=0.1,
            max_condition_number=10.0,
        )
        self.assertTrue(surface.watertight)
        self.assertTrue(surface.manifold)
        self.assertEqual(volume.negative_volume_count, 0)
        self.assertEqual(volume.tet_count, 1)

    def test_open_surface_is_rejected(self):
        with self.assertRaisesRegex(AssetValidationError, "open boundary edges"):
            validate_surface(self.vertices, self.faces[:-1])

    def test_inverted_tet_is_rejected(self):
        inverted = self.tets[:, [1, 0, 2, 3]]
        with self.assertRaisesRegex(AssetValidationError, "negatively wound"):
            validate_tets(
                self.vertices,
                inverted,
                min_volume_m3=1.0e-15,
                min_quality=0.1,
                max_condition_number=10.0,
            )


if __name__ == "__main__":
    unittest.main()
