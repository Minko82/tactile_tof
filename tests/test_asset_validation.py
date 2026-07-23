import unittest
from pathlib import Path

import numpy as np

from sim.mechanics.mesh import (
    AssetValidationError,
    boundary_faces,
    inspect_surface,
    validate_surface,
    validate_tets,
    weld_stl_vertices,
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

    def test_inconsistent_surface_winding_is_rejected(self):
        faces = self.faces.copy()
        faces[0] = faces[0, ::-1]
        report = inspect_surface(self.vertices, faces)
        self.assertFalse(report.winding_consistent)
        with self.assertRaisesRegex(
            AssetValidationError, "edges have inconsistent face winding"
        ):
            validate_surface(self.vertices, faces)
    def test_invalid_two_body_mold_fixture_is_rejected(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest("trimesh is an optional asset-preparation dependency")

        fixture = Path(__file__).parent / "fixtures/invalid_two_body_mold.stl"
        loaded = trimesh.load_mesh(fixture, process=False)
        vertices, faces = weld_stl_vertices(
            np.asarray(loaded.vertices),
            np.asarray(loaded.faces),
            decimal_digits=7,
        )
        with self.assertRaisesRegex(
            AssetValidationError, "204 duplicate faces; 715 non-manifold edges"
        ):
            validate_surface(vertices * 1.0e-3, faces)

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
