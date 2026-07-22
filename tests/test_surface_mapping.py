import unittest

import numpy as np

from sim.mechanics.mapping import build_surface_mapping, reconstruct_surface


class SurfaceMappingTests(unittest.TestCase):
    def test_barycentric_mapping_follows_affine_deformation(self):
        rest = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        surface = np.asarray(
            [[0.0, 0.0, 0.0], [0.2, 0.3, 0.0], [0.1, 0.2, 0.3], [0.0, 0.0, 1.0]]
        )
        mapping = build_surface_mapping(surface, rest, tets, candidate_count=1)
        affine = np.asarray([[1.2, 0.1, 0.0], [0.0, 0.9, 0.2], [0.1, 0.0, 1.1]])
        translation = np.asarray([0.4, -0.2, 0.3])
        deformed_tets = rest @ affine.T + translation
        reconstructed = reconstruct_surface(deformed_tets, tets, mapping)
        expected = surface @ affine.T + translation
        np.testing.assert_allclose(reconstructed, expected, atol=1.0e-12)
        self.assertLess(float(np.max(mapping.reconstruction_error_m)), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
