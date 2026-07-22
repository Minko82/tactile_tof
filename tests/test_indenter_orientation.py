import math

import numpy as np
import pytest

from sim.mechanics.indenter import indenter_support_distance


def quaternion_y(degrees):
    radians = math.radians(degrees)
    return [0.0, math.sin(radians / 2.0), 0.0, math.cos(radians / 2.0)]


def test_rotated_plate_uses_oriented_support_distance():
    plate = {
        "type": "flat_plate",
        "width_m": 0.020,
        "depth_m": 0.010,
        "thickness_m": 0.002,
        "quaternion_xyzw": quaternion_y(90.0),
    }
    support = indenter_support_distance(plate, np.array([0.0, 0.0, 1.0]))
    assert support == pytest.approx(0.010)


def test_rotated_cylinder_uses_axis_and_radial_support():
    cylinder = {
        "type": "cylinder",
        "radius_m": 0.004,
        "height_m": 0.020,
        "quaternion_xyzw": quaternion_y(90.0),
    }
    axial = indenter_support_distance(cylinder, np.array([1.0, 0.0, 0.0]))
    radial = indenter_support_distance(cylinder, np.array([0.0, 0.0, 1.0]))
    assert axial == pytest.approx(0.010)
    assert radial == pytest.approx(0.004)
