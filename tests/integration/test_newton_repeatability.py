import json

import numpy as np
import pytest

from .helpers import abbreviated_config, run_rollout


pytestmark = pytest.mark.newton_integration

# CUDA VBD is compared numerically, not bitwise. These tolerances correspond to
# 0.2 micrometres in position, 1e-6 in relative volume, and 0.1 mN reaction.
POSITION_ATOL_M = 2.0e-7
RELATIVE_VOLUME_ATOL = 1.0e-6
REACTION_ATOL_N = 1.0e-4


def test_two_real_newton_rollouts_agree_within_documented_tolerances(tmp_path):
    config_path = abbreviated_config(tmp_path, "repeatability")
    first, _ = run_rollout(config_path, tmp_path / "first")
    second, _ = run_rollout(config_path, tmp_path / "second")

    np.testing.assert_allclose(
        first["tet_particle_positions_m"],
        second["tet_particle_positions_m"],
        rtol=1.0e-5,
        atol=POSITION_ATOL_M,
    )
    np.testing.assert_allclose(
        first["object_position_m"], second["object_position_m"], atol=1.0e-12
    )
    np.testing.assert_allclose(
        first["object_quaternion_xyzw"],
        second["object_quaternion_xyzw"],
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(first["contact_flag"], second["contact_flag"])
    np.testing.assert_allclose(
        first["minimum_relative_tet_volume"],
        second["minimum_relative_tet_volume"],
        atol=RELATIVE_VOLUME_ATOL,
    )
    np.testing.assert_allclose(
        first["estimated_axial_reaction_n"],
        second["estimated_axial_reaction_n"],
        rtol=1.0e-3,
        atol=REACTION_ATOL_N,
    )


def test_output_rate_does_not_change_final_particle_state(tmp_path):
    base_path = abbreviated_config(tmp_path, "output_rate_invariance")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    low_rate = json.loads(json.dumps(base))
    high_rate = json.loads(json.dumps(base))
    low_rate["output"]["rate_hz"] = 10.0
    high_rate["output"]["rate_hz"] = 60.0
    low_path = tmp_path / "output_10hz.json"
    high_path = tmp_path / "output_60hz.json"
    low_path.write_text(json.dumps(low_rate, indent=2) + "\n", encoding="utf-8")
    high_path.write_text(json.dumps(high_rate, indent=2) + "\n", encoding="utf-8")

    low_frames, _ = run_rollout(low_path, tmp_path / "low_rate")
    high_frames, _ = run_rollout(high_path, tmp_path / "high_rate")

    np.testing.assert_allclose(
        low_frames["tet_particle_positions_m"][-1],
        high_frames["tet_particle_positions_m"][-1],
        rtol=1.0e-5,
        atol=POSITION_ATOL_M,
    )
    np.testing.assert_allclose(
        low_frames["object_position_m"][-1],
        high_frames["object_position_m"][-1],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        low_frames["minimum_relative_tet_volume"][-1],
        high_frames["minimum_relative_tet_volume"][-1],
        atol=RELATIVE_VOLUME_ATOL,
    )
