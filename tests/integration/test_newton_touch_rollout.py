from pathlib import Path

import numpy as np
import pytest

from .helpers import abbreviated_config, run_rollout


pytestmark = pytest.mark.newton_integration


def test_real_newton_touch_rollout_settles_contacts_and_releases(tmp_path):
    config_path = abbreviated_config(tmp_path, "rollout")
    output_dir = tmp_path / "rollout_output"

    frames, metrics = run_rollout(config_path, output_dir)

    assert np.isfinite(frames["tet_particle_positions_m"]).all()
    assert np.min(frames["minimum_relative_tet_volume"]) >= 0.15
    assert np.any(frames["contact_flag"])
    assert not bool(frames["contact_flag"][-1])
    phases = set(frames["trajectory_phase"].tolist())
    assert {"settling", "capture_baseline", "press", "release", "recovery"} <= phases
    contact_phases = frames["trajectory_phase"][frames["contact_flag"].astype(bool)]
    assert contact_phases[0] in {"press", "hold"}
    assert contact_phases[-1] in {"hold", "release"}
    assert np.isfinite(frames["displacement_from_equilibrated_baseline_m"][2:]).any()
    assert float(metrics[-1]["estimated_axial_reaction_n"]) == 0.0

    for filename in (
        "run_config.json",
        "asset_manifest.json",
        "regions.npz",
        "surface_mapping.npz",
        "newton_environment.json",
        "frames_manifest.json",
        "metrics.csv",
    ):
        assert (output_dir / filename).is_file(), filename
    assert list(Path(output_dir).glob("frames_*.npz"))
