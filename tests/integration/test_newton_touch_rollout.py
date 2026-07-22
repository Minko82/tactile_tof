import json
from pathlib import Path

import numpy as np
import pytest

from .helpers import abbreviated_config, run_rollout, run_rollout_process


pytestmark = pytest.mark.newton_integration


def test_real_newton_touch_rollout_settles_contacts_and_releases(tmp_path):
    config_path = abbreviated_config(tmp_path, "rollout")
    output_dir = tmp_path / "rollout_output"

    frames, metrics = run_rollout(config_path, output_dir)

    assert np.isfinite(frames["tet_particle_positions_m"]).all()
    assert np.isfinite(frames["object_position_m"]).all()
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

    run_config = json.loads(
        (output_dir / "run_config.json").read_text(encoding="utf-8")
    )
    requested_clearance = float(run_config["trajectory"]["requested_clearance_m"])
    effective_clearance = float(run_config["trajectory"]["clearance_m"])
    assert effective_clearance >= requested_clearance

    phases_array = frames["trajectory_phase"]
    baseline_index = np.flatnonzero(phases_array == "capture_baseline")[-1]
    approach_index = np.flatnonzero(phases_array == "approach")[0]
    approach_time = float(frames["trajectory_time_s"][approach_index])
    approach_duration = float(run_config["trajectory"]["durations_s"]["approach"])
    expected_travel = effective_clearance * min(approach_time / approach_duration, 1.0)
    direction = np.asarray(run_config["contact"]["direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    actual_travel = (
        frames["object_position_m"][approach_index]
        - frames["object_position_m"][baseline_index]
    )
    np.testing.assert_allclose(
        actual_travel,
        direction * expected_travel,
        rtol=1.0e-6,
        atol=1.0e-9,
    )

    assert not np.any(frames["contact_buffer_saturation_flag"])
    assert np.max(frames["contact_buffer_maximum_count_observed"]) > 0
    assert np.all(
        frames["contact_buffer_maximum_count_observed"]
        < frames["contact_buffer_configured_capacity"]
    )
    environment = json.loads(
        (output_dir / "newton_environment.json").read_text(encoding="utf-8")
    )
    assert environment["contact_buffer_saturation_flag"] is False
    assert environment["maximum_contact_count_observed"] > 0
    assert (
        environment["maximum_contact_count_observed"]
        < environment["configured_contact_capacity"]
    )

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


def test_real_newton_rollout_fails_on_contact_buffer_saturation(tmp_path):
    config_path = abbreviated_config(tmp_path, "saturation")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["solver"]["rigid_body_particle_contact_buffer_size"] = 1
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    output_dir = tmp_path / "saturation_output"

    completed = run_rollout_process(config_path, output_dir)

    assert completed.returncode != 0
    combined_output = completed.stdout + completed.stderr
    assert "Rigid-body particle-contact buffer saturated" in combined_output
    with np.load(output_dir / "failure_state.npz") as failure:
        assert str(failure["failure_reason"]) == "contact_buffer_saturation"
        assert int(failure["configured_contact_capacity"]) == 1
        assert int(failure["observed_contact_count"]) >= 1
    environment = json.loads(
        (output_dir / "newton_environment.json").read_text(encoding="utf-8")
    )
    assert environment["contact_buffer_saturation_flag"] is True
    assert environment["maximum_contact_count_observed"] >= 1
    assert environment["first_contact_buffer_saturation_substep"] is not None
