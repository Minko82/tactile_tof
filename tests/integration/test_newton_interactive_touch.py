from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import REPO_ROOT, require_newton_runtime


pytestmark = pytest.mark.newton_integration

BASE_CONFIG = REPO_ROOT / "sim/config/mechanics/experiments/interactive_manual.json"


def _abbreviated_interactive_config(tmp_path: Path) -> Path:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base = BASE_CONFIG.parent
    config["asset_config"] = str((base / config["asset_config"]).resolve())
    config["material_config"] = str((base / config["material_config"]).resolve())
    config["probe"]["mesh"] = str((base / config["probe"]["mesh"]).resolve())
    config["equilibration"] = {
        "minimum_duration_s": 0.5,
        "maximum_duration_s": 2.0,
        "velocity_tolerance_m_s": 0.0001,
        "stable_frames": 5,
        "timeout_behavior": "fail",
    }
    config["solver"].update(
        {
            "substeps": 10,
            "vbd_iterations": 10,
            "newton_strict": True,
        }
    )
    config["output"] = {
        "directory": str((tmp_path / "configured_output").resolve()),
        "rate_hz": 10.0,
        "chunk_size_frames": 16,
    }
    path = tmp_path / "interactive.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def test_real_interactive_controller_settles_moves_contacts_and_resets(tmp_path):
    require_newton_runtime()
    from sim.mechanics.interactive_runner import InteractiveTouchController
    from sim.mechanics.interactive_safety import ProbePose

    config_path = _abbreviated_interactive_config(tmp_path)
    output_dir = tmp_path / "interactive_output"
    args = SimpleNamespace(
        config=str(config_path),
        output_dir=str(output_dir),
        record_video=False,
        video_path=None,
        strict_newton=True,
    )
    controller = InteractiveTouchController(None, args)
    try:
        for _ in range(130):
            controller.step()
            if controller.equilibrated_particle_positions is not None:
                break
        assert controller.equilibrated_particle_positions is not None
        initial_probe_position = controller.current_body_position.copy()
        initial_particles = controller.equilibrated_particle_positions.copy()

        controller.mouse_target_pose = ProbePose(
            initial_probe_position + np.asarray([0.0, 0.0, -0.012]),
            controller.indenter_quaternion,
        )
        contact_observed = False
        for _ in range(160):
            controller.step()
            contact_observed |= bool(controller.live_metrics.get("contact", False))
            if controller.safety_stop_active:
                break

        assert contact_observed
        assert controller.safety_stop_active
        assert controller.safety_stop_reason in {
            "commanded_indentation_limit",
            "estimated_force_limit",
            "minimum_relative_tet_volume",
        }
        assert controller.current_body_position[2] < initial_probe_position[2]
        assert np.isfinite(controller._particle_positions()).all()
        assert (
            np.min(controller._relative_tet_volumes(controller._particle_positions()))
            >= controller.manual_config["safety"]["stop_minimum_relative_tet_volume"]
        )
        assert controller.maximum_contact_buffer_count > 0
        assert not controller.contact_buffer_saturated

        safety_path = controller.last_safety_export_path
        assert safety_path is not None and safety_path.is_file()
        with np.load(safety_path) as safety:
            assert str(safety["safety_event_type"]) == "recoverable_safety_stop"
            assert str(safety["safety_reason"]) == controller.safety_stop_reason
            assert "minimum_relative_j" in safety
            assert "affected_tet_indices" in safety
            assert "probe_position_m" in safety
            assert "commanded_indentation_m" in safety

        stopped_position = controller.current_body_position.copy()
        controller.mouse_target_pose = ProbePose(
            stopped_position + np.asarray([0.0, 0.0, 0.001]),
            controller.indenter_quaternion,
        )
        for _ in range(20):
            controller.step()
            if not controller.safety_stop_active:
                break
        assert not controller.safety_stop_active
        assert controller.current_body_position[2] > stopped_position[2]

        snapshot = controller._save_current_state()
        assert snapshot is not None and snapshot.is_file()
        with np.load(snapshot) as saved:
            assert "estimated_world_reaction_n" in saved
            assert "deformed_inner_coating_vertices_m" in saved

        controller._reset_manual_state()
        assert controller.speed_regime == "free_space"
        np.testing.assert_allclose(
            controller.current_body_position,
            initial_probe_position,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            controller._particle_positions(),
            initial_particles,
            rtol=1.0e-6,
            atol=2.0e-7,
        )
    finally:
        controller.finalize()
