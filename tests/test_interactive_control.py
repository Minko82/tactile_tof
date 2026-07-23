from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import warp as wp
import pytest
import trimesh

from sim.mechanics.config import ConfigError, load_run_config
from sim.mechanics.interactive_runner import (
    InteractiveTouchController,
    limited_transform_step,
    quaternion_from_axis_angle_xyzw,
)
from sim.scripts.run_interactive_touch import close_viewer_after_fatal_error


CONFIG_PATH = Path("sim/config/mechanics/experiments/interactive_manual.json")


def _portable_config() -> dict:
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base = CONFIG_PATH.parent
    source["asset_config"] = str((base / source["asset_config"]).resolve())
    source["material_config"] = str((base / source["material_config"]).resolve())
    source["probe"]["mesh"] = str((base / source["probe"]["mesh"]).resolve())
    return source


def test_interactive_example_reuses_mechanics_schema_without_trajectory_source():
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert "trajectory" not in source
    assert "indenter" not in source

    resolved = load_run_config(CONFIG_PATH)

    assert resolved["mode"] == "interactive_manual"
    assert resolved["_interactive_compatibility_generated"] is True
    assert resolved["probe"]["type"] == "rounded_block"
    assert Path(resolved["probe"]["mesh"]).is_absolute()
    assert resolved["manual_control"]["free_space_linear_speed_m_s"] == 0.025
    assert resolved["manual_control"]["near_contact_linear_speed_m_s"] == 0.005
    assert resolved["manual_control"]["contact_linear_speed_m_s"] == 0.0015
    assert (
        resolved["asset"]["interactive_safety"]["maximum_commanded_indentation_m"]
        == 0.00075
    )
    np.testing.assert_allclose(
        resolved["indenter"]["quaternion_xyzw"], [0.0, 0.0, 0.0, 1.0]
    )


def test_rounded_block_asset_is_watertight_and_has_configured_dimensions():
    resolved = load_run_config(CONFIG_PATH)
    mesh = trimesh.load_mesh(resolved["probe"]["mesh"], process=True)

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0.0
    np.testing.assert_allclose(
        mesh.extents * float(resolved["probe"]["scale_to_m"]),
        resolved["probe"]["dimensions_m"],
        atol=1.0e-9,
    )


def test_manual_translation_obeys_speed_and_per_frame_limits():
    limited_by_speed = limited_transform_step(
        current_position_m=[0.0, 0.0, 0.0],
        current_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        target_position_m=[1.0, 0.0, 0.0],
        target_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        dt_s=0.1,
        max_linear_speed_m_s=0.03,
        max_angular_speed_deg_s=90.0,
        max_translation_per_frame_m=0.01,
        max_rotation_per_frame_deg=3.0,
    )
    np.testing.assert_allclose(limited_by_speed.position_m, [0.003, 0.0, 0.0])
    np.testing.assert_allclose(limited_by_speed.linear_velocity_m_s, [0.03, 0.0, 0.0])

    limited_per_frame = limited_transform_step(
        current_position_m=[0.0, 0.0, 0.0],
        current_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        target_position_m=[1.0, 0.0, 0.0],
        target_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        dt_s=1.0,
        max_linear_speed_m_s=1.0,
        max_angular_speed_deg_s=90.0,
        max_translation_per_frame_m=0.001,
        max_rotation_per_frame_deg=3.0,
    )
    np.testing.assert_allclose(limited_per_frame.position_m, [0.001, 0.0, 0.0])


def test_manual_rotation_obeys_speed_and_per_frame_limits():
    target = quaternion_from_axis_angle_xyzw([0.0, 0.0, 1.0], np.pi)
    step = limited_transform_step(
        current_position_m=[0.0, 0.0, 0.0],
        current_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        target_position_m=[0.0, 0.0, 0.0],
        target_quaternion_xyzw=target,
        dt_s=0.1,
        max_linear_speed_m_s=0.03,
        max_angular_speed_deg_s=90.0,
        max_translation_per_frame_m=0.001,
        max_rotation_per_frame_deg=3.0,
    )
    expected = quaternion_from_axis_angle_xyzw([0.0, 0.0, 1.0], np.deg2rad(3.0))
    np.testing.assert_allclose(step.quaternion_xyzw, expected, atol=1.0e-12)
    np.testing.assert_allclose(
        step.angular_velocity_rad_s,
        [0.0, 0.0, np.deg2rad(30.0)],
        atol=1.0e-12,
    )


def test_contact_dependent_speed_transitions_free_near_and_contact():
    controller = InteractiveTouchController.__new__(InteractiveTouchController)
    controller.manual_config = {
        "free_space_linear_speed_m_s": 0.025,
        "near_contact_linear_speed_m_s": 0.005,
        "contact_linear_speed_m_s": 0.0015,
        "near_contact_distance_m": 0.002,
    }

    controller.last_contact_flag = False
    controller.current_commanded_indentation_m = -0.003
    assert controller._select_linear_speed() == 0.025
    assert controller.speed_regime == "free_space"

    controller.current_commanded_indentation_m = -0.001
    assert controller._select_linear_speed() == 0.005
    assert controller.speed_regime == "near_contact"

    controller.last_contact_flag = True
    assert controller._select_linear_speed() == 0.0015
    assert controller.speed_regime == "contact"


def test_fatal_cleanup_requests_exit_and_closes_viewer():
    events = []

    class EventLoop:
        def exit(self):
            events.append("exit")

    class App:
        event_loop = EventLoop()

    class Renderer:
        app = App()

    class Viewer:
        renderer = Renderer()

        def close(self):
            events.append("close")

    close_viewer_after_fatal_error(Viewer())

    assert events == ["exit", "close"]


def test_render_uses_per_point_arrays_required_by_gl_viewer():
    class RecordingViewer:
        def __init__(self):
            self.point_names = []

        def begin_frame(self, _time):
            pass

        def log_state(self, _state):
            pass

        def log_contacts(self, _contacts, _state):
            pass

        def log_points(self, name, points, radii, colors, hidden):
            assert hidden is False
            assert points is not None
            assert radii is not None
            assert radii.dtype == wp.float32
            assert len(radii) == len(points)
            assert colors is not None
            assert colors.dtype == wp.vec3
            assert len(colors) == len(points)
            self.point_names.append(name)

        def end_frame(self):
            pass

    device = "cpu"
    controller = InteractiveTouchController.__new__(InteractiveTouchController)
    controller.pending_reset = False
    controller.pending_baseline = False
    controller.pending_capture = False
    controller.viewer = RecordingViewer()
    controller.sim_time = 0.0
    controller.state_0 = object()
    controller.contacts = object()
    controller.display_config = {
        "show_contacts": True,
        "displacement_heatmap": True,
        "show_safety_tets": False,
        "show_mount_vertices": False,
    }
    controller.live_contact_points = wp.array(
        [[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device
    )
    controller.live_contact_radii = wp.full(1, 0.001, dtype=wp.float32, device=device)
    controller.live_contact_colors = wp.full(
        1, wp.vec3(1.0, 0.1, 0.1), dtype=wp.vec3, device=device
    )
    controller.heatmap_points = wp.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.001]],
        dtype=wp.vec3,
        device=device,
    )
    controller.heatmap_radii = wp.full(2, 0.0001, dtype=wp.float32, device=device)
    controller.heatmap_colors = wp.full(
        2, wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=device
    )
    controller.render()

    assert controller.viewer.point_names == [
        "/interactive/soft_contact_points",
        "/interactive/displacement_heatmap",
    ]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("manual_control", "max_linear_speed_m_s", 0.0),
        ("manual_control", "max_rotation_per_frame_deg", -1.0),
        ("display", "metrics_rate_hz", 0.0),
        ("probe", "color_rgb", [1.2, 0.0, 0.0]),
        ("probe", "initial_orientation_wxyz", [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_invalid_interactive_configuration_fails_cleanly(
    tmp_path, section, field, value
):
    source = _portable_config()
    invalid = copy.deepcopy(source)
    invalid[section][field] = value
    path = tmp_path / "interactive_invalid.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_run_config(path)


def test_invalid_interactive_safety_threshold_order_fails_cleanly(tmp_path):
    invalid = _portable_config()
    invalid["manual_control"]["safety"]["stop_minimum_relative_tet_volume"] = 0.10
    path = tmp_path / "interactive_invalid_safety.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(
        ConfigError,
        match="circuit_breaker < stop < warning",
    ):
        load_run_config(path)
