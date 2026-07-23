"""Interactive, speed-limited fingertip mechanics controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton._src.viewer.picking import Picking

from .contact import contact_face_mask, masked_triangle_area
from .exporter import export_failure_state
from .indenter import normalized_quaternion_xyzw, quaternion_rotate_xyzw
from .interactive_safety import (
    CandidateSafety,
    ProbePose,
    commanded_indentation,
    evaluate_candidate_safety,
)
from .mapping import reconstruct_surface
from .newton_runner import TouchMechanicsControllerV2, _set_kinematic_pose_v2


def quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the normalized Hamilton product ``left * right``."""

    left = normalized_quaternion_xyzw(left)
    right = normalized_quaternion_xyzw(right)
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return normalized_quaternion_xyzw(
        np.asarray(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float64,
        )
    )


def quaternion_from_axis_angle_xyzw(axis: Any, angle_rad: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis_array))
    if axis_array.shape != (3,) or not np.isfinite(axis_array).all() or norm <= 0.0:
        raise ValueError("rotation axis must be a finite nonzero 3-vector")
    half = 0.5 * float(angle_rad)
    return np.concatenate(
        (axis_array / norm * math.sin(half), np.asarray([math.cos(half)]))
    )


def _quaternion_delta_xyzw(
    current: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    current = normalized_quaternion_xyzw(current)
    target = normalized_quaternion_xyzw(target)
    conjugate = np.asarray([-current[0], -current[1], -current[2], current[3]])
    delta = quaternion_multiply_xyzw(target, conjugate)
    if delta[3] < 0.0:
        delta = -delta
    scalar = float(np.clip(delta[3], -1.0, 1.0))
    angle = 2.0 * math.acos(scalar)
    sine = math.sqrt(max(0.0, 1.0 - scalar * scalar))
    axis = (
        np.asarray([1.0, 0.0, 0.0])
        if sine <= 1.0e-12
        else np.asarray(delta[:3], dtype=np.float64) / sine
    )
    return delta, angle, axis


@dataclass(frozen=True)
class LimitedTransformStep:
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray


def limited_transform_step(
    *,
    current_position_m: Any,
    current_quaternion_xyzw: Any,
    target_position_m: Any,
    target_quaternion_xyzw: Any,
    dt_s: float,
    max_linear_speed_m_s: float,
    max_angular_speed_deg_s: float,
    max_translation_per_frame_m: float,
    max_rotation_per_frame_deg: float,
) -> LimitedTransformStep:
    """Move one frame toward a manual target without exceeding either limit."""

    dt_s = float(dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    current_position = np.asarray(current_position_m, dtype=np.float64)
    target_position = np.asarray(target_position_m, dtype=np.float64)
    if (
        current_position.shape != (3,)
        or target_position.shape != (3,)
        or not np.isfinite(current_position).all()
        or not np.isfinite(target_position).all()
    ):
        raise ValueError("manual positions must be finite 3-vectors")
    current_quaternion = normalized_quaternion_xyzw(current_quaternion_xyzw)
    target_quaternion = normalized_quaternion_xyzw(target_quaternion_xyzw)

    translation = target_position - current_position
    translation_distance = float(np.linalg.norm(translation))
    translation_limit = min(
        float(max_linear_speed_m_s) * dt_s,
        float(max_translation_per_frame_m),
    )
    if translation_distance > translation_limit > 0.0:
        translation *= translation_limit / translation_distance
    next_position = current_position + translation
    linear_velocity = translation / dt_s

    _, target_angle, axis = _quaternion_delta_xyzw(
        current_quaternion, target_quaternion
    )
    rotation_limit = min(
        math.radians(float(max_angular_speed_deg_s)) * dt_s,
        math.radians(float(max_rotation_per_frame_deg)),
    )
    step_angle = min(target_angle, rotation_limit)
    if step_angle <= 1.0e-14:
        next_quaternion = current_quaternion.copy()
        angular_velocity = np.zeros(3, dtype=np.float64)
    else:
        step_rotation = quaternion_from_axis_angle_xyzw(axis, step_angle)
        next_quaternion = quaternion_multiply_xyzw(step_rotation, current_quaternion)
        angular_velocity = axis * (step_angle / dt_s)

    return LimitedTransformStep(
        position_m=next_position,
        quaternion_xyzw=next_quaternion,
        linear_velocity_m_s=linear_velocity,
        angular_velocity_rad_s=angular_velocity,
    )


class KinematicProbePicking(Picking):
    """Use Newton's ray/drag target while admitting exactly one kinematic body."""

    def __init__(self, model, probe_body: int, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.probe_body = int(probe_body)

    def pick(self, state, ray_start, ray_dir) -> None:
        flags = self.model.body_flags.numpy()
        original = int(flags[self.probe_body])
        flags[self.probe_body] = original & ~int(newton.BodyFlags.KINEMATIC)
        self.model.body_flags.assign(flags)
        try:
            super().pick(state, ray_start, ray_dir)
        finally:
            flags[self.probe_body] = original
            self.model.body_flags.assign(flags)
        if int(self.pick_body.numpy()[0]) != self.probe_body:
            self.release()


class InteractiveTouchController(TouchMechanicsControllerV2):
    """Equilibrate once, then follow a mouse-controlled kinematic probe target."""

    def __init__(self, viewer, args: Any):
        self.probe_shape = -1
        self.probe_local_vertices_m: np.ndarray | None = None
        super().__init__(viewer, args)
        if self.config.get("mode") != "interactive_manual":
            raise ValueError("interactive runner requires mode='interactive_manual'")

        self.manual_config = self.config["manual_control"]
        self.display_config = self.config["display"]
        probe = self.config["probe"]
        self.probe_initial_position = np.asarray(
            probe["initial_position_m"], dtype=np.float64
        )
        orientation_wxyz = np.asarray(
            probe["initial_orientation_wxyz"], dtype=np.float64
        )
        self.probe_initial_quaternion = normalized_quaternion_xyzw(
            [
                orientation_wxyz[1],
                orientation_wxyz[2],
                orientation_wxyz[3],
                orientation_wxyz[0],
            ],
            name="probe initial_orientation_wxyz",
        )
        self.current_body_position = self.probe_initial_position.copy()
        self.indenter_quaternion = self.probe_initial_quaternion.copy()
        self.current_body_velocity = np.zeros(3, dtype=np.float64)
        self.current_body_angular_velocity = np.zeros(3, dtype=np.float64)
        initial_pose = ProbePose(
            self.probe_initial_position,
            self.probe_initial_quaternion,
        )
        self.mouse_target_pose = initial_pose.copy()
        self.last_safe_probe_pose = initial_pose.copy()
        self.safety_config = self.manual_config["safety"]
        self.maximum_commanded_indentation_m = float(
            self.config["asset"]["interactive_safety"][
                "maximum_commanded_indentation_m"
            ]
        )
        self.contact_reference_location_m = np.asarray(
            self.config["contact"].get(
                "manual_reference_location_m",
                self.config["contact"]["location_m"],
            ),
            dtype=np.float64,
        )
        self.current_commanded_indentation_m = self._commanded_indentation_for_pose(
            initial_pose
        )
        self.last_contact_flag = False
        self.speed_regime = "free_space"
        self.safety_stop_active = False
        self.safety_stop_reason = ""
        self.safety_warning_reasons: tuple[str, ...] = ()
        self.safety_stop_indentation_m = float("-inf")
        self.safety_stop_count = 0
        self.safety_snapshot_index = 0
        self.safety_affected_tet_indices = np.empty(0, dtype=np.int32)
        self.last_safety_evaluation: CandidateSafety | None = None
        self._set_probe_pose_on_states(
            self.probe_initial_position,
            self.probe_initial_quaternion,
            np.zeros(3),
            np.zeros(3),
        )

        color = tuple(float(component) for component in probe["color_rgb"])
        if self.probe_shape < 0:
            raise RuntimeError("interactive probe shape was not created")
        self.model.shape_color[self.probe_shape : self.probe_shape + 1].fill_(
            wp.vec3(*color)
        )

        self.lifecycle_phase = "initialization"
        self.rotation_input_pending = False
        self.pending_reset = False
        self.pending_baseline = False
        self.pending_capture = False
        self.snapshot_index = 0
        self.live_metrics: dict[str, Any] = {}
        self.live_contact_points: wp.array | None = None
        self.live_contact_radii: wp.array | None = None
        self.live_contact_colors: wp.array | None = None
        self.heatmap_points: wp.array | None = None
        self.heatmap_radii: wp.array | None = None
        self.heatmap_colors: wp.array | None = None
        self.safety_tet_line_starts: wp.array | None = None
        self.safety_tet_line_ends: wp.array | None = None
        self.safety_tet_line_colors: wp.array | None = None
        self.mount_visualization_points: wp.array | None = None
        self.mount_visualization_radii: wp.array | None = None
        self.mount_visualization_colors: wp.array | None = None
        self.last_safety_export_path: Path | None = None
        self.minimum_relative_j_seen = float("inf")
        self.fatal_error_message = ""
        self.next_metrics_time_s = 0.0

        self.environment.update(
            {
                "mode": "interactive_manual",
                "trajectory_execution": "disabled",
                "probe_control": "speed_limited_kinematic_mouse_target",
                "repeatability_scope": "interactive_manual_not_deterministic",
                "manual_safety": {
                    "candidate_state_acceptance": "transactional_alternate_buffer",
                    "circuit_breaker_minimum_relative_j": self.config["monitoring"][
                        "minimum_relative_tet_volume"
                    ],
                    "stop_minimum_relative_j": self.safety_config[
                        "stop_minimum_relative_tet_volume"
                    ],
                    "warning_minimum_relative_j": self.safety_config[
                        "warning_minimum_relative_tet_volume"
                    ],
                    "maximum_commanded_indentation_m": self.maximum_commanded_indentation_m,
                },
                "world_force_field": "estimated_world_reaction_n",
            }
        )
        self._write_environment()
        self._configure_interactive_viewer()
        print(
            "[interactive] Force values use the VBD penalty reconstruction and "
            "are estimates, not calibrated ground truth."
        )
        print(
            "[interactive] Right-drag: move probe | Shift+right-drag: rotate | "
            "Space: pause | R: reset | B: baseline | C: save | Esc: exit"
        )

    def _add_indenter_shape(self, builder, body: int, shape_cfg) -> None:
        probe = self.config["probe"]
        kind = probe["type"]
        if kind in {"rounded_block", "custom_rigid_stl"}:
            mesh = newton.Mesh.create_from_file(probe["mesh"])
            scale = float(probe["scale_to_m"])
            self.probe_local_vertices_m = (
                np.asarray(mesh.vertices, dtype=np.float64) * scale
            )
            self.probe_shape = builder.add_shape_mesh(
                body,
                mesh=mesh,
                scale=wp.vec3(scale, scale, scale),
                cfg=shape_cfg,
            )
        elif kind == "capsule":
            self.probe_shape = builder.add_shape_capsule(
                body,
                radius=float(probe["radius_m"]),
                half_height=0.5 * float(probe["height_m"]),
                cfg=shape_cfg,
            )
        elif kind == "cylinder":
            self.probe_shape = builder.add_shape_cylinder(
                body,
                radius=float(probe["radius_m"]),
                half_height=0.5 * float(probe["height_m"]),
                cfg=shape_cfg,
            )
        elif kind == "sphere":
            self.probe_shape = builder.add_shape_sphere(
                body, radius=float(probe["radius_m"]), cfg=shape_cfg
            )
        else:
            raise ValueError(f"unsupported interactive probe type {kind!r}")

    def _commanded_indentation_for_pose(self, pose: ProbePose) -> float:
        return commanded_indentation(
            pose,
            probe=self.config["probe"],
            contact_location_m=self.contact_reference_location_m,
            contact_direction=self.loading_direction,
            mesh_vertices_m=self.probe_local_vertices_m,
        )

    def _current_probe_pose(self) -> ProbePose:
        return ProbePose(
            self.current_body_position,
            self.indenter_quaternion,
        )

    def _select_linear_speed(self) -> float:
        if self.last_contact_flag:
            self.speed_regime = "contact"
            key = "contact_linear_speed_m_s"
        elif self.current_commanded_indentation_m >= -float(
            self.manual_config["near_contact_distance_m"]
        ):
            self.speed_regime = "near_contact"
            key = "near_contact_linear_speed_m_s"
        else:
            self.speed_regime = "free_space"
            key = "free_space_linear_speed_m_s"
        return float(self.manual_config[key])

    def _configure_interactive_viewer(self) -> None:
        if self.viewer is None:
            return
        self.viewer.show_contacts = bool(self.display_config["show_contacts"])
        if hasattr(self.viewer, "picking") and self.viewer.picking is not None:
            picking = KinematicProbePicking(
                self.model,
                self.indenter_body,
                world_offsets=self.viewer.world_offsets,
            )
            picking.visible_worlds_mask = self.viewer._visible_worlds_mask
            self.viewer.picking = picking
            self.viewer.picking_enabled = False
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            renderer.register_key_press(self._on_key_press)
            renderer.register_mouse_drag(self._on_mouse_drag)

    def _set_probe_pose(
        self,
        state,
        position_m: np.ndarray,
        quaternion_xyzw: np.ndarray,
        linear_velocity_m_s: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
    ) -> None:
        wp.launch(
            _set_kinematic_pose_v2,
            dim=1,
            inputs=[
                self.indenter_body,
                *position_m,
                *quaternion_xyzw,
                *linear_velocity_m_s,
                *angular_velocity_rad_s,
            ],
            outputs=[state.body_q, state.body_qd],
            device=self.model.device,
        )

    def _set_probe_pose_on_states(
        self,
        position_m: np.ndarray,
        quaternion_xyzw: np.ndarray,
        linear_velocity_m_s: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
    ) -> None:
        self._set_probe_pose(
            self.state_0,
            position_m,
            quaternion_xyzw,
            linear_velocity_m_s,
            angular_velocity_rad_s,
        )
        self._set_probe_pose(
            self.state_1,
            position_m,
            quaternion_xyzw,
            linear_velocity_m_s,
            angular_velocity_rad_s,
        )
        wp.synchronize()

    def _consume_picker_target(self) -> None:
        if self.viewer is None:
            return
        picking = getattr(self.viewer, "picking", None)
        if (
            picking is None
            or not picking.is_picking()
            or int(picking.pick_body.numpy()[0]) != self.indenter_body
        ):
            self.rotation_input_pending = False
            return
        state = picking.pick_state.numpy()
        if not self.rotation_input_pending:
            local_point = np.asarray(state[0]["picked_point_local"], dtype=np.float64)
            target_point = np.asarray(
                state[0]["picking_target_world"], dtype=np.float64
            )
            rotated_local = quaternion_rotate_xyzw(
                self.mouse_target_pose.quaternion_xyzw, local_point
            )
            self.mouse_target_pose = ProbePose(
                target_point - rotated_local,
                self.mouse_target_pose.quaternion_xyzw,
            )
        self.rotation_input_pending = False

    def _sync_picker_point(self, *, reset_target: bool = False) -> None:
        if self.viewer is None:
            return
        picking = getattr(self.viewer, "picking", None)
        if (
            picking is None
            or not picking.is_picking()
            or int(picking.pick_body.numpy()[0]) != self.indenter_body
        ):
            return
        state = picking.pick_state.numpy()
        local_point = np.asarray(state[0]["picked_point_local"], dtype=np.float64)
        picked_world = self.current_body_position + quaternion_rotate_xyzw(
            self.indenter_quaternion, local_point
        )
        state[0]["picked_point_world"] = picked_world
        if reset_target:
            state[0]["picking_target_world"] = picked_world
        picking.pick_state.assign(state)

    def _on_mouse_drag(
        self,
        _x: float,
        _y: float,
        dx: float,
        dy: float,
        _buttons: int,
        modifiers: int,
    ) -> None:
        if self.viewer is None or self.equilibrated_particle_positions is None:
            return
        picking = getattr(self.viewer, "picking", None)
        if (
            picking is None
            or not picking.is_picking()
            or int(picking.pick_body.numpy()[0]) != self.indenter_body
        ):
            return
        try:
            import pyglet
        except ImportError:
            return
        if not modifiers & pyglet.window.key.MOD_SHIFT:
            return

        sensitivity = math.radians(
            float(self.manual_config["rotation_sensitivity_deg_per_pixel"])
        )
        camera_up = np.asarray(tuple(self.viewer.camera.get_up()), dtype=np.float64)
        camera_right = np.asarray(
            tuple(self.viewer.camera.get_right()), dtype=np.float64
        )
        yaw = quaternion_from_axis_angle_xyzw(camera_up, -float(dx) * sensitivity)
        pitch = quaternion_from_axis_angle_xyzw(camera_right, float(dy) * sensitivity)
        target_quaternion = quaternion_multiply_xyzw(
            pitch,
            quaternion_multiply_xyzw(yaw, self.mouse_target_pose.quaternion_xyzw),
        )
        self.mouse_target_pose = ProbePose(
            self.mouse_target_pose.position_m,
            target_quaternion,
        )
        self.rotation_input_pending = True
        pick_state = picking.pick_state.numpy()
        pick_state[0]["picking_target_world"] = pick_state[0]["picked_point_world"]
        picking.pick_state.assign(pick_state)

    def _key_matches(self, symbol: int, configured: str) -> bool:
        try:
            import pyglet
        except ImportError:
            return False
        name = configured.strip().upper()
        return symbol == getattr(pyglet.window.key, name, -1)

    def _on_key_press(self, symbol: int, _modifiers: int) -> None:
        if self._key_matches(symbol, self.manual_config["reset_key"]):
            self.pending_reset = True
        elif self._key_matches(symbol, self.manual_config["baseline_key"]):
            self.pending_baseline = True
        elif self._key_matches(symbol, self.manual_config["capture_key"]):
            self.pending_capture = True

    def _evaluate_candidate_state(
        self, pose: ProbePose
    ) -> tuple[CandidateSafety, Any, np.ndarray, np.ndarray]:
        candidate_global = self.state_1.particle_q.numpy().astype(np.float64)
        previous_global = self.state_0.particle_q.numpy().astype(np.float64)
        particles = candidate_global[self.particle_start : self.particle_end]
        relative = self._relative_tet_volumes(particles)
        contact = self._contact_summary(candidate_global, previous_global)
        force_magnitude = float(np.linalg.norm(contact.estimated_world_reaction_n))
        indentation = self._commanded_indentation_for_pose(pose)
        evaluation = evaluate_candidate_safety(
            relative_j=relative,
            estimated_force_magnitude_n=force_magnitude,
            commanded_indentation_m=indentation,
            circuit_breaker_minimum_j=float(
                self.config["monitoring"]["minimum_relative_tet_volume"]
            ),
            stop_minimum_j=float(
                self.safety_config["stop_minimum_relative_tet_volume"]
            ),
            warning_minimum_j=float(
                self.safety_config["warning_minimum_relative_tet_volume"]
            ),
            stop_estimated_force_n=float(self.safety_config["stop_estimated_force_n"]),
            warning_estimated_force_n=float(
                self.safety_config["warning_estimated_force_n"]
            ),
            maximum_commanded_indentation_m=self.maximum_commanded_indentation_m,
            warning_commanded_indentation_m=float(
                self.safety_config["warning_commanded_indentation_m"]
            ),
        )
        self.minimum_relative_j_seen = min(
            self.minimum_relative_j_seen, evaluation.minimum_relative_j
        )
        return evaluation, contact, particles, relative

    def _export_safety_event(
        self,
        path: Path,
        *,
        event_type: str,
        reason: str,
        evaluation: CandidateSafety,
        particles: np.ndarray,
        relative: np.ndarray,
        candidate_pose: ProbePose,
    ) -> None:
        export_failure_state(
            path,
            safety_event_type=np.asarray(event_type),
            safety_reason=np.asarray(reason),
            minimum_relative_j=np.float64(evaluation.minimum_relative_j),
            relative_j=relative,
            affected_tet_indices=evaluation.affected_tet_indices,
            candidate_particle_q=particles,
            current_tet_volumes=relative * self.rest_volumes,
            rest_tet_volumes=self.rest_volumes,
            probe_position_m=candidate_pose.position_m,
            probe_quaternion_xyzw=candidate_pose.quaternion_xyzw,
            last_safe_probe_position_m=self.last_safe_probe_pose.position_m,
            last_safe_probe_quaternion_xyzw=(self.last_safe_probe_pose.quaternion_xyzw),
            commanded_indentation_m=np.float64(evaluation.commanded_indentation_m),
            estimated_force_magnitude_n=np.float64(
                evaluation.estimated_force_magnitude_n
            ),
            circuit_breaker_minimum_j=np.float64(
                self.config["monitoring"]["minimum_relative_tet_volume"]
            ),
            stop_minimum_j=np.float64(
                self.safety_config["stop_minimum_relative_tet_volume"]
            ),
            substep=np.int64(self.substep),
            sim_time_s=np.float64(self.sim_time),
            trajectory_phase=np.asarray("interactive_manual"),
            newton_version=np.asarray(self.newton_version),
            newton_git_sha=np.asarray(self.newton_revision),
        )

    def _update_safety_visualization(self, affected_tets: np.ndarray) -> None:
        particles = self._particle_positions()
        device = self.model.device
        valid = np.asarray(affected_tets, dtype=np.int32)
        valid = valid[(valid >= 0) & (valid < len(self.mapping_tets))]
        if len(valid) and bool(self.display_config["show_safety_tets"]):
            tet_vertices = self.mapping_tets[valid]
            edge_pairs = np.asarray(
                [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                dtype=np.int32,
            )
            starts = particles[tet_vertices[:, edge_pairs[:, 0]].reshape(-1)]
            ends = particles[tet_vertices[:, edge_pairs[:, 1]].reshape(-1)]
            self.safety_tet_line_starts = wp.array(
                starts.astype(np.float32), dtype=wp.vec3, device=device
            )
            self.safety_tet_line_ends = wp.array(
                ends.astype(np.float32), dtype=wp.vec3, device=device
            )
            self.safety_tet_line_colors = wp.full(
                len(starts),
                wp.vec3(1.0, 0.05, 0.05),
                dtype=wp.vec3,
                device=device,
            )
        else:
            self.safety_tet_line_starts = None
            self.safety_tet_line_ends = None
            self.safety_tet_line_colors = None

        if bool(self.display_config["show_mount_vertices"]):
            mount_points = particles[self.mount_vertices]
            self.mount_visualization_points = wp.array(
                mount_points.astype(np.float32), dtype=wp.vec3, device=device
            )
            self.mount_visualization_radii = wp.full(
                len(mount_points),
                max(0.16 * self.particle_radius, 4.0e-5),
                dtype=wp.float32,
                device=device,
            )
            self.mount_visualization_colors = wp.full(
                len(mount_points),
                wp.vec3(1.0, 0.8, 0.05),
                dtype=wp.vec3,
                device=device,
            )

    def _fatal_candidate(
        self,
        evaluation: CandidateSafety,
        particles: np.ndarray,
        relative: np.ndarray,
        candidate_pose: ProbePose,
    ) -> None:
        reason = evaluation.fatal_reason or "fatal_interactive_safety_error"
        self._export_safety_event(
            self.failure_path,
            event_type="fatal_circuit_breaker",
            reason=reason,
            evaluation=evaluation,
            particles=particles,
            relative=relative,
            candidate_pose=candidate_pose,
        )
        self.fatal_error_message = (
            f"{reason}: minimum relative tet volume {evaluation.minimum_relative_j:.6f}"
        )
        self.finalize()
        raise FloatingPointError(
            f"{self.fatal_error_message}; state saved to {self.failure_path}"
        )

    def _reject_candidate(
        self,
        evaluation: CandidateSafety,
        particles: np.ndarray,
        relative: np.ndarray,
        candidate_pose: ProbePose,
    ) -> None:
        reason = evaluation.stop_reason or "interactive_safety_stop"
        destination = (
            self.exporter.output_dir
            / f"safety_stop_{self.safety_snapshot_index:05d}.npz"
        )
        self._export_safety_event(
            destination,
            event_type="recoverable_safety_stop",
            reason=reason,
            evaluation=evaluation,
            particles=particles,
            relative=relative,
            candidate_pose=candidate_pose,
        )
        self.safety_snapshot_index += 1
        self.safety_stop_count += 1
        self.last_safety_export_path = destination
        self.safety_stop_active = True
        self.safety_stop_reason = reason
        self.safety_stop_indentation_m = evaluation.commanded_indentation_m
        self.safety_warning_reasons = evaluation.warning_reasons
        self.safety_affected_tet_indices = evaluation.affected_tet_indices.copy()
        self.last_safety_evaluation = evaluation
        self.mouse_target_pose = self.last_safe_probe_pose.copy()

        self.current_body_position = self.last_safe_probe_pose.position_m.copy()
        self.indenter_quaternion = self.last_safe_probe_pose.quaternion_xyzw.copy()
        self.current_body_velocity = np.zeros(3, dtype=np.float64)
        self.current_body_angular_velocity = np.zeros(3, dtype=np.float64)
        self.current_commanded_indentation_m = self._commanded_indentation_for_pose(
            self.last_safe_probe_pose
        )
        self._set_probe_pose(
            self.state_0,
            self.current_body_position,
            self.indenter_quaternion,
            self.current_body_velocity,
            self.current_body_angular_velocity,
        )
        self.state_1.assign(self.state_0)
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self._sync_picker_point(reset_target=True)
        self._update_safety_visualization(self.safety_affected_tet_indices)
        self.environment["manual_safety"]["last_stop"] = {
            "reason": reason,
            "minimum_relative_j": evaluation.minimum_relative_j,
            "commanded_indentation_m": evaluation.commanded_indentation_m,
            "affected_tet_indices": evaluation.affected_tet_indices.tolist(),
            "file": str(destination),
        }
        self._write_environment()
        print(
            "[interactive][SAFETY STOP] "
            f"reason={reason} minJ={evaluation.minimum_relative_j:.4f} "
            f"command={evaluation.commanded_indentation_m * 1000.0:.3f} mm; "
            "retract the probe or press R"
        )

    def _accept_candidate(
        self,
        evaluation: CandidateSafety,
        contact: Any,
        candidate_pose: ProbePose,
        linear_velocity_m_s: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
    ) -> None:
        previous_warnings = self.safety_warning_reasons
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.current_body_position = candidate_pose.position_m.copy()
        self.indenter_quaternion = candidate_pose.quaternion_xyzw.copy()
        self.current_body_velocity = np.asarray(
            linear_velocity_m_s, dtype=np.float64
        ).copy()
        self.current_body_angular_velocity = np.asarray(
            angular_velocity_rad_s, dtype=np.float64
        ).copy()
        self.current_commanded_indentation_m = evaluation.commanded_indentation_m
        self.last_safe_probe_pose = candidate_pose.copy()
        self.last_contact_flag = bool(contact.contact_flag)
        self.last_safety_evaluation = evaluation
        self.safety_warning_reasons = evaluation.warning_reasons
        self._monitor_contact_buffer()

        if self.safety_stop_active:
            retraction = (
                self.safety_stop_indentation_m - self.current_commanded_indentation_m
            )
            recovered_j = evaluation.minimum_relative_j >= float(
                self.safety_config["stop_minimum_relative_tet_volume"]
            )
            if (
                retraction >= float(self.safety_config["retraction_clearance_m"])
                and recovered_j
            ):
                self.safety_stop_active = False
                self.safety_stop_reason = ""
                print("[interactive] safety stop cleared; inward motion re-enabled")

        if self.safety_stop_active:
            affected = self.safety_affected_tet_indices
        elif evaluation.warning_reasons:
            affected = evaluation.affected_tet_indices
            self.safety_affected_tet_indices = affected.copy()
            if evaluation.warning_reasons != previous_warnings:
                self._update_safety_visualization(affected)
        elif previous_warnings:
            self.safety_affected_tet_indices = np.empty(0, dtype=np.int32)
            self._update_safety_visualization(self.safety_affected_tet_indices)

        if (
            evaluation.warning_reasons
            and evaluation.warning_reasons != previous_warnings
        ):
            print(
                "[interactive][SAFETY WARNING] "
                f"reasons={','.join(evaluation.warning_reasons)} "
                f"minJ={evaluation.minimum_relative_j:.4f} "
                f"command={evaluation.commanded_indentation_m * 1000.0:.3f} mm "
                f"|F_est|={evaluation.estimated_force_magnitude_n:.4f} N"
            )

    def simulate(self, *, advance_trajectory: bool) -> None:
        del advance_trajectory
        if self.equilibrated_particle_positions is not None:
            self._consume_picker_target()

        target_pose = self.mouse_target_pose
        if self.safety_stop_active:
            target_indentation = self._commanded_indentation_for_pose(target_pose)
            if target_indentation >= self.current_commanded_indentation_m - 1.0e-12:
                self.mouse_target_pose = self.last_safe_probe_pose.copy()
                target_pose = self.mouse_target_pose
                self._sync_picker_point(reset_target=True)

        limited = limited_transform_step(
            current_position_m=self.current_body_position,
            current_quaternion_xyzw=self.indenter_quaternion,
            target_position_m=target_pose.position_m,
            target_quaternion_xyzw=target_pose.quaternion_xyzw,
            dt_s=self.frame_dt,
            max_linear_speed_m_s=self._select_linear_speed(),
            max_angular_speed_deg_s=float(
                self.manual_config["max_angular_speed_deg_s"]
            ),
            max_translation_per_frame_m=float(
                self.manual_config["max_translation_per_frame_m"]
            ),
            max_rotation_per_frame_deg=float(
                self.manual_config["max_rotation_per_frame_deg"]
            ),
        )
        start_position = self.current_body_position.copy()
        start_quaternion = self.indenter_quaternion.copy()
        total_rotation = _quaternion_delta_xyzw(
            start_quaternion, limited.quaternion_xyzw
        )[1]
        movement_rejected = False
        for substep_index in range(self.sim_substeps):
            if movement_rejected:
                candidate_pose = self.last_safe_probe_pose.copy()
                linear_velocity = np.zeros(3, dtype=np.float64)
                angular_velocity = np.zeros(3, dtype=np.float64)
            else:
                alpha = (substep_index + 1) / self.sim_substeps
                position = start_position + alpha * (
                    limited.position_m - start_position
                )
                incremental = limited_transform_step(
                    current_position_m=start_position,
                    current_quaternion_xyzw=start_quaternion,
                    target_position_m=limited.position_m,
                    target_quaternion_xyzw=limited.quaternion_xyzw,
                    dt_s=1.0,
                    max_linear_speed_m_s=1.0e9,
                    max_angular_speed_deg_s=1.0e9,
                    max_translation_per_frame_m=1.0e9,
                    max_rotation_per_frame_deg=math.degrees(alpha * total_rotation),
                )
                candidate_pose = ProbePose(
                    position,
                    incremental.quaternion_xyzw,
                )
                linear_velocity = limited.linear_velocity_m_s
                angular_velocity = limited.angular_velocity_rad_s

            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self._set_probe_pose(
                self.state_0,
                candidate_pose.position_m,
                candidate_pose.quaternion_xyzw,
                linear_velocity,
                angular_velocity,
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.substep += 1
            evaluation, contact, particles, relative = self._evaluate_candidate_state(
                candidate_pose
            )
            if evaluation.fatal:
                self._fatal_candidate(
                    evaluation,
                    particles,
                    relative,
                    candidate_pose,
                )
            if evaluation.stopped:
                self._reject_candidate(
                    evaluation,
                    particles,
                    relative,
                    candidate_pose,
                )
                movement_rejected = True
                continue
            self._accept_candidate(
                evaluation,
                contact,
                candidate_pose,
                linear_velocity,
                angular_velocity,
            )
        self._sync_picker_point()

    def _phase_name(self) -> str:
        if self.lifecycle_phase in {
            "initialization",
            "settling",
            "capture_baseline",
        }:
            return self.lifecycle_phase
        return "interactive_manual"

    def _check_tet_state(self) -> None:
        particles = self._particle_positions()
        relative = self._relative_tet_volumes(particles)
        threshold = float(self.config["monitoring"]["minimum_relative_tet_volume"])
        bad = np.flatnonzero(~np.isfinite(relative) | (relative < threshold))
        if not len(bad):
            return
        export_failure_state(
            self.failure_path,
            particle_q=particles,
            current_tet_volumes=relative * self.rest_volumes,
            rest_tet_volumes=self.rest_volumes,
            relative_j=relative,
            bad_tet_indices=bad,
            relative_j_threshold=np.float64(threshold),
            substep=np.int64(self.substep),
            sim_time_s=np.float64(self.sim_time),
            trajectory_phase=np.asarray("interactive_manual"),
            probe_position_m=self.current_body_position,
            probe_quaternion_xyzw=self.indenter_quaternion,
            newton_version=np.asarray(self.newton_version),
            newton_git_sha=np.asarray(self.newton_revision),
        )
        self.finalize()
        raise FloatingPointError(
            f"relative tet volume crossed {threshold:g}; "
            f"state saved to {self.failure_path}"
        )

    def _capture_equilibrated_baseline(self) -> None:
        super()._capture_equilibrated_baseline()
        self.lifecycle_phase = "interactive_manual"
        if self.viewer is not None:
            self.viewer.picking_enabled = True
        self._update_safety_visualization(np.empty(0, dtype=np.int32))
        self._update_live_metrics(force=True)
        print("[interactive] settling complete; probe manipulation enabled")

    def _reset_manual_state(self) -> None:
        if self.equilibrated_particle_positions is None:
            return
        for state in (self.state_0, self.state_1):
            positions = state.particle_q.numpy()
            positions[self.particle_start : self.particle_end] = (
                self.equilibrated_particle_positions
            )
            state.particle_q.assign(positions)
            state.particle_qd.zero_()
            state.clear_forces()
        self.current_body_position = self.probe_initial_position.copy()
        self.indenter_quaternion = self.probe_initial_quaternion.copy()
        self.current_body_velocity = np.zeros(3, dtype=np.float64)
        self.current_body_angular_velocity = np.zeros(3, dtype=np.float64)
        initial_pose = ProbePose(
            self.probe_initial_position,
            self.probe_initial_quaternion,
        )
        self.mouse_target_pose = initial_pose.copy()
        self.last_safe_probe_pose = initial_pose.copy()
        self.current_commanded_indentation_m = self._commanded_indentation_for_pose(
            initial_pose
        )
        self.last_contact_flag = False
        self.speed_regime = "free_space"
        self.safety_stop_active = False
        self.safety_stop_reason = ""
        self.safety_warning_reasons = ()
        self.safety_affected_tet_indices = np.empty(0, dtype=np.int32)
        self.last_safety_evaluation = None
        self._set_probe_pose_on_states(
            self.current_body_position,
            self.indenter_quaternion,
            self.current_body_velocity,
            self.current_body_angular_velocity,
        )
        if self.viewer is not None and getattr(self.viewer, "picking", None):
            self.viewer.picking.release()
        self.current_contact_buffer_count = 0
        self._update_safety_visualization(np.empty(0, dtype=np.int32))
        self._update_live_metrics(force=True)
        print("[interactive] probe and fingertip reset to settled baseline")

    def _capture_new_baseline(self) -> None:
        if self.equilibrated_particle_positions is None:
            return
        particles = self._particle_positions()
        self.equilibrated_particle_positions = particles.copy()
        self.equilibrated_surface_positions = reconstruct_surface(
            particles, self.mapping_tets, self.surface_mapping
        )
        self.environment["manual_baseline_capture_time_s"] = self.sim_time
        self._write_environment()
        self._update_live_metrics(force=True)
        print("[interactive] captured new deformation baseline")

    def _contact_geometry(
        self,
    ) -> tuple[Any, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        current_global = self.state_0.particle_q.numpy().astype(np.float64)
        previous_global = self.state_1.particle_q.numpy().astype(np.float64)
        particles = current_global[self.particle_start : self.particle_end]
        deformed_surface = reconstruct_surface(
            particles, self.mapping_tets, self.surface_mapping
        )
        contact = self._contact_summary(current_global, previous_global)
        active_local = contact.active_particle_indices - self.particle_start
        active_local = active_local[
            (active_local >= 0)
            & (active_local < self.particle_end - self.particle_start)
        ]
        face_mask = contact_face_mask(
            deformed_surface,
            self.surface_faces,
            self.outer_faces,
            particles[active_local],
            distance_m=float(
                self.config["contact_parameters"]["face_mask_radius_multiplier"]
            )
            * self.particle_radius,
        )
        area = masked_triangle_area(
            deformed_surface,
            self.surface_faces,
            self.outer_faces,
            face_mask,
        )
        return (
            contact,
            particles,
            deformed_surface,
            area,
            face_mask,
            particles[active_local],
        )

    def _update_live_metrics(self, *, force: bool = False) -> None:
        if self.equilibrated_particle_positions is None:
            return
        if not force and self.sim_time + 1.0e-12 < self.next_metrics_time_s:
            return
        period = 1.0 / float(self.display_config["metrics_rate_hz"])
        self.next_metrics_time_s = self.sim_time + period
        (
            contact,
            particles,
            deformed_surface,
            approx_area,
            face_mask,
            active_points,
        ) = self._contact_geometry()
        displacement = particles - self.equilibrated_particle_positions
        max_displacement = float(np.max(np.linalg.norm(displacement, axis=1)))
        relative = self._relative_tet_volumes(particles)
        minimum_relative = float(np.nanmin(relative))
        world_force = np.asarray(contact.estimated_world_reaction_n, dtype=np.float64)
        force_magnitude = float(np.linalg.norm(world_force))
        orientation_wxyz = np.asarray(
            [
                self.indenter_quaternion[3],
                self.indenter_quaternion[0],
                self.indenter_quaternion[1],
                self.indenter_quaternion[2],
            ]
        )
        self.live_metrics = {
            "contact": bool(contact.contact_flag),
            "maximum_displacement_m": max_displacement,
            "estimated_force_magnitude_n": force_magnitude,
            "estimated_force_vector_n": world_force,
            "approx_contact_area_m2": approx_area,
            "probe_position_m": self.current_body_position.copy(),
            "probe_orientation_wxyz": orientation_wxyz,
            "minimum_relative_tet_volume": minimum_relative,
            "active_contact_count": int(len(contact.active_contact_indices)),
            "contact_buffer_observed_count": self.current_contact_buffer_count,
            "contact_buffer_capacity": self.contact_buffer_capacity,
            "commanded_indentation_m": self.current_commanded_indentation_m,
            "maximum_commanded_indentation_m": self.maximum_commanded_indentation_m,
            "speed_regime": self.speed_regime,
            "safety_stop_active": self.safety_stop_active,
            "safety_stop_reason": self.safety_stop_reason,
            "safety_warning_reasons": self.safety_warning_reasons,
            "affected_tet_indices": self.safety_affected_tet_indices.copy(),
            "face_mask": face_mask,
        }
        device = self.model.device
        if len(active_points):
            self.live_contact_points = wp.array(
                active_points.astype(np.float32), dtype=wp.vec3, device=device
            )
            self.live_contact_radii = wp.full(
                len(active_points),
                max(0.2 * self.particle_radius, 5.0e-5),
                dtype=wp.float32,
                device=device,
            )
            self.live_contact_colors = wp.full(
                len(active_points),
                wp.vec3(1.0, 0.1, 0.1),
                dtype=wp.vec3,
                device=device,
            )
        else:
            self.live_contact_points = None
            self.live_contact_radii = None
            self.live_contact_colors = None
        if bool(self.display_config["displacement_heatmap"]):
            surface_displacement = np.linalg.norm(
                deformed_surface - self.equilibrated_surface_positions, axis=1
            )
            limit = float(self.display_config["heatmap_max_displacement_m"])
            ratio = np.clip(surface_displacement / limit, 0.0, 1.0)
            low = np.asarray([0.70, 0.60, 0.40])
            high = np.asarray([1.00, 0.05, 0.05])
            colors = low + ratio[:, None] * (high - low)
            self.heatmap_points = wp.array(
                deformed_surface.astype(np.float32),
                dtype=wp.vec3,
                device=device,
            )
            self.heatmap_radii = wp.full(
                len(deformed_surface),
                max(0.08 * self.particle_radius, 2.0e-5),
                dtype=wp.float32,
                device=device,
            )
            self.heatmap_colors = wp.array(
                colors.astype(np.float32), dtype=wp.vec3, device=device
            )

        if self.viewer is not None:
            self.viewer.log_scalar(
                "interactive/max displacement (mm)",
                max_displacement * 1000.0,
            )
            self.viewer.log_scalar(
                "interactive/estimated force magnitude (N)", force_magnitude
            )
            self.viewer.log_scalar(
                "interactive/contact area (mm^2)", approx_area * 1.0e6
            )
            self.viewer.log_scalar(
                "interactive/min relative tet volume", minimum_relative
            )
            self.viewer.log_scalar(
                "interactive/commanded indentation (mm)",
                self.current_commanded_indentation_m * 1000.0,
            )

        if bool(self.display_config["show_metrics"]):
            position = self.current_body_position
            print(
                "[interactive] "
                f"contact={str(contact.contact_flag).lower()} "
                f"disp={max_displacement * 1000.0:.3f} mm "
                f"|F_est|={force_magnitude:.4f} N "
                f"F_est=({world_force[0]:.4f}, {world_force[1]:.4f}, "
                f"{world_force[2]:.4f}) N "
                f"area={approx_area * 1.0e6:.3f} mm^2 "
                f"probe=({position[0]:.4f}, {position[1]:.4f}, "
                f"{position[2]:.4f}) m "
                f"minJ={minimum_relative:.4f} "
                f"command={self.current_commanded_indentation_m * 1000.0:.3f} mm "
                f"speed={self.speed_regime} "
                f"safety={'STOP' if self.safety_stop_active else 'run'} "
                f"active={len(contact.active_contact_indices)} "
                f"buffer={self.current_contact_buffer_count}/"
                f"{self.contact_buffer_capacity}"
            )

    def _save_current_state(self) -> Path | None:
        if self.equilibrated_particle_positions is None:
            return None
        (
            contact,
            particles,
            deformed_surface,
            approx_area,
            face_mask,
            _active_points,
        ) = self._contact_geometry()
        relative = self._relative_tet_volumes(particles)
        destination = (
            self.exporter.output_dir
            / f"interactive_state_{self.snapshot_index:05d}.npz"
        )
        np.savez_compressed(
            destination,
            timestamp_s=np.float64(self.sim_time),
            tet_particle_positions_m=particles,
            deformed_surface_vertices_m=deformed_surface,
            deformed_inner_coating_vertices_m=deformed_surface[self.inner_vertices],
            deformed_outer_surface_vertices_m=deformed_surface[self.outer_vertices],
            displacement_from_equilibrated_baseline_m=(
                particles - self.equilibrated_particle_positions
            ),
            probe_position_m=self.current_body_position,
            probe_quaternion_xyzw=self.indenter_quaternion,
            probe_linear_velocity_m_s=self.current_body_velocity,
            probe_angular_velocity_rad_s=self.current_body_angular_velocity,
            contact_flag=np.bool_(contact.contact_flag),
            contact_face_mask=face_mask,
            approx_contact_area_m2=np.float64(approx_area),
            estimated_world_reaction_n=contact.estimated_world_reaction_n,
            estimated_force_magnitude_n=np.float64(
                np.linalg.norm(contact.estimated_world_reaction_n)
            ),
            minimum_relative_tet_volume=np.float64(np.nanmin(relative)),
            contact_buffer_observed_count=np.int32(self.current_contact_buffer_count),
            contact_buffer_capacity=np.int32(self.contact_buffer_capacity),
            commanded_indentation_m=np.float64(self.current_commanded_indentation_m),
            safety_stop_active=np.bool_(self.safety_stop_active),
            safety_reason=np.asarray(self.safety_stop_reason),
            safety_warning_reasons=np.asarray(self.safety_warning_reasons),
            affected_tet_indices=self.safety_affected_tet_indices,
            last_safety_export_path=np.asarray(str(self.last_safety_export_path or "")),
        )
        self.snapshot_index += 1
        print(f"[interactive] saved state: {destination}")
        return destination

    def _process_pending_actions(self) -> None:
        if self.pending_reset:
            self.pending_reset = False
            self._reset_manual_state()
        if self.pending_baseline:
            self.pending_baseline = False
            self._capture_new_baseline()
        if self.pending_capture:
            self.pending_capture = False
            self._save_current_state()

    def step(self) -> None:
        self._process_pending_actions()
        if self.equilibrated_particle_positions is None:
            self._step_settling()
            return
        self.lifecycle_phase = "interactive_manual"
        self.simulate(advance_trajectory=False)
        self.sim_time += self.frame_dt
        self._update_live_metrics()

    def gui(self, imgui) -> None:
        metrics = self.live_metrics
        imgui.separator()
        imgui.text("Interactive Touch")
        imgui.text("Right drag: move | Shift + right drag: rotate")
        imgui.text("Space: pause | R: reset | B: baseline | C: save")
        imgui.text("Estimated forces are not calibrated.")
        if not metrics:
            imgui.text(f"Phase: {self._phase_name()}")
            return
        if metrics["safety_stop_active"]:
            imgui.separator()
            imgui.text("*** RECOVERABLE SAFETY STOP ***")
            imgui.text(f"Reason: {metrics['safety_stop_reason']}")
            imgui.text("Further inward motion is blocked.")
            imgui.text("Retract the probe or press R to reset.")
            imgui.separator()
        elif metrics["safety_warning_reasons"]:
            imgui.separator()
            imgui.text(
                "SAFETY WARNING: " + ", ".join(metrics["safety_warning_reasons"])
            )
        force = metrics["estimated_force_vector_n"]
        position = metrics["probe_position_m"]
        orientation = metrics["probe_orientation_wxyz"]
        imgui.text(f"Contact: {metrics['contact']}")
        imgui.text(
            f"Max displacement: {metrics['maximum_displacement_m'] * 1000.0:.3f} mm"
        )
        imgui.text(f"Estimated |force|: {metrics['estimated_force_magnitude_n']:.4f} N")
        imgui.text(
            f"Estimated force: ({force[0]:.4f}, {force[1]:.4f}, {force[2]:.4f}) N"
        )
        imgui.text(
            f"Approx. area: {metrics['approx_contact_area_m2'] * 1.0e6:.3f} mm^2"
        )
        imgui.text(
            f"Probe position: ({position[0]:.4f}, {position[1]:.4f}, "
            f"{position[2]:.4f}) m"
        )
        imgui.text(
            f"Probe orientation wxyz: ({orientation[0]:.3f}, "
            f"{orientation[1]:.3f}, {orientation[2]:.3f}, "
            f"{orientation[3]:.3f})"
        )
        imgui.text(
            f"Min relative tet volume: {metrics['minimum_relative_tet_volume']:.4f}"
        )
        imgui.text(
            "Commanded indentation: "
            f"{metrics['commanded_indentation_m'] * 1000.0:.3f}/"
            f"{metrics['maximum_commanded_indentation_m'] * 1000.0:.3f} mm"
        )
        imgui.text(f"Speed regime: {metrics['speed_regime']}")
        imgui.text(f"Affected tets: {len(metrics['affected_tet_indices'])}")
        imgui.text(f"Active contacts: {metrics['active_contact_count']}")
        imgui.text(
            "Contact buffer: "
            f"{metrics['contact_buffer_observed_count']}/"
            f"{metrics['contact_buffer_capacity']}"
        )

    def render(self) -> None:
        self._process_pending_actions()
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if bool(self.display_config["show_contacts"]):
            self.viewer.log_contacts(self.contacts, self.state_0)
            self.viewer.log_points(
                "/interactive/soft_contact_points",
                self.live_contact_points,
                radii=self.live_contact_radii,
                colors=self.live_contact_colors,
                hidden=self.live_contact_points is None,
            )
        if bool(self.display_config["displacement_heatmap"]):
            self.viewer.log_points(
                "/interactive/displacement_heatmap",
                self.heatmap_points,
                radii=self.heatmap_radii,
                colors=self.heatmap_colors,
                hidden=self.heatmap_points is None,
            )
        if bool(self.display_config["show_safety_tets"]):
            self.viewer.log_lines(
                "/interactive/safety_affected_tets",
                self.safety_tet_line_starts,
                self.safety_tet_line_ends,
                colors=self.safety_tet_line_colors,
                width=max(0.12 * self.particle_radius, 3.0e-5),
                hidden=self.safety_tet_line_starts is None,
            )
        if bool(self.display_config["show_mount_vertices"]):
            self.viewer.log_points(
                "/interactive/fixed_mount_vertices",
                self.mount_visualization_points,
                radii=self.mount_visualization_radii,
                colors=self.mount_visualization_colors,
                hidden=self.mount_visualization_points is None,
            )
        self.viewer.end_frame()
