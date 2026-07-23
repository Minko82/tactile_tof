"""Shared GPU-only Newton frame stepping and transactional state acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import warp as wp

from .gpu_monitoring import (
    GpuSafetyMonitor,
    GpuSafetySnapshot,
    SAFETY_REASON_COMMANDED_INDENTATION,
)


@wp.kernel
def _build_limited_probe_commands(
    frame_dt: float,
    substeps: int,
    free_speed_m_s: float,
    near_speed_m_s: float,
    contact_speed_m_s: float,
    near_contact_distance_m: float,
    max_angular_speed_rad_s: float,
    max_translation_per_frame_m: float,
    max_rotation_per_frame_rad: float,
    body_index: int,
    target_q: wp.array[wp.transform],
    target_indentation: wp.array[float],
    last_safe_q: wp.array[wp.transform],
    contact_counts: wp.array[int],
    stepper_floats: wp.array[float],
    stepper_ints: wp.array[int],
    command_q: wp.array[wp.transform],
    command_qd: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    start = last_safe_q[0]
    target = target_q[0]
    if (
        stepper_ints[0] != 0
        and target_indentation[0] >= stepper_floats[0] - 1.0e-12
    ):
        target = start

    start_position = wp.transform_get_translation(start)
    target_position = wp.transform_get_translation(target)
    delta = target_position - start_position
    distance = wp.length(delta)
    speed = free_speed_m_s
    if contact_counts[body_index] > 0:
        speed = contact_speed_m_s
    elif stepper_floats[0] >= -near_contact_distance_m:
        speed = near_speed_m_s
    max_translation = speed * frame_dt
    if max_translation > max_translation_per_frame_m:
        max_translation = max_translation_per_frame_m
    translation_ratio = 1.0
    if distance > max_translation and distance > 1.0e-12:
        translation_ratio = max_translation / distance
    final_position = start_position + delta * translation_ratio

    start_rotation = wp.transform_get_rotation(start)
    target_rotation = wp.transform_get_rotation(target)
    relative_rotation = target_rotation * wp.quat_inverse(start_rotation)
    _axis, rotation_angle = wp.quat_to_axis_angle(relative_rotation)
    if rotation_angle > wp.pi:
        rotation_angle = 2.0 * wp.pi - rotation_angle
    max_rotation = max_angular_speed_rad_s * frame_dt
    if max_rotation > max_rotation_per_frame_rad:
        max_rotation = max_rotation_per_frame_rad
    rotation_ratio = 1.0
    if rotation_angle > max_rotation and rotation_angle > 1.0e-12:
        rotation_ratio = max_rotation / rotation_angle
    final_rotation = wp.quat_slerp(
        start_rotation,
        target_rotation,
        rotation_ratio,
    )

    alpha = float(tid + 1) / float(substeps)
    position = start_position + (final_position - start_position) * alpha
    rotation = wp.quat_slerp(start_rotation, final_rotation, alpha)
    linear_velocity = (final_position - start_position) / frame_dt
    frame_relative_rotation = final_rotation * wp.quat_inverse(start_rotation)
    angular_axis, angular_angle = wp.quat_to_axis_angle(frame_relative_rotation)
    if angular_angle > wp.pi:
        angular_angle = 2.0 * wp.pi - angular_angle
        angular_axis = -angular_axis
    angular_velocity = angular_axis * (angular_angle / frame_dt)
    command_q[tid] = wp.transform(position, rotation)
    command_qd[tid] = wp.spatial_vector(linear_velocity, angular_velocity)


@wp.kernel
def _reset_probe_supports(supports: wp.array[float]):
    supports[wp.tid()] = -1.0e30


@wp.kernel
def _mesh_probe_supports(
    loading_direction: wp.vec3,
    local_vertices: wp.array[wp.vec3],
    command_q: wp.array[wp.transform],
    supports: wp.array[float],
):
    substep_index, vertex_index = wp.tid()
    rotation = wp.transform_get_rotation(command_q[substep_index])
    world_vertex = wp.quat_rotate(rotation, local_vertices[vertex_index])
    wp.atomic_max(
        supports,
        substep_index,
        wp.dot(world_vertex, loading_direction),
    )


@wp.kernel
def _commanded_indentation_from_support(
    geometry_kind: int,
    radius: float,
    half_height: float,
    contact_location: wp.vec3,
    loading_direction: wp.vec3,
    supports: wp.array[float],
    command_q: wp.array[wp.transform],
    commanded_indentation: wp.array[float],
):
    tid = wp.tid()
    transform = command_q[tid]
    support = radius
    if geometry_kind == 1 or geometry_kind == 2:
        rotation = wp.transform_get_rotation(transform)
        axis = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
        axial_cosine = wp.dot(loading_direction, axis)
        if axial_cosine < 0.0:
            axial_cosine = -axial_cosine
        if geometry_kind == 1:
            radial_squared = 1.0 - axial_cosine * axial_cosine
            if radial_squared < 0.0:
                radial_squared = 0.0
            support = radius * wp.sqrt(radial_squared) + half_height * axial_cosine
        else:
            support = radius + half_height * axial_cosine
    elif geometry_kind == 3:
        support = supports[tid]
    position = wp.transform_get_translation(transform)
    commanded_indentation[tid] = (
        wp.dot(position - contact_location, loading_direction) + support
    )


@wp.kernel
def _set_commanded_probe_pose(
    substep_index: int,
    body_index: int,
    maximum_commanded_indentation_m: float,
    command_q: wp.array[wp.transform],
    command_qd: wp.array[wp.spatial_vector],
    commanded_indentation: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    int_metrics: wp.array[int],
):
    body_q[body_index] = command_q[substep_index]
    body_qd[body_index] = command_qd[substep_index]
    if (
        maximum_commanded_indentation_m >= 0.0
        and commanded_indentation[substep_index]
        > maximum_commanded_indentation_m
    ):
        int_metrics[15] = 1
        wp.atomic_min(
            int_metrics,
            16,
            SAFETY_REASON_COMMANDED_INDENTATION,
        )


@wp.kernel
def _save_previous_particles(
    accepted_q: wp.array[wp.vec3],
    previous_q: wp.array[wp.vec3],
):
    tid = wp.tid()
    previous_q[tid] = accepted_q[tid]


@wp.kernel
def _reset_stepper_event(stepper_ints: wp.array[int]):
    stepper_ints[3] = 0


@wp.kernel
def _update_recoverable_latch(
    substep_index: int,
    retraction_clearance_m: float,
    commanded_indentation: wp.array[float],
    monitor_floats: wp.array[float],
    monitor_ints: wp.array[int],
    stepper_floats: wp.array[float],
    stepper_ints: wp.array[int],
):
    candidate_indent = commanded_indentation[substep_index]
    latched = stepper_ints[0]
    rejected = monitor_ints[15]
    current_indent = stepper_floats[0]

    if latched != 0 and candidate_indent >= current_indent - 1.0e-12:
        rejected = 1
        monitor_ints[15] = 1
        if monitor_ints[16] == 2_147_483_647:
            monitor_ints[16] = stepper_ints[1]

    if rejected != 0:
        if latched == 0:
            stepper_ints[0] = 1
            stepper_ints[1] = monitor_ints[16]
            stepper_ints[2] = stepper_ints[2] + 1
            stepper_ints[3] = 1
            stepper_floats[1] = candidate_indent
        return

    if latched != 0:
        retraction = stepper_floats[1] - candidate_indent
        if retraction >= retraction_clearance_m and monitor_ints[14] == 0:
            stepper_ints[0] = 0
            stepper_ints[1] = 2_147_483_647


@wp.kernel
def _capture_rejected_particles(
    candidate_q: wp.array[wp.vec3],
    captured_q: wp.array[wp.vec3],
    stepper_ints: wp.array[int],
):
    tid = wp.tid()
    if stepper_ints[3] != 0:
        captured_q[tid] = candidate_q[tid]


@wp.kernel
def _capture_rejected_probe(
    substep_index: int,
    command_q: wp.array[wp.transform],
    commanded_indentation: wp.array[float],
    captured_q: wp.array[wp.transform],
    captured_indentation: wp.array[float],
    stepper_ints: wp.array[int],
):
    if stepper_ints[3] != 0:
        captured_q[0] = command_q[substep_index]
        captured_indentation[0] = commanded_indentation[substep_index]


@wp.kernel
def _conditionally_accept_particles(
    candidate_q: wp.array[wp.vec3],
    candidate_qd: wp.array[wp.vec3],
    accepted_q: wp.array[wp.vec3],
    accepted_qd: wp.array[wp.vec3],
    reject_on_frame_fatal: int,
    monitor_ints: wp.array[int],
):
    tid = wp.tid()
    rejected = monitor_ints[15]
    if reject_on_frame_fatal != 0 and monitor_ints[5] != 0:
        rejected = 1
    if rejected == 0:
        accepted_q[tid] = candidate_q[tid]
        accepted_qd[tid] = candidate_qd[tid]


@wp.kernel
def _conditionally_accept_probe(
    substep_index: int,
    body_index: int,
    reject_on_frame_fatal: int,
    command_q: wp.array[wp.transform],
    command_qd: wp.array[wp.spatial_vector],
    commanded_indentation: wp.array[float],
    last_safe_q: wp.array[wp.transform],
    last_safe_qd: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    monitor_ints: wp.array[int],
    stepper_floats: wp.array[float],
):
    rejected = monitor_ints[15]
    if reject_on_frame_fatal != 0 and monitor_ints[5] != 0:
        rejected = 1
    if rejected == 0:
        last_safe_q[0] = command_q[substep_index]
        last_safe_qd[0] = command_qd[substep_index]
        stepper_floats[0] = commanded_indentation[substep_index]
    else:
        body_q[body_index] = last_safe_q[0]
        body_qd[body_index] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))


@dataclass(frozen=True)
class StepperControlSnapshot:
    safety_stop_active: bool
    safety_reason_code: int
    safety_generation: int
    safety_event_pending: bool
    current_commanded_indentation_m: float
    stop_commanded_indentation_m: float


class InteractiveGpuCommandBuilder:
    """Generate speed-limited manual substep commands entirely on the GPU."""

    _KINDS = {
        "sphere": 0,
        "cylinder": 1,
        "capsule": 2,
        "rounded_block": 3,
        "custom_rigid_stl": 3,
    }

    def __init__(
        self,
        *,
        stepper: GpuFrameStepper,
        probe: dict,
        mesh_vertices_m: np.ndarray | None,
        contact_location_m: np.ndarray,
        loading_direction: np.ndarray,
        frame_dt: float,
        free_speed_m_s: float,
        near_speed_m_s: float,
        contact_speed_m_s: float,
        near_contact_distance_m: float,
        max_angular_speed_deg_s: float,
        max_translation_per_frame_m: float,
        max_rotation_per_frame_deg: float,
    ) -> None:
        self.stepper = stepper
        self.device = stepper.device
        self.geometry_kind = self._KINDS[str(probe["type"])]
        self.radius = float(probe.get("radius_m", 0.0))
        self.half_height = 0.5 * float(probe.get("height_m", 0.0))
        self.contact_location = wp.vec3(*np.asarray(contact_location_m, dtype=float))
        self.loading_direction = wp.vec3(*np.asarray(loading_direction, dtype=float))
        self.frame_dt = float(frame_dt)
        self.free_speed_m_s = float(free_speed_m_s)
        self.near_speed_m_s = float(near_speed_m_s)
        self.contact_speed_m_s = float(contact_speed_m_s)
        self.near_contact_distance_m = float(near_contact_distance_m)
        self.max_angular_speed_rad_s = float(
            np.deg2rad(max_angular_speed_deg_s)
        )
        self.max_translation_per_frame_m = float(max_translation_per_frame_m)
        self.max_rotation_per_frame_rad = float(
            np.deg2rad(max_rotation_per_frame_deg)
        )
        self.target_q_device = wp.zeros(1, dtype=wp.transform, device=self.device)
        self.target_indentation_device = wp.zeros(
            1, dtype=float, device=self.device
        )
        self.supports_device = wp.zeros(
            stepper.sim_substeps, dtype=float, device=self.device
        )
        if self.geometry_kind == 3:
            vertices = np.asarray(mesh_vertices_m, dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                raise ValueError("mesh probe requires finite local support vertices")
            self.local_vertices_device = wp.array(
                vertices, dtype=wp.vec3, device=self.device
            )
        else:
            self.local_vertices_device = wp.empty(
                0, dtype=wp.vec3, device=self.device
            )

    def update_target(
        self, transform: np.ndarray, commanded_indentation_m: float
    ) -> None:
        self.target_q_device.assign(np.asarray(transform).reshape(1, 7))
        self.target_indentation_device.assign(
            np.asarray([commanded_indentation_m], dtype=np.float32)
        )

    def build(self) -> None:
        wp.launch(
            _build_limited_probe_commands,
            dim=self.stepper.sim_substeps,
            inputs=[
                self.frame_dt,
                self.stepper.sim_substeps,
                self.free_speed_m_s,
                self.near_speed_m_s,
                self.contact_speed_m_s,
                self.near_contact_distance_m,
                self.max_angular_speed_rad_s,
                self.max_translation_per_frame_m,
                self.max_rotation_per_frame_rad,
                self.stepper.probe_body,
                self.target_q_device,
                self.target_indentation_device,
                self.stepper.last_safe_q_device,
                self.stepper.monitor.contact_counts,
                self.stepper.stepper_floats_device,
                self.stepper.stepper_ints_device,
                self.stepper.command_q_device,
                self.stepper.command_qd_device,
            ],
            device=self.device,
        )
        wp.launch(
            _reset_probe_supports,
            dim=self.stepper.sim_substeps,
            inputs=[self.supports_device],
            device=self.device,
        )
        if self.geometry_kind == 3:
            wp.launch(
                _mesh_probe_supports,
                dim=(
                    self.stepper.sim_substeps,
                    len(self.local_vertices_device),
                ),
                inputs=[
                    self.loading_direction,
                    self.local_vertices_device,
                    self.stepper.command_q_device,
                    self.supports_device,
                ],
                device=self.device,
            )
        wp.launch(
            _commanded_indentation_from_support,
            dim=self.stepper.sim_substeps,
            inputs=[
                self.geometry_kind,
                self.radius,
                self.half_height,
                self.contact_location,
                self.loading_direction,
                self.supports_device,
                self.stepper.command_q_device,
                self.stepper.commanded_indentation_device,
            ],
            device=self.device,
        )


class GpuFrameStepper:
    """Execute all Newton substeps without host readback.

    ``state_0`` remains the last accepted state and ``state_1`` is always a
    scratch candidate. This makes an unsafe interactive movement recoverable
    without a Python decision inside the substep loop.
    """

    def __init__(
        self,
        *,
        model,
        state_0,
        state_1,
        control,
        contacts,
        collision_pipeline,
        solver,
        monitor: GpuSafetyMonitor,
        probe_body: int,
        sim_substeps: int,
        sim_dt: float,
        maximum_commanded_indentation_m: float = -1.0,
        recoverable_stop: bool = False,
        retraction_clearance_m: float = 0.0,
        reject_on_frame_fatal: bool = True,
        cuda_graph_requested: bool = True,
    ) -> None:
        self.model = model
        self.state_0 = state_0
        self.state_1 = state_1
        self.control = control
        self.contacts = contacts
        self.collision_pipeline = collision_pipeline
        self.solver = solver
        self.monitor = monitor
        self.probe_body = int(probe_body)
        self.sim_substeps = int(sim_substeps)
        self.sim_dt = float(sim_dt)
        self.maximum_commanded_indentation_m = float(
            maximum_commanded_indentation_m
        )
        self.recoverable_stop = bool(recoverable_stop)
        self.retraction_clearance_m = float(retraction_clearance_m)
        self.reject_on_frame_fatal = int(reject_on_frame_fatal)
        self.device = model.device
        self.cuda_graph_requested = bool(cuda_graph_requested)
        self.cuda_graph = None
        self.cuda_graph_supported = bool(
            self.device.is_cuda and wp.is_mempool_enabled(self.device)
        )
        self.cuda_graph_error = ""
        self.command_builder: Callable[[], None] | None = None

        self.command_q_device = wp.zeros(
            self.sim_substeps, dtype=wp.transform, device=self.device
        )
        self.command_qd_device = wp.zeros(
            self.sim_substeps, dtype=wp.spatial_vector, device=self.device
        )
        self.commanded_indentation_device = wp.zeros(
            self.sim_substeps, dtype=float, device=self.device
        )
        self.last_safe_q_device = wp.zeros(1, dtype=wp.transform, device=self.device)
        self.last_safe_qd_device = wp.zeros(
            1, dtype=wp.spatial_vector, device=self.device
        )
        self.rejected_particle_q_device = wp.zeros_like(self.state_0.particle_q)
        self.previous_particle_q_device = wp.zeros_like(self.state_0.particle_q)
        self.previous_particle_q_device.assign(self.state_0.particle_q)
        self.rejected_probe_q_device = wp.zeros(
            1, dtype=wp.transform, device=self.device
        )
        self.rejected_indentation_device = wp.zeros(
            1, dtype=float, device=self.device
        )
        # [latched, reason, generation, capture_event]
        self.stepper_ints_device = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.stepper_ints_device.fill_(2_147_483_647)
        self.stepper_ints_device.assign(
            np.asarray([0, 2_147_483_647, 0, 0], dtype=np.int32)
        )
        # [current indentation, stop indentation]
        self.stepper_floats_device = wp.zeros(2, dtype=float, device=self.device)
        self.last_safe_q_device.assign(
            wp.array(
                self.state_0.body_q.numpy()[self.probe_body : self.probe_body + 1],
                dtype=wp.transform,
                device=self.device,
            )
        )
        self.last_safe_qd_device.assign(
            wp.array(
                self.state_0.body_qd.numpy()[self.probe_body : self.probe_body + 1],
                dtype=wp.spatial_vector,
                device=self.device,
            )
        )

    @property
    def graph_enabled(self) -> bool:
        return self.cuda_graph is not None

    def upload_commands(
        self,
        transforms: np.ndarray,
        velocities: np.ndarray,
        commanded_indentation_m: np.ndarray,
    ) -> None:
        transforms = np.asarray(transforms)
        velocities = np.asarray(velocities)
        indentation = np.asarray(commanded_indentation_m, dtype=np.float32)
        if len(transforms) != self.sim_substeps:
            raise ValueError("one probe transform is required per simulation substep")
        self.command_q_device.assign(transforms)
        self.command_qd_device.assign(velocities)
        self.commanded_indentation_device.assign(indentation)

    def initialize_last_safe_pose(
        self,
        transform: np.ndarray,
        velocity: np.ndarray,
        commanded_indentation_m: float,
    ) -> None:
        self.last_safe_q_device.assign(np.asarray(transform).reshape(1, 7))
        self.last_safe_qd_device.assign(np.asarray(velocity).reshape(1, 6))
        self.stepper_floats_device.assign(
            np.asarray(
                [commanded_indentation_m, commanded_indentation_m],
                dtype=np.float32,
            )
        )

    def reset_transaction(
        self,
        transform: np.ndarray,
        velocity: np.ndarray,
        commanded_indentation_m: float,
    ) -> None:
        self.initialize_last_safe_pose(
            transform,
            velocity,
            commanded_indentation_m,
        )
        self.stepper_ints_device.assign(
            np.asarray([0, 2_147_483_647, 0, 0], dtype=np.int32)
        )
        self.previous_particle_q_device.assign(self.state_0.particle_q)

    def _build_frame_operations(self, *, include_free_speed: bool) -> None:
        if self.command_builder is not None:
            self.command_builder()
        self.monitor.reset_frame()
        for substep_index in range(self.sim_substeps):
            self.monitor.reset_candidate()
            wp.launch(
                _reset_stepper_event,
                dim=1,
                inputs=[self.stepper_ints_device],
                device=self.device,
            )
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            wp.launch(
                _save_previous_particles,
                dim=len(self.state_0.particle_q),
                inputs=[
                    self.state_0.particle_q,
                    self.previous_particle_q_device,
                ],
                device=self.device,
            )
            wp.launch(
                _set_commanded_probe_pose,
                dim=1,
                inputs=[
                    substep_index,
                    self.probe_body,
                    self.maximum_commanded_indentation_m,
                    self.command_q_device,
                    self.command_qd_device,
                    self.commanded_indentation_device,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.monitor.int_metrics_device,
                ],
                device=self.device,
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.monitor.evaluate_candidate(
                self.state_1,
                include_free_speed=include_free_speed,
            )
            if self.recoverable_stop:
                wp.launch(
                    _update_recoverable_latch,
                    dim=1,
                    inputs=[
                        substep_index,
                        self.retraction_clearance_m,
                        self.commanded_indentation_device,
                        self.monitor.float_metrics_device,
                        self.monitor.int_metrics_device,
                        self.stepper_floats_device,
                        self.stepper_ints_device,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _capture_rejected_particles,
                    dim=len(self.state_0.particle_q),
                    inputs=[
                        self.state_1.particle_q,
                        self.rejected_particle_q_device,
                        self.stepper_ints_device,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _capture_rejected_probe,
                    dim=1,
                    inputs=[
                        substep_index,
                        self.command_q_device,
                        self.commanded_indentation_device,
                        self.rejected_probe_q_device,
                        self.rejected_indentation_device,
                        self.stepper_ints_device,
                    ],
                    device=self.device,
                )
            wp.launch(
                _conditionally_accept_particles,
                dim=len(self.state_0.particle_q),
                inputs=[
                    self.state_1.particle_q,
                    self.state_1.particle_qd,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.reject_on_frame_fatal,
                    self.monitor.int_metrics_device,
                ],
                device=self.device,
            )
            wp.launch(
                _conditionally_accept_probe,
                dim=1,
                inputs=[
                    substep_index,
                    self.probe_body,
                    self.reject_on_frame_fatal,
                    self.command_q_device,
                    self.command_qd_device,
                    self.commanded_indentation_device,
                    self.last_safe_q_device,
                    self.last_safe_qd_device,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.monitor.int_metrics_device,
                    self.stepper_floats_device,
                ],
                device=self.device,
            )

    def capture(self, *, include_free_speed: bool = False) -> bool:
        if (
            not self.cuda_graph_requested
            or not self.cuda_graph_supported
            or self.cuda_graph is not None
        ):
            return self.cuda_graph is not None
        try:
            with wp.ScopedCapture(
                device=self.device,
                force_module_load=False,
            ) as capture:
                self._build_frame_operations(
                    include_free_speed=include_free_speed
                )
            self.cuda_graph = capture.graph
        except Exception as exc:  # pragma: no cover - runtime/backend dependent
            self.cuda_graph = None
            self.cuda_graph_error = f"{type(exc).__name__}: {exc}"
        return self.cuda_graph is not None

    def run_frame(self, *, include_free_speed: bool = False) -> GpuSafetySnapshot:
        if self.cuda_graph is not None:
            wp.capture_launch(self.cuda_graph)
        else:
            self._build_frame_operations(include_free_speed=include_free_speed)
        return self.monitor.readback()

    def control_readback(self) -> StepperControlSnapshot:
        ints = self.stepper_ints_device.numpy()
        floats = self.stepper_floats_device.numpy()
        return StepperControlSnapshot(
            safety_stop_active=bool(ints[0]),
            safety_reason_code=int(ints[1]),
            safety_generation=int(ints[2]),
            safety_event_pending=bool(ints[3]),
            current_commanded_indentation_m=float(floats[0]),
            stop_commanded_indentation_m=float(floats[1]),
        )

    def rejected_state_readback(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        particles = self.rejected_particle_q_device.numpy()
        pose = self.rejected_probe_q_device.numpy()
        indentation = float(self.rejected_indentation_device.numpy()[0])
        return particles, pose, indentation

    def accepted_pose_readback(self) -> tuple[np.ndarray, np.ndarray]:
        return self.last_safe_q_device.numpy()[0], self.last_safe_qd_device.numpy()[0]

    def set_host_recoverable_latch(
        self, *, reason_code: int, commanded_indentation_m: float
    ) -> bool:
        ints = self.stepper_ints_device.numpy()
        if bool(ints[0]):
            return False
        floats = self.stepper_floats_device.numpy()
        ints[0] = 1
        ints[1] = int(reason_code)
        floats[1] = float(commanded_indentation_m)
        self.stepper_ints_device.assign(ints)
        self.stepper_floats_device.assign(floats)
        return True


class CudaGraphFrame:
    """Small reusable graph-capture wrapper for non-stepper GPU frame paths."""

    def __init__(self, device, build: Callable[[], None], *, requested: bool) -> None:
        self.device = device
        self.graph = None
        self.error = ""
        if not requested or not device.is_cuda or not wp.is_mempool_enabled(device):
            return
        try:
            with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                build()
            self.graph = capture.graph
        except Exception as exc:  # pragma: no cover - runtime/backend dependent
            self.error = f"{type(exc).__name__}: {exc}"

    def launch(self, fallback: Callable[[], None]) -> None:
        if self.graph is None:
            fallback()
        else:
            wp.capture_launch(self.graph)
