#!/usr/bin/env python3
"""Configuration-driven custom-fingertip mechanics simulation.

This is deliberately separate from the ToF simulator.  It runs Newton VBD,
tracks a prepared high-resolution coating surface, and exports mechanical
ground truth for a deterministic touch trajectory.
"""

# ruff: noqa: E402 -- the in-tree Newton checkout must precede package imports.

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_LOCAL_NEWTON_ROOT = Path(__file__).resolve().parents[1] / "newton"
if (_LOCAL_NEWTON_ROOT / "newton").is_dir():
    sys.path.insert(0, str(_LOCAL_NEWTON_ROOT))

import warp as wp

import newton
import newton.examples
from newton.solvers import SolverVBD

from sim.mechanics.config import load_run_config, material_lame_parameters
from sim.mechanics.contact import (
    ContactSummary,
    contact_face_mask,
    estimate_contact_summary,
    masked_triangle_area,
    quaternion_rotate_xyzw,
)
from sim.mechanics.exporter import MechanicalDataExporter, export_failure_state
from sim.mechanics.mapping import SurfaceMapping, reconstruct_surface
from sim.mechanics.mesh import tet_signed_volumes, validate_tets
from sim.mechanics.trajectory import DeterministicTrajectory


@wp.kernel
def _set_kinematic_pose(
    body_index: int,
    px: float,
    py: float,
    pz: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    vx: float,
    vy: float,
    vz: float,
    wx: float,
    wy: float,
    wz: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    body_q[body_index] = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
    body_qd[body_index] = wp.spatial_vector(wp.vec3(vx, vy, vz), wp.vec3(wx, wy, wz))


def _git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _normalized_quaternion(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (4,) or norm <= 0.0:
        raise ValueError("quaternion_xyzw must contain four nonzero values")
    return result / norm


def _indenter_contact_translation(
    config: dict[str, Any], direction: np.ndarray
) -> np.ndarray:
    location = np.asarray(config["contact"]["location_m"], dtype=np.float64)
    indenter = config["indenter"]
    kind = indenter["type"]
    if kind == "sphere":
        support = float(indenter["radius_m"])
        return location - direction * support
    if kind == "flat_plate":
        support = 0.5 * float(indenter["thickness_m"])
        return location - direction * support
    if kind == "cylinder":
        mode = indenter.get("contact_surface", "cap")
        support = (
            0.5 * float(indenter["height_m"])
            if mode == "cap"
            else float(indenter["radius_m"])
        )
        return location - direction * support
    if kind == "rigid_stl":
        local_point = np.asarray(indenter["contact_point_local_m"], dtype=np.float64)
        quaternion = _normalized_quaternion(indenter["quaternion_xyzw"])
        return location - quaternion_rotate_xyzw(quaternion, local_point)
    raise ValueError(f"unsupported indenter type {kind!r}")


class TouchMechanicsController:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.config = load_run_config(args.config)
        self.trajectory = DeterministicTrajectory(self.config["trajectory"])
        solver_cfg = self.config["solver"]
        self.frame_dt = 1.0 / float(solver_cfg["simulation_fps"])
        self.sim_substeps = int(solver_cfg["substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = int(solver_cfg["vbd_iterations"])
        self.sim_time = 0.0
        self.substep = 0
        self.next_output_time = 0.0
        self.output_period = 1.0 / float(self.config["output"]["rate_hz"])
        self.finalized = False

        direction = np.asarray(self.config["contact"]["direction"], dtype=np.float64)
        self.loading_direction = direction / np.linalg.norm(direction)
        self.indenter_quaternion = _normalized_quaternion(
            self.config["indenter"]["quaternion_xyzw"]
        )
        contact_translation = _indenter_contact_translation(
            self.config, self.loading_direction
        )
        self.indenter_start_position = (
            contact_translation
            - self.loading_direction * float(self.config["trajectory"]["clearance_m"])
        )
        self.current_sample = self.trajectory.sample(0.0)
        self.current_body_position = self.indenter_start_position.copy()
        self.current_body_velocity = np.zeros(3, dtype=np.float64)
        self.current_body_angular_velocity = np.zeros(3, dtype=np.float64)

        asset = self.config["asset"]
        for field in (
            "surface_stl",
            "volume_msh",
            "regions_npz",
            "surface_mapping_npz",
        ):
            if not Path(asset[field]).is_file():
                raise FileNotFoundError(
                    f"prepared asset is missing {field}: {asset[field]}"
                )
        regions = np.load(asset["regions_npz"])
        mapping_file = np.load(asset["surface_mapping_npz"])
        self.surface_rest_vertices_m = np.asarray(
            mapping_file["surface_rest_vertices_m"], dtype=np.float64
        )
        self.surface_faces = np.asarray(mapping_file["surface_faces"], dtype=np.int32)
        self.mapping_tets = np.asarray(mapping_file["tet_indices"], dtype=np.int32)
        self.surface_mapping = SurfaceMapping(
            tet_ids=np.asarray(mapping_file["surface_tet_ids"], dtype=np.int32),
            barycentric_weights=np.asarray(
                mapping_file["barycentric_weights"], dtype=np.float64
            ),
            reconstruction_error_m=np.asarray(
                mapping_file["reconstruction_error_m"], dtype=np.float64
            ),
        )
        self.inner_vertices = np.asarray(
            regions["inner_coating_vertices"], dtype=np.int32
        )
        self.outer_vertices = np.asarray(
            regions["outer_contact_vertices"], dtype=np.int32
        )
        self.outer_faces = np.asarray(regions["outer_contact_faces"], dtype=np.int32)
        mount_vertices = np.asarray(regions["mount_vertices"], dtype=np.int32)

        tetmesh = newton.TetMesh.create_from_file(asset["volume_msh"])
        local_vertices = np.asarray(tetmesh.vertices, dtype=np.float64)
        local_tets = np.asarray(tetmesh.tet_indices, dtype=np.int32).reshape(-1, 4)
        if not np.array_equal(local_tets, self.mapping_tets):
            raise RuntimeError(
                "surface_mapping.npz tetrahedra do not match volume.msh; rerun prepare_fingertip.py"
            )
        transform_cfg = self.config["fingertip_transform"]
        scale = float(transform_cfg["scale"])
        position = np.asarray(transform_cfg["position_m"], dtype=np.float64)
        quaternion = _normalized_quaternion(transform_cfg["quaternion_xyzw"])
        transformed_vertices = (
            quaternion_rotate_xyzw(quaternion, local_vertices * scale) + position
        )
        monitor_cfg = self.config["monitoring"]
        rest_report = validate_tets(
            transformed_vertices,
            local_tets,
            min_volume_m3=float(monitor_cfg["minimum_rest_tet_volume_m3"]),
            min_quality=float(monitor_cfg["minimum_rest_tet_quality"]),
            max_condition_number=float(
                monitor_cfg["maximum_rest_tet_condition_number"]
            ),
        )
        self.rest_volumes = tet_signed_volumes(transformed_vertices, local_tets)
        self.particle_radius = (
            float(solver_cfg["particle_radius_edge_fraction"])
            * rest_report.median_boundary_edge_m
        )

        material = self.config["material"]
        mu, lam = material_lame_parameters(material)
        builder = newton.ModelBuilder(gravity=float(solver_cfg["gravity_m_s2"]))
        self.particle_start = len(builder.particle_q)
        builder.add_soft_mesh(
            pos=wp.vec3(*position),
            rot=wp.quat(*quaternion),
            scale=scale,
            vel=wp.vec3(0.0),
            mesh=tetmesh,
            density=float(material["density_kg_m3"]),
            k_mu=mu,
            k_lambda=lam,
            k_damp=float(material["damping"]["value"]),
            particle_radius=self.particle_radius,
        )
        self.particle_end = len(builder.particle_q)
        if self.particle_end - self.particle_start != len(local_vertices):
            raise RuntimeError("Newton changed the prepared tet vertex count")
        self.tet_indices_global = local_tets + self.particle_start
        for local_index in mount_vertices:
            builder.particle_mass[self.particle_start + int(local_index)] = 0.0

        body = builder.add_body(
            xform=wp.transform(
                wp.vec3(*self.indenter_start_position),
                wp.quat(*self.indenter_quaternion),
            ),
            label="touch_indenter",
            is_kinematic=True,
        )
        self.indenter_body = body
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        contact_cfg = self.config["contact_parameters"]
        shape_cfg.density = 0.0
        shape_cfg.ke = float(contact_cfg["normal_stiffness"])
        shape_cfg.kd = float(contact_cfg["normal_damping"])
        shape_cfg.mu = float(contact_cfg["static_friction"])
        self._add_indenter_shape(builder, body, shape_cfg)
        builder.color()
        self.model = builder.finalize()
        self.model.soft_contact_ke = float(contact_cfg["normal_stiffness"])
        self.model.soft_contact_kd = float(contact_cfg["normal_damping"])
        self.model.soft_contact_mu = float(contact_cfg["dynamic_friction"])
        self.model.shape_material_ke.fill_(float(contact_cfg["normal_stiffness"]))
        self.model.shape_material_kd.fill_(float(contact_cfg["normal_damping"]))
        self.model.shape_material_mu.fill_(float(contact_cfg["static_friction"]))

        self.solver = SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=bool(
                solver_cfg["particle_enable_self_contact"]
            ),
            particle_enable_tile_solve=bool(solver_cfg["particle_enable_tile_solve"]),
            particle_collision_detection_interval=int(
                solver_cfg["particle_collision_detection_interval"]
            ),
            rigid_body_particle_contact_buffer_size=int(
                solver_cfg["rigid_body_particle_contact_buffer_size"]
            ),
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, soft_contact_margin=float(contact_cfg["margin_m"])
        )
        self.contacts = self.collision_pipeline.contacts()
        self.rest_particle_positions = self.state_0.particle_q.numpy()[
            self.particle_start : self.particle_end
        ].astype(np.float64)
        self.particle_radii = self.model.particle_radius.numpy().astype(np.float64)

        self.exporter = MechanicalDataExporter(
            args.output_dir or self.config["output"]["directory"], self.config
        )
        mapping_destination = self.exporter.output_dir / "surface_mapping.npz"
        if (
            Path(asset["surface_mapping_npz"]).resolve()
            != mapping_destination.resolve()
        ):
            shutil.copy2(asset["surface_mapping_npz"], mapping_destination)
        self.failure_path = self.exporter.output_dir / "failure_state.npz"
        if self.viewer is not None:
            self.viewer.set_model(self.model)
            camera = self.config.get("viewer", {}).get("camera")
            if camera:
                self.viewer.set_camera(
                    wp.vec3(*camera["position_m"]),
                    float(camera["yaw_degrees"]),
                    float(camera["pitch_degrees"]),
                )

        self.newton_version = str(getattr(newton, "__version__", "unknown"))
        self.newton_revision = _git_revision(_LOCAL_NEWTON_ROOT)
        print(
            f"[mechanics] asset={asset.get('asset_id', asset['volume_msh'])} "
            f"particles={self.particle_end - self.particle_start} tets={len(local_tets)} "
            f"mount_vertices={len(mount_vertices)} particle_radius={self.particle_radius * 1000:.3f} mm"
        )
        print(
            f"[mechanics] material E={material['youngs_modulus_pa']} Pa "
            f"nu={material['poisson_ratio']} rho={material['density_kg_m3']} kg/m^3 "
            f"mu={mu:.3f} Pa lambda={lam:.3f} Pa"
        )
        print(
            f"[mechanics] Newton version={self.newton_version} commit={self.newton_revision} "
            f"package={newton.__file__} damping={material['damping']['semantics']}"
        )
        print(
            f"[mechanics] trajectory duration={self.trajectory.total_duration_s:.3f} s "
            f"phases={','.join(self.trajectory.phase_order)} output={self.exporter.output_dir}"
        )
        self._export_frame(0.0)

    def _add_indenter_shape(self, builder, body: int, shape_cfg) -> None:
        indenter = self.config["indenter"]
        kind = indenter["type"]
        if kind == "sphere":
            builder.add_shape_sphere(
                body, radius=float(indenter["radius_m"]), cfg=shape_cfg
            )
        elif kind == "flat_plate":
            builder.add_shape_box(
                body,
                hx=0.5 * float(indenter["width_m"]),
                hy=0.5 * float(indenter["depth_m"]),
                hz=0.5 * float(indenter["thickness_m"]),
                cfg=shape_cfg,
            )
        elif kind == "cylinder":
            builder.add_shape_cylinder(
                body,
                radius=float(indenter["radius_m"]),
                half_height=0.5 * float(indenter["height_m"]),
                cfg=shape_cfg,
            )
        elif kind == "rigid_stl":
            mesh = newton.Mesh.create_from_file(indenter["stl"])
            scale = float(indenter["scale_to_m"])
            builder.add_shape_mesh(
                body,
                mesh=mesh,
                scale=wp.vec3(scale, scale, scale),
                cfg=shape_cfg,
            )

    def _command_indenter(self, state, time_s: float) -> None:
        sample = self.trajectory.sample(time_s)
        position = (
            self.indenter_start_position
            + self.loading_direction * sample.normal_travel_m
            + sample.lateral_offset_m
        )
        velocity = (
            self.loading_direction * sample.normal_velocity_m_s
            + sample.lateral_velocity_m_s
        )
        q = self.indenter_quaternion
        wp.launch(
            _set_kinematic_pose,
            dim=1,
            inputs=[
                self.indenter_body,
                *position,
                *q,
                *velocity,
                *self.current_body_angular_velocity,
            ],
            outputs=[state.body_q, state.body_qd],
            device=self.model.device,
        )
        self.current_sample = sample
        self.current_body_position = position
        self.current_body_velocity = velocity

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self._command_indenter(self.state_0, (self.substep + 1) * self.sim_dt)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.substep += 1
            self.state_0, self.state_1 = self.state_1, self.state_0
            interval = int(self.config["monitoring"]["tet_check_interval_substeps"])
            if self.substep % interval == 0:
                self._check_tet_state()

    def _relative_tet_volumes(self, particle_positions: np.ndarray) -> np.ndarray:
        current = tet_signed_volumes(particle_positions, self.mapping_tets)
        return current / self.rest_volumes

    def _check_tet_state(self) -> None:
        particles = self.state_0.particle_q.numpy()[
            self.particle_start : self.particle_end
        ].astype(np.float64)
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
            sim_time_s=np.float64(self.substep * self.sim_dt),
            trajectory_phase=np.asarray(self.current_sample.phase),
            nominal_indentation_m=np.float64(
                self.trajectory.nominal_indentation_m(self.current_sample)
            ),
            newton_version=np.asarray(self.newton_version),
            newton_revision=np.asarray(self.newton_revision),
        )
        self.exporter.finalize()
        raise FloatingPointError(
            f"relative tet volume crossed {threshold:g}; state saved to {self.failure_path}"
        )

    def _contact_summary(
        self, current_global: np.ndarray, previous_global: np.ndarray
    ) -> ContactSummary:
        count = min(
            int(self.contacts.soft_contact_count.numpy()[0]),
            self.contacts.soft_contact_max,
        )
        if count <= 0:
            return estimate_contact_summary(
                particle_positions_m=current_global,
                previous_particle_positions_m=previous_global,
                particle_radii_m=self.particle_radii,
                contact_particles=np.empty(0, dtype=np.int32),
                contact_normals=np.empty((0, 3)),
                contact_body_positions_m=np.empty((0, 3)),
                penalty_stiffness_n_m=np.empty(0),
                damping_ratio=np.empty(0),
                friction_coefficients=np.empty(0),
                body_position_m=self.current_body_position,
                body_quaternion_xyzw=self.indenter_quaternion,
                body_linear_velocity_m_s=self.current_body_velocity,
                body_angular_velocity_rad_s=self.current_body_angular_velocity,
                loading_direction=self.loading_direction,
                dt_s=self.sim_dt,
                force_threshold_n=float(
                    self.config["contact_parameters"]["force_threshold_n"]
                ),
                friction_epsilon_m_s=float(
                    self.config["contact_parameters"]["friction_epsilon_m_s"]
                ),
            )
        contact_cfg = self.config["contact_parameters"]
        stiffness_array = getattr(self.solver, "body_particle_contact_penalty_k", None)
        damping_array = getattr(self.solver, "body_particle_contact_material_kd", None)
        friction_array = getattr(self.solver, "body_particle_contact_material_mu", None)
        stiffness = (
            stiffness_array.numpy()[:count]
            if stiffness_array is not None
            else np.full(count, float(contact_cfg["normal_stiffness"]))
        )
        damping = (
            damping_array.numpy()[:count]
            if damping_array is not None
            else np.full(count, float(contact_cfg["normal_damping"]))
        )
        if friction_array is not None:
            # VBD caches sqrt(particle_mu * shape_mu), then combines that
            # cached value with shape_mu once more inside its evaluator.
            friction = np.sqrt(
                friction_array.numpy()[:count] * float(contact_cfg["static_friction"])
            )
        else:
            friction = np.full(
                count,
                math.sqrt(
                    float(contact_cfg["dynamic_friction"])
                    * float(contact_cfg["static_friction"])
                ),
            )
        return estimate_contact_summary(
            particle_positions_m=current_global,
            previous_particle_positions_m=previous_global,
            particle_radii_m=self.particle_radii,
            contact_particles=self.contacts.soft_contact_particle.numpy()[:count],
            contact_normals=self.contacts.soft_contact_normal.numpy()[:count],
            contact_body_positions_m=self.contacts.soft_contact_body_pos.numpy()[
                :count
            ],
            penalty_stiffness_n_m=stiffness,
            damping_ratio=damping,
            friction_coefficients=friction,
            body_position_m=self.current_body_position,
            body_quaternion_xyzw=self.indenter_quaternion,
            body_linear_velocity_m_s=self.current_body_velocity,
            body_angular_velocity_rad_s=self.current_body_angular_velocity,
            loading_direction=self.loading_direction,
            dt_s=self.sim_dt,
            force_threshold_n=float(contact_cfg["force_threshold_n"]),
            friction_epsilon_m_s=float(contact_cfg["friction_epsilon_m_s"]),
        )

    def _export_frame(self, time_s: float) -> None:
        current_global = self.state_0.particle_q.numpy().astype(np.float64)
        previous_global = self.state_1.particle_q.numpy().astype(np.float64)
        particles = current_global[self.particle_start : self.particle_end]
        deformed_surface = reconstruct_surface(
            particles, self.mapping_tets, self.surface_mapping
        )
        relative = self._relative_tet_volumes(particles)
        contact = self._contact_summary(current_global, previous_global)
        active_global = contact.active_particle_indices
        active_local = active_global - self.particle_start
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
            deformed_surface, self.surface_faces, self.outer_faces, face_mask
        )
        displacement = np.linalg.norm(particles - self.rest_particle_positions, axis=1)
        inverted = int(np.count_nonzero(relative <= 0.0))
        minimum_relative = float(np.nanmin(relative))
        maximum_displacement = float(np.nanmax(displacement))
        sample = self.trajectory.sample(time_s)
        body_velocity = np.concatenate(
            (self.current_body_velocity, self.current_body_angular_velocity)
        )
        frame = {
            "timestamp_s": np.float64(time_s),
            "object_position_m": self.current_body_position.copy(),
            "object_quaternion_xyzw": self.indenter_quaternion.copy(),
            "object_velocity_m_s": body_velocity,
            "tet_particle_positions_m": particles.copy(),
            "deformed_surface_vertices_m": deformed_surface,
            "deformed_inner_coating_vertices_m": deformed_surface[self.inner_vertices],
            "deformed_outer_surface_vertices_m": deformed_surface[self.outer_vertices],
            "contact_flag": bool(contact.contact_flag),
            "contact_face_mask": face_mask,
            "contact_area_m2": np.float64(area),
            "normal_force_n": np.float64(contact.normal_force_n),
            "tangential_force_n": contact.tangential_force_n,
            "slip_velocity_m_s": contact.slip_velocity_m_s,
            "maximum_displacement_m": np.float64(maximum_displacement),
            "minimum_relative_tet_volume": np.float64(minimum_relative),
            "inverted_tet_count": np.int32(inverted),
            "trajectory_phase": sample.phase,
        }
        metric = {
            "timestamp_s": time_s,
            "trajectory_phase": sample.phase,
            "contact_flag": int(contact.contact_flag),
            "contact_area_m2": area,
            "normal_force_n": contact.normal_force_n,
            "tangential_force_x_n": contact.tangential_force_n[0],
            "tangential_force_y_n": contact.tangential_force_n[1],
            "tangential_force_z_n": contact.tangential_force_n[2],
            "slip_velocity_x_m_s": contact.slip_velocity_m_s[0],
            "slip_velocity_y_m_s": contact.slip_velocity_m_s[1],
            "slip_velocity_z_m_s": contact.slip_velocity_m_s[2],
            "maximum_displacement_m": maximum_displacement,
            "minimum_relative_tet_volume": minimum_relative,
            "inverted_tet_count": inverted,
        }
        self.exporter.append(frame, metric)

    def step(self) -> None:
        if self.finalized:
            return
        self.simulate()
        self.sim_time = min(
            self.sim_time + self.frame_dt, self.trajectory.total_duration_s
        )
        if self.sim_time + 1.0e-12 >= self.next_output_time + self.output_period:
            while self.next_output_time + self.output_period <= self.sim_time + 1.0e-12:
                self.next_output_time += self.output_period
            self._export_frame(self.sim_time)
        if self.sim_time + 1.0e-12 >= self.trajectory.total_duration_s:
            if not self.exporter.metrics or not math.isclose(
                float(self.exporter.metrics[-1]["timestamp_s"]), self.sim_time
            ):
                self._export_frame(self.sim_time)
            self.exporter.finalize()
            self.finalized = True
            last = self.exporter.metrics[-1]
            print(
                f"[mechanics] complete frames={len(self.exporter.metrics)} "
                f"final_phase={last['trajectory_phase']} min_J="
                f"{min(row['minimum_relative_tet_volume'] for row in self.exporter.metrics):.6f}"
            )

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if bool(self.config.get("viewer", {}).get("render_contacts", False)):
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def main() -> None:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "mechanics"
        / "experiments"
        / "sphere_regression.json"
    )
    parser = newton.examples.create_parser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir", default=None)
    pre_args, _ = parser.parse_known_args()
    resolved = load_run_config(pre_args.config)
    trajectory = DeterministicTrajectory(resolved["trajectory"])
    required_frames = int(
        math.ceil(
            trajectory.total_duration_s * float(resolved["solver"]["simulation_fps"])
        )
    )
    parser.set_defaults(num_frames=required_frames)
    viewer, args = newton.examples.init(parser)
    example = TouchMechanicsController(viewer, args)
    newton.examples.run(example, args)
    if not example.finalized:
        example.exporter.finalize()


if __name__ == "__main__":
    main()
