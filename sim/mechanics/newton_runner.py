"""Newton VBD controller for equilibrated normal-indentation mechanics."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from .config import load_run_config, material_lame_parameters
from .contact import (
    ContactSummary,
    contact_face_mask,
    estimate_contact_summary,
    masked_triangle_area,
)
from .exporter import MechanicalDataExporter, export_failure_state
from .indenter import (
    indenter_contact_translation,
    normalized_quaternion_xyzw,
    normalized_vector,
)
from .mapping import SurfaceMapping, reconstruct_surface
from .mesh import tet_signed_volumes, validate_tets
from .newton_support import (
    SUPPORTED_NEWTON_GIT_SHA,
    SUPPORTED_NEWTON_VERSION,
    deterministic_constructor_kwargs,
    git_revision,
    verify_newton_revision,
)
from .schema import (
    CONTACT_FORCE_ESTIMATOR_VERSION,
    CONTACT_METRIC_MODEL,
    MECHANICS_OUTPUT_SCHEMA_VERSION,
    SHEAR_VALIDATED,
    SIMULATION_CAPABILITY,
    SLIP_VALIDATED,
)
from .trajectory import PrescribedTrajectory
from .video import VideoRecorder


@wp.kernel
def _set_kinematic_pose_v2(
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copy_if_different(source: str | Path, destination: Path) -> None:
    source_path = Path(source).resolve()
    if source_path != destination.resolve():
        shutil.copy2(source_path, destination)


class TouchMechanicsControllerV2:
    """Equilibrate a mounted soft body, then run a prescribed touch trajectory."""

    def __init__(self, viewer, args: Any | None = None):
        args = args or SimpleNamespace(
            config=None,
            output_dir=None,
            record_video=False,
            video_path=None,
            strict_newton=False,
        )
        self.viewer = viewer
        self.config = load_run_config(args.config)
        self._configure_video(args)
        self.video_enabled = bool(self.config["video"]["enabled"])
        self.trajectory = PrescribedTrajectory(self.config["trajectory"])
        solver_cfg = self.config["solver"]
        self.frame_dt = 1.0 / float(solver_cfg["simulation_fps"])
        self.sim_substeps = int(solver_cfg["substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = int(solver_cfg["vbd_iterations"])
        self.post_recovery_s = float(self.config["trajectory"]["post_recovery_s"])
        self.touch_runtime_s = self.trajectory.total_duration_s + self.post_recovery_s
        self.sim_time = 0.0
        self.trajectory_time_s = 0.0
        self.substep = 0
        self.trajectory_substep = 0
        self.next_output_time = 0.0
        self.output_period = 1.0 / float(self.config["output"]["rate_hz"])
        self.lifecycle_phase = "initialization"
        self.finalized = False
        self.video_recorder: VideoRecorder | None = None

        self.equilibration_config = self.config["equilibration"]
        self.equilibration_elapsed_s = 0.0
        self.equilibration_stable_frames = 0
        self.equilibration_converged = False
        self.equilibration_timed_out = False
        self.max_free_particle_speed_m_s = float("nan")
        self.equilibrated_particle_positions: np.ndarray | None = None
        self.equilibrated_surface_positions: np.ndarray | None = None

        self.loading_direction = normalized_vector(
            self.config["contact"]["direction"], name="contact direction"
        )
        self.indenter_quaternion = normalized_quaternion_xyzw(
            self.config["indenter"]["quaternion_xyzw"]
        )
        contact_translation = indenter_contact_translation(
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

        self.newton_version = str(getattr(newton, "__version__", "unknown"))
        newton_root = Path(__file__).resolve().parents[1] / "newton"
        self.newton_revision = git_revision(newton_root)
        strict_newton = bool(
            getattr(args, "strict_newton", False)
            or solver_cfg.get("newton_strict", False)
        )
        self.newton_revision_matches = verify_newton_revision(
            self.newton_revision, strict=strict_newton
        )

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
        with np.load(asset["regions_npz"]) as regions:
            self.inner_vertices = np.asarray(
                regions["inner_coating_vertices"], dtype=np.int32
            )
            self.outer_vertices = np.asarray(
                regions["outer_contact_vertices"], dtype=np.int32
            )
            self.outer_faces = np.asarray(
                regions["outer_contact_faces"], dtype=np.int32
            )
            self.mount_vertices = np.asarray(regions["mount_vertices"], dtype=np.int32)
        with np.load(asset["surface_mapping_npz"]) as mapping_file:
            self.surface_rest_vertices_m = np.asarray(
                mapping_file["surface_rest_vertices_m"], dtype=np.float64
            )
            self.surface_faces = np.asarray(
                mapping_file["surface_faces"], dtype=np.int32
            )
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
        quaternion = normalized_quaternion_xyzw(transform_cfg["quaternion_xyzw"])
        from .indenter import quaternion_rotate_xyzw

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
        requested_clearance_m = float(self.config["trajectory"]["clearance_m"])
        settling_gap_m = max(
            requested_clearance_m,
            float(self.config["contact_parameters"]["margin_m"])
            + 2.0 * self.particle_radius,
        )
        self.config["trajectory"]["requested_clearance_m"] = requested_clearance_m
        self.config["trajectory"]["clearance_m"] = settling_gap_m
        self.trajectory = PrescribedTrajectory(self.config["trajectory"])
        self.touch_runtime_s = self.trajectory.total_duration_s + self.post_recovery_s
        self.indenter_start_position = (
            contact_translation - self.loading_direction * settling_gap_m
        )
        self.indenter_settling_position = self.indenter_start_position.copy()
        self.current_sample = self.trajectory.sample(0.0)
        self.current_body_position = self.indenter_start_position.copy()

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
        for local_index in self.mount_vertices:
            builder.particle_mass[self.particle_start + int(local_index)] = 0.0
        all_local = np.arange(len(local_vertices), dtype=np.int32)
        self.free_particle_local_indices = np.setdiff1d(
            all_local, self.mount_vertices, assume_unique=False
        )
        if not len(self.free_particle_local_indices):
            raise RuntimeError("mount region fixes every tet particle")

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

        deterministic_requested = bool(solver_cfg["deterministic"])
        solver_deterministic_kwargs, solver_supports_determinism = (
            deterministic_constructor_kwargs(SolverVBD, deterministic_requested)
        )
        pipeline_deterministic_kwargs, pipeline_supports_determinism = (
            deterministic_constructor_kwargs(
                newton.CollisionPipeline, deterministic_requested
            )
        )
        self.deterministic_supported = (
            solver_supports_determinism or pipeline_supports_determinism
        )
        self.deterministic_applied = bool(
            solver_deterministic_kwargs or pipeline_deterministic_kwargs
        )
        if deterministic_requested and not self.deterministic_supported:
            print(
                "[mechanics] NOTICE: this pinned Newton API exposes no deterministic "
                "execution option; only the prescribed trajectory is deterministic."
            )

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
            **solver_deterministic_kwargs,
        )
        self.contact_buffer_capacity = int(
            solver_cfg["rigid_body_particle_contact_buffer_size"]
        )
        self.contact_buffer_counts = getattr(
            self.solver, "body_particle_contact_counts", None
        )
        if self.contact_buffer_counts is None:
            raise RuntimeError(
                "Pinned SolverVBD exposes no body_particle_contact_counts; "
                "contact-buffer saturation cannot be monitored safely"
            )
        self.current_contact_buffer_count = 0
        self.maximum_contact_buffer_count = 0
        self.contact_buffer_saturated = False
        self.first_contact_buffer_saturation_frame = -1
        self.first_contact_buffer_saturation_substep = -1
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=float(contact_cfg["margin_m"]),
            **pipeline_deterministic_kwargs,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.cad_rest_particle_positions = self.state_0.particle_q.numpy()[
            self.particle_start : self.particle_end
        ].astype(np.float64)
        self.particle_radii = self.model.particle_radius.numpy().astype(np.float64)

        device = self.model.device
        self.environment = {
            "newton_version": self.newton_version,
            "newton_git_sha": self.newton_revision,
            "supported_newton_version": SUPPORTED_NEWTON_VERSION,
            "supported_newton_git_sha": SUPPORTED_NEWTON_GIT_SHA,
            "newton_revision_matches": self.newton_revision_matches,
            "warp_version": str(getattr(wp, "__version__", "unknown")),
            "device": str(device),
            "device_name": str(getattr(device, "name", "unknown")),
            "mechanics_output_schema_version": MECHANICS_OUTPUT_SCHEMA_VERSION,
            "contact_metric_model": CONTACT_METRIC_MODEL,
            "force_estimator_version": CONTACT_FORCE_ESTIMATOR_VERSION,
            "deterministic_requested": deterministic_requested,
            "deterministic_supported": self.deterministic_supported,
            "deterministic_applied": self.deterministic_applied,
            "repeatability_scope": "prescribed_repeatable_trajectory",
            "simulation_capability": SIMULATION_CAPABILITY,
            "shear_validated": SHEAR_VALIDATED,
            "slip_validated": SLIP_VALIDATED,
            "configured_contact_capacity": self.contact_buffer_capacity,
            "maximum_contact_count_observed": self.maximum_contact_buffer_count,
            "contact_buffer_saturation_flag": self.contact_buffer_saturated,
            "first_contact_buffer_saturation_frame": None,
            "first_contact_buffer_saturation_substep": None,
            "contact_buffer_monitor_source": (
                "SolverVBD.body_particle_contact_counts[indenter_body]"
            ),
            "requested_clearance_m": requested_clearance_m,
            "effective_clearance_m": settling_gap_m,
        }
        self.config["output_metadata"] = {
            key: self.environment[key]
            for key in (
                "newton_version",
                "newton_git_sha",
                "mechanics_output_schema_version",
                "contact_metric_model",
                "force_estimator_version",
                "simulation_capability",
                "shear_validated",
                "slip_validated",
            )
        }
        self._warn_provisional_shear_configuration()

        self.exporter = MechanicalDataExporter(
            args.output_dir or self.config["output"]["directory"], self.config
        )
        _write_json(self.exporter.output_dir / "asset_manifest.json", asset)
        _copy_if_different(
            asset["regions_npz"], self.exporter.output_dir / "regions.npz"
        )
        _copy_if_different(
            asset["surface_mapping_npz"],
            self.exporter.output_dir / "surface_mapping.npz",
        )
        self.environment_path = self.exporter.output_dir / "newton_environment.json"
        self._write_environment()
        self.failure_path = self.exporter.output_dir / "failure_state.npz"

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            camera = self.config.get("viewer", {}).get("camera")
            if camera:
                self.viewer.set_camera(
                    pos=wp.vec3(*camera["position_m"]),
                    pitch=float(camera["pitch_degrees"]),
                    yaw=float(camera["yaw_degrees"]),
                )
        if self.video_enabled:
            video_cfg = self.config["video"]
            self.video_recorder = VideoRecorder(
                video_cfg["path"],
                fps=float(video_cfg["fps"]),
                codec=str(video_cfg["codec"]),
                quality=int(video_cfg["quality"]),
                include_ui=bool(video_cfg["include_ui"]),
            )

        print(
            f"[mechanics] asset={asset.get('asset_id', asset['volume_msh'])} "
            f"particles={self.particle_end - self.particle_start} tets={len(local_tets)} "
            f"mount_vertices={len(self.mount_vertices)} "
            f"particle_radius={self.particle_radius * 1000:.3f} mm"
        )
        print(
            f"[mechanics] trajectory=prescribed_repeatable duration="
            f"{self.trajectory.total_duration_s:.3f} s; settling before contact "
            f"for {self.equilibration_config['minimum_duration_s']}-"
            f"{self.equilibration_config['maximum_duration_s']} s"
        )
        self._export_frame(0.0, "initialization")

    def _configure_video(self, args: Any) -> None:
        video_cfg = self.config["video"]
        explicit_path = getattr(args, "video_path", None)
        if explicit_path:
            video_cfg["enabled"] = True
            video_cfg["path"] = str(Path(explicit_path).resolve())
        elif getattr(args, "record_video", False):
            video_cfg["enabled"] = True
            path = Path(video_cfg["path"])
            if not path.is_absolute():
                experiment_dir = Path(self.config["_experiment_config_path"]).parent
                video_cfg["path"] = str((experiment_dir / path).resolve())

    def _warn_provisional_shear_configuration(self) -> None:
        contact = self.config["contact_parameters"]
        friction = max(
            abs(float(contact["static_friction"])),
            abs(float(contact["dynamic_friction"])),
        )
        slip_enabled = bool(
            self.config["trajectory"].get("lateral_slip", {}).get("enabled", False)
        )
        if friction > 0.0 or slip_enabled:
            warnings.warn(
                "Nonzero friction and lateral slip are provisional experimental "
                "features; shear and stick-slip behavior are not validated.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _write_environment(self) -> None:
        self._refresh_contact_buffer_environment()
        _write_json(self.environment_path, self.environment)

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

    def _command_indenter(self, state, trajectory_time_s: float) -> None:
        sample = self.trajectory.sample(trajectory_time_s)
        if self.lifecycle_phase == "settling":
            position = self.indenter_settling_position
            velocity = np.zeros(3, dtype=np.float64)
        else:
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
            _set_kinematic_pose_v2,
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

    def _refresh_contact_buffer_environment(self) -> None:
        if not hasattr(self, "environment"):
            return
        self.environment.update(
            {
                "configured_contact_capacity": self.contact_buffer_capacity,
                "maximum_contact_count_observed": self.maximum_contact_buffer_count,
                "contact_buffer_saturation_flag": self.contact_buffer_saturated,
                "first_contact_buffer_saturation_frame": (
                    None
                    if self.first_contact_buffer_saturation_frame < 0
                    else self.first_contact_buffer_saturation_frame
                ),
                "first_contact_buffer_saturation_substep": (
                    None
                    if self.first_contact_buffer_saturation_substep < 0
                    else self.first_contact_buffer_saturation_substep
                ),
            }
        )

    def _monitor_contact_buffer(self) -> None:
        observed = int(self.contact_buffer_counts.numpy()[self.indenter_body])
        self.current_contact_buffer_count = observed
        self.maximum_contact_buffer_count = max(
            self.maximum_contact_buffer_count, observed
        )
        if observed < self.contact_buffer_capacity:
            return

        self.contact_buffer_saturated = True
        if self.first_contact_buffer_saturation_substep < 0:
            self.first_contact_buffer_saturation_frame = self.exporter.frame_count
            self.first_contact_buffer_saturation_substep = self.substep
        self._refresh_contact_buffer_environment()
        self._write_environment()
        export_failure_state(
            self.failure_path,
            particle_q=self._particle_positions(),
            sim_time_s=np.float64(self.substep * self.sim_dt),
            trajectory_phase=np.asarray(self._phase_name()),
            failure_reason=np.asarray("contact_buffer_saturation"),
            configured_contact_capacity=np.int32(self.contact_buffer_capacity),
            observed_contact_count=np.int32(observed),
            first_saturation_frame=np.int64(self.first_contact_buffer_saturation_frame),
            first_saturation_substep=np.int64(
                self.first_contact_buffer_saturation_substep
            ),
        )
        self.exporter.finalize()
        raise RuntimeError(
            "Rigid-body particle-contact buffer saturated: "
            f"observed {observed}, capacity {self.contact_buffer_capacity}; "
            "increase rigid_body_particle_contact_buffer_size"
        )

    def simulate(self, *, advance_trajectory: bool) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            if advance_trajectory:
                self.trajectory_substep += 1
                command_time = self.trajectory_substep * self.sim_dt
            else:
                command_time = 0.0
            self._command_indenter(self.state_0, command_time)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.substep += 1
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._monitor_contact_buffer()
            interval = int(self.config["monitoring"]["tet_check_interval_substeps"])
            if self.substep % interval == 0:
                self._check_tet_state()

    def _particle_positions(self) -> np.ndarray:
        return self.state_0.particle_q.numpy()[
            self.particle_start : self.particle_end
        ].astype(np.float64)

    def _relative_tet_volumes(self, particle_positions: np.ndarray) -> np.ndarray:
        current = tet_signed_volumes(particle_positions, self.mapping_tets)
        return current / self.rest_volumes

    def _phase_name(self) -> str:
        if self.lifecycle_phase in {"initialization", "settling", "capture_baseline"}:
            return self.lifecycle_phase
        if self.trajectory_time_s > self.trajectory.total_duration_s + 1.0e-12:
            return "post_recovery"
        return self.trajectory.sample(self.trajectory_time_s).phase

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
            trajectory_phase=np.asarray(self._phase_name()),
            nominal_indentation_m=np.float64(
                self.trajectory.nominal_indentation_m(self.current_sample)
            ),
            newton_version=np.asarray(self.newton_version),
            newton_git_sha=np.asarray(self.newton_revision),
        )
        self.finalize()
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
        contact_cfg = self.config["contact_parameters"]
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
                force_threshold_n=float(contact_cfg["force_threshold_n"]),
                friction_epsilon_m_s=float(contact_cfg["friction_epsilon_m_s"]),
            )
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

    def _export_frame(self, time_s: float, phase: str | None = None) -> None:
        current_global = self.state_0.particle_q.numpy().astype(np.float64)
        previous_global = self.state_1.particle_q.numpy().astype(np.float64)
        particles = current_global[self.particle_start : self.particle_end]
        deformed_surface = reconstruct_surface(
            particles, self.mapping_tets, self.surface_mapping
        )
        relative = self._relative_tet_volumes(particles)
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
        approx_area = masked_triangle_area(
            deformed_surface, self.surface_faces, self.outer_faces, face_mask
        )
        displacement_cad = particles - self.cad_rest_particle_positions
        if self.equilibrated_particle_positions is None:
            displacement_baseline = np.full_like(particles, np.nan)
            max_baseline = float("nan")
        else:
            displacement_baseline = particles - self.equilibrated_particle_positions
            max_baseline = float(
                np.nanmax(np.linalg.norm(displacement_baseline, axis=1))
            )
        max_cad = float(np.nanmax(np.linalg.norm(displacement_cad, axis=1)))
        inverted = int(np.count_nonzero(relative <= 0.0))
        minimum_relative = float(np.nanmin(relative))
        phase = phase or self._phase_name()
        transverse = contact.estimated_transverse_reaction_n
        tangential_velocity = contact.estimated_tangential_relative_velocity_m_s
        axial = float(contact.estimated_axial_reaction_n)
        frame = {
            "mechanics_output_schema_version": np.int32(
                MECHANICS_OUTPUT_SCHEMA_VERSION
            ),
            "timestamp_s": np.float64(time_s),
            "trajectory_time_s": np.float64(self.trajectory_time_s),
            "object_position_m": self.current_body_position.copy(),
            "object_quaternion_xyzw": self.indenter_quaternion.copy(),
            "object_linear_velocity_m_s": self.current_body_velocity.copy(),
            "object_angular_velocity_rad_s": self.current_body_angular_velocity.copy(),
            "tet_particle_positions_m": particles.copy(),
            "deformed_surface_vertices_m": deformed_surface,
            "deformed_inner_coating_vertices_m": deformed_surface[self.inner_vertices],
            "deformed_outer_surface_vertices_m": deformed_surface[self.outer_vertices],
            "displacement_from_cad_rest_m": displacement_cad,
            "displacement_from_equilibrated_baseline_m": displacement_baseline,
            "contact_flag": bool(contact.contact_flag),
            "contact_buffer_configured_capacity": np.int32(
                self.contact_buffer_capacity
            ),
            "contact_buffer_observed_count": np.int32(
                self.current_contact_buffer_count
            ),
            "contact_buffer_maximum_count_observed": np.int32(
                self.maximum_contact_buffer_count
            ),
            "contact_buffer_saturation_flag": bool(self.contact_buffer_saturated),
            "contact_buffer_first_saturation_frame": np.int64(
                self.first_contact_buffer_saturation_frame
            ),
            "contact_buffer_first_saturation_substep": np.int64(
                self.first_contact_buffer_saturation_substep
            ),
            "contact_face_mask": face_mask,
            "approx_contact_area_m2": np.float64(approx_area),
            "estimated_axial_reaction_n": np.float64(axial),
            "estimated_transverse_reaction_n": transverse,
            "estimated_tangential_relative_velocity_m_s": tangential_velocity,
            "maximum_displacement_from_cad_rest_m": np.float64(max_cad),
            "maximum_displacement_from_equilibrated_baseline_m": np.float64(
                max_baseline
            ),
            "minimum_relative_tet_volume": np.float64(minimum_relative),
            "inverted_tet_count": np.int32(inverted),
            "trajectory_phase": phase,
            # Deprecated aliases retained for schema version 2 only.
            "contact_area_m2": np.float64(approx_area),
            "normal_force_n": np.float64(axial),
            "tangential_force_n": transverse,
            "slip_velocity_m_s": tangential_velocity,
            "maximum_displacement_m": np.float64(max_baseline),
        }
        metric = {
            "timestamp_s": time_s,
            "trajectory_time_s": self.trajectory_time_s,
            "contact_buffer_configured_capacity": self.contact_buffer_capacity,
            "contact_buffer_observed_count": self.current_contact_buffer_count,
            "contact_buffer_maximum_count_observed": (
                self.maximum_contact_buffer_count
            ),
            "contact_buffer_saturation_flag": int(self.contact_buffer_saturated),
            "contact_buffer_first_saturation_frame": (
                self.first_contact_buffer_saturation_frame
            ),
            "contact_buffer_first_saturation_substep": (
                self.first_contact_buffer_saturation_substep
            ),
            "trajectory_phase": phase,
            "contact_flag": int(contact.contact_flag),
            "approx_contact_area_m2": approx_area,
            "estimated_axial_reaction_n": axial,
            "estimated_transverse_reaction_x_n": transverse[0],
            "estimated_transverse_reaction_y_n": transverse[1],
            "estimated_transverse_reaction_z_n": transverse[2],
            "estimated_tangential_relative_velocity_x_m_s": tangential_velocity[0],
            "estimated_tangential_relative_velocity_y_m_s": tangential_velocity[1],
            "estimated_tangential_relative_velocity_z_m_s": tangential_velocity[2],
            "maximum_displacement_from_cad_rest_m": max_cad,
            "maximum_displacement_from_equilibrated_baseline_m": max_baseline,
            "minimum_relative_tet_volume": minimum_relative,
            "inverted_tet_count": inverted,
            "max_free_particle_speed_m_s": self.max_free_particle_speed_m_s,
            "equilibration_stable_frames": self.equilibration_stable_frames,
            "contact_area_m2": approx_area,
            "normal_force_n": axial,
            "tangential_force_x_n": transverse[0],
            "tangential_force_y_n": transverse[1],
            "tangential_force_z_n": transverse[2],
            "slip_velocity_x_m_s": tangential_velocity[0],
            "slip_velocity_y_m_s": tangential_velocity[1],
            "slip_velocity_z_m_s": tangential_velocity[2],
            "maximum_displacement_m": max_baseline,
        }
        self.exporter.append(frame, metric)

    def _output_due(self) -> bool:
        if self.sim_time + 1.0e-12 < self.next_output_time + self.output_period:
            return False
        while self.next_output_time + self.output_period <= self.sim_time + 1.0e-12:
            self.next_output_time += self.output_period
        return True

    def _capture_equilibrated_baseline(self) -> None:
        particles = self._particle_positions()
        self.equilibrated_particle_positions = particles.copy()
        self.equilibrated_surface_positions = reconstruct_surface(
            particles, self.mapping_tets, self.surface_mapping
        )
        self.lifecycle_phase = "capture_baseline"
        self.environment["equilibration"] = {
            "converged": self.equilibration_converged,
            "timed_out": self.equilibration_timed_out,
            "duration_s": self.equilibration_elapsed_s,
            "max_free_particle_speed_m_s": self.max_free_particle_speed_m_s,
            "stable_frames": self.equilibration_stable_frames,
        }
        self._write_environment()
        self._export_frame(self.sim_time, "capture_baseline")
        self.next_output_time = self.sim_time

    def _step_settling(self) -> None:
        self.lifecycle_phase = "settling"
        self.simulate(advance_trajectory=False)
        self.sim_time += self.frame_dt
        self.equilibration_elapsed_s += self.frame_dt
        velocities = self.state_0.particle_qd.numpy()[
            self.particle_start : self.particle_end
        ].astype(np.float64)
        free_speeds = np.linalg.norm(
            velocities[self.free_particle_local_indices], axis=1
        )
        self.max_free_particle_speed_m_s = float(np.max(free_speeds))
        minimum = float(self.equilibration_config["minimum_duration_s"])
        tolerance = float(self.equilibration_config["velocity_tolerance_m_s"])
        if (
            self.equilibration_elapsed_s + 1.0e-12 >= minimum
            and np.isfinite(self.max_free_particle_speed_m_s)
            and self.max_free_particle_speed_m_s <= tolerance
        ):
            self.equilibration_stable_frames += 1
        else:
            self.equilibration_stable_frames = 0
        required = int(self.equilibration_config["stable_frames"])
        self.equilibration_converged = self.equilibration_stable_frames >= required
        maximum = float(self.equilibration_config["maximum_duration_s"])
        self.equilibration_timed_out = (
            not self.equilibration_converged
            and self.equilibration_elapsed_s + 1.0e-12 >= maximum
        )
        if self.equilibration_converged or self.equilibration_timed_out:
            if self.equilibration_timed_out:
                message = (
                    "gravity equilibration did not converge before "
                    f"{maximum:g} s; max free-particle speed="
                    f"{self.max_free_particle_speed_m_s:.6g} m/s"
                )
                if self.equilibration_config["timeout_behavior"] == "fail":
                    self._export_frame(self.sim_time, "settling")
                    export_failure_state(
                        self.failure_path,
                        particle_q=self._particle_positions(),
                        sim_time_s=np.float64(self.sim_time),
                        trajectory_phase=np.asarray("settling"),
                        failure_reason=np.asarray("equilibration_timeout"),
                        max_free_particle_speed_m_s=np.float64(
                            self.max_free_particle_speed_m_s
                        ),
                    )
                    self.finalize()
                    raise RuntimeError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            self._capture_equilibrated_baseline()
        elif self._output_due():
            self._export_frame(self.sim_time, "settling")

    def _step_touch(self) -> None:
        self.lifecycle_phase = "touch"
        self.simulate(advance_trajectory=True)
        self.sim_time += self.frame_dt
        self.trajectory_time_s = min(
            self.trajectory_time_s + self.frame_dt, self.touch_runtime_s
        )
        if self._output_due():
            self._export_frame(self.sim_time)
        if self.trajectory_time_s + 1.0e-12 >= self.touch_runtime_s:
            if self.exporter.last_metric is None or not math.isclose(
                float(self.exporter.last_metric["timestamp_s"]), self.sim_time
            ):
                self._export_frame(self.sim_time)
            self.exporter.finalize()
            self.finalized = True
            last = self.exporter.last_metric or {}
            print(
                f"[mechanics] complete frames={self.exporter.frame_count} "
                f"final_phase={last.get('trajectory_phase', 'unknown')} min_J="
                f"{self.exporter.minimum_relative_tet_volume_seen:.6f}"
            )

    def step(self) -> None:
        if self.finalized:
            return
        if self.equilibrated_particle_positions is None:
            self._step_settling()
        else:
            self._step_touch()

    def _close_video(self) -> None:
        if self.video_recorder is None or self.video_recorder.closed:
            return
        self.video_recorder.close()
        print(
            f"[mechanics] video complete frames={self.video_recorder.frame_count} "
            f"path={self.video_recorder.path}"
        )

    def _request_viewer_exit(self) -> None:
        if self.viewer is None:
            return
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None and hasattr(renderer, "app"):
            renderer.app.event_loop.exit()
        elif hasattr(self.viewer, "num_frames") and hasattr(self.viewer, "frame_count"):
            self.viewer.num_frames = self.viewer.frame_count

    def finalize(self) -> None:
        self.exporter.finalize()
        self._close_video()
        self._write_environment()
        self.finalized = True

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if bool(self.config.get("viewer", {}).get("render_contacts", False)):
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
        if self.video_recorder is not None:
            self.video_recorder.capture(self.viewer, self.sim_time)
        if self.finalized:
            self._close_video()
            self._request_viewer_exit()
