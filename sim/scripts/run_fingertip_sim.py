# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# LEGACY ECOFLEX REGRESSION RUNNER — NOT THE ACTIVE TOUCHIQ MATERIAL
#
# The configuration-driven runners use SORTA-Clear 37. This file is retained
# only so historical Ecoflex mechanics experiments remain reproducible.
#
# Ecoflex Fingertip Soft-Body Simulation (Step 3 validation)
#
# Loads the wildmeshing-generated tetrahedral mesh for the silicone fingertip,
# wraps it with Ecoflex 00-30 neo-Hookean material parameters, anchors the
# base mounting ring to the reference plane, and presses a flat virtual plate
# down onto the dome apex to visually verify deformation behaviour before the
# sim-to-real calibration step.
#
# Orientation convention: the fingertip base sits flat on the base plane
# (z = 0) and the dome tip points upward (+Z). The rigid indenter comes down
# from above.
#
# Reference material properties (Ecoflex 00-30):
#   Young's modulus   E       = 50 kPa    (valid range 30-70 kPa)
#   Poisson's ratio   nu      = 0.47      (range 0.45-0.49, nearly incompressible)
#   Shear modulus     mu      = E / (2(1+nu))                  ~= 17.0 kPa
#   First Lame        lambda  = E*nu / ((1+nu)(1-2nu))         ~= 266 kPa
#   Density           rho     = 1070 kg/m^3
#   Elongation at break       > 800%
#
# Constitutive model: neo-Hookean, valid up to ~30% engineering strain
# (~0.9 mm compression for a 3 mm slab). Switch to Ogden 2-parameter once
# calibration data is available to capture strain stiffening beyond 30%.
#
# The simulation runs in meter scale.
###########################################################################

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

# Prefer the in-tree Newton source checkout at sim/newton/ over any other
# `newton` package that might be on the system path. The script lives at
# sim/scripts/run_fingertip_sim.py, so the repo root is ../newton/ and its
# package dir is ../newton/newton/.
_LOCAL_NEWTON_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "newton")
)
if os.path.isdir(os.path.join(_LOCAL_NEWTON_ROOT, "newton")):
    sys.path.insert(0, _LOCAL_NEWTON_ROOT)

import newton
import newton.examples
import warp as wp
from newton.solvers import SolverVBD

# ---------------------------------------------------------------------------
# Ecoflex 00-30 mechanical / constitutive parameters
# ---------------------------------------------------------------------------
ECOFLEX_YOUNGS = 50.0e3  # Young's modulus E [Pa]
ECOFLEX_POISSON = 0.40  # Poisson's ratio nu
ECOFLEX_DENSITY = 1070.0  # density rho [kg/m^3]

# Newton commit 8baee876 (the in-tree checkout when this configuration was
# written) uses stiffness-relative, dimensionless damping in SolverVBD. Set
# this flag deliberately when updating Newton: newer checkouts may instead use
# Pa*s for tet damping and N*s/m for contact damping.
NEWTON_USES_ABSOLUTE_DAMPING = False
ECOFLEX_K_DAMP = 50.0 if NEWTON_USES_ABSOLUTE_DAMPING else 1.0e-3

# Lame parameters derived from E, nu
#   neo-Hookean strain energy:  W = (mu/2) (I1 - 3) - mu ln(J) + (lambda/2) ln(J)^2
ECOFLEX_MU = ECOFLEX_YOUNGS / (2.0 * (1.0 + ECOFLEX_POISSON))
ECOFLEX_LAMBDA = (
    ECOFLEX_YOUNGS
    * ECOFLEX_POISSON
    / ((1.0 + ECOFLEX_POISSON) * (1.0 - 2.0 * ECOFLEX_POISSON))
)

# Ecoflex wall thickness / contact geometry (paper design space)
WALL_THICKNESS_M = 0.003  # 3 mm nominal (design-space range 3-5 mm)
INDENTER_RADIUS_M = 0.009  # conforming sphere radius for broader apex contact
INDENTER_DIAM_M = INDENTER_RADIUS_M * 2.0
INDENTER_AREA_M2 = 3.141592653589793 * INDENTER_RADIUS_M**2
MAX_COMPRESSION_M = 0.00075  # hard limit: 25% of the nominal 3 mm wall
LINEAR_STRAIN_LIMIT = 0.30  # neo-Hookean valid to ~30% engineering strain

# ---------------------------------------------------------------------------
# Contact / friction parameters (tuned for Ecoflex against a rigid indenter)
# ---------------------------------------------------------------------------
SOFT_CONTACT_KE = 1.0e4  # conservative starting point for normal indentation
SOFT_CONTACT_KD = 1.0 if NEWTON_USES_ABSOLUTE_DAMPING else 1.0e-4
STATIC_FRICTION = 0.0  # add friction only after normal contact is stable
DYNAMIC_FRICTION = 0.0
SOFT_CONTACT_MARGIN = 0.0005
INDENTER_SPEED_M_S = 0.002  # deterministic 2 mm/s calibration trajectory
INDENTER_MAX_SPEED_M_S = 0.02  # configuration guard for future trajectories
INITIAL_CONTACT_CLEARANCE_M = 0.0005

# Runtime diagnostics. Disable after diagnosing if you need CUDA graph capture
# and maximum interactive frame rate.
ENABLE_INVERSION_DEBUG = True
SIM_SUBSTEPS = 20
VBD_ITERATIONS = 15
TET_CHECK_INTERVAL_SUBSTEPS = 5
RENDER_CONTACTS = False
TIP_DEBUG_PARTICLE_COUNT = 12
TIP_DEBUG_FRAME_INTERVAL = 30
TIP_BAND_M = 0.002
MIN_REST_TET_VOLUME_M3 = 1.0e-18
MAX_REST_TET_CONDITION_NUMBER = 100.0
MIN_RELATIVE_TET_VOLUME = 0.15
MOUNT_PLANE_TOLERANCE_M = 1.0e-5
PARTICLE_RADIUS_EDGE_FRACTION = 0.35
FAILURE_STATE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "output", "fingertip_failure_state.npz"
    )
)


@wp.kernel
def _set_kinematic_body_pose(
    body_index: int,
    center_x: float,
    center_y: float,
    center_z: float,
    velocity_z: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    body_q[body_index] = wp.transform(
        wp.vec3(center_x, center_y, center_z),
        wp.quat_identity(),
    )
    body_qd[body_index] = wp.spatial_vector(
        wp.vec3(0.0, 0.0, velocity_z),
        wp.vec3(0.0),
    )


@wp.kernel
def _compute_tet_signed_volumes(
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    signed_volumes: wp.array[float],
):
    tid = wp.tid()

    p0 = particle_q[tet_indices[tid, 0]]
    p1 = particle_q[tet_indices[tid, 1]]
    p2 = particle_q[tet_indices[tid, 2]]
    p3 = particle_q[tet_indices[tid, 3]]

    signed_volumes[tid] = wp.dot(p1 - p0, wp.cross(p2 - p0, p3 - p0)) / 6.0


def _tet_signed_volumes_np(vertices, tet_indices, scale):
    verts = np.asarray(vertices, dtype=np.float64) * scale
    tets = np.asarray(tet_indices, dtype=np.int64).reshape(-1, 4)

    p0 = verts[tets[:, 0]]
    p1 = verts[tets[:, 1]]
    p2 = verts[tets[:, 2]]
    p3 = verts[tets[:, 3]]

    return np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)) / 6.0


def _boundary_faces_np(tets):
    faces = np.concatenate(
        (
            tets[:, (1, 2, 3)],
            tets[:, (0, 3, 2)],
            tets[:, (0, 1, 3)],
            tets[:, (0, 2, 1)],
        ),
        axis=0,
    )
    canonical_faces = np.sort(faces, axis=1)
    _, first_indices, counts = np.unique(
        canonical_faces,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    return faces[first_indices[counts == 1]]


def _validate_rest_mesh(tetmesh, scale):
    verts_m = np.asarray(tetmesh.vertices, dtype=np.float64) * scale
    tets = np.asarray(tetmesh.tet_indices, dtype=np.int64).reshape(-1, 4)

    if not np.isfinite(verts_m).all():
        raise RuntimeError("Rest mesh contains nonfinite vertex coordinates")
    if tets.size == 0:
        raise RuntimeError("Rest mesh contains no tetrahedra")
    if np.min(tets) < 0 or np.max(tets) >= len(verts_m):
        raise RuntimeError("Rest mesh contains out-of-range tetrahedron indices")

    volumes_m3 = _tet_signed_volumes_np(tetmesh.vertices, tetmesh.tet_indices, scale)
    abs_volumes = np.abs(volumes_m3)
    finite = np.isfinite(volumes_m3)

    if not finite.all():
        raise RuntimeError(
            f"Rest mesh contains {int(np.count_nonzero(~finite))} nonfinite tetrahedron volumes"
        )
    effectively_zero = np.flatnonzero(abs_volumes < MIN_REST_TET_VOLUME_M3)
    if effectively_zero.size:
        raise RuntimeError(
            f"Rest mesh contains {effectively_zero.size} effectively zero-volume tetrahedra"
        )

    negative_count = int(np.count_nonzero(volumes_m3 < 0.0))
    if negative_count:
        raise RuntimeError(
            f"Rest mesh contains {negative_count} negatively wound tetrahedra; "
            "Newton's soft-mesh builder requires positive rest volumes"
        )

    p0 = verts_m[tets[:, 0]]
    dm = np.stack(
        (
            verts_m[tets[:, 1]] - p0,
            verts_m[tets[:, 2]] - p0,
            verts_m[tets[:, 3]] - p0,
        ),
        axis=-1,
    )
    condition_numbers = np.linalg.cond(dm)
    severe_slivers = np.flatnonzero(
        (~np.isfinite(condition_numbers))
        | (condition_numbers > MAX_REST_TET_CONDITION_NUMBER)
    )
    if severe_slivers.size:
        raise RuntimeError(
            f"Rest mesh contains {severe_slivers.size} tetrahedra with condition number "
            f"> {MAX_REST_TET_CONDITION_NUMBER:g}"
        )

    boundary_faces = _boundary_faces_np(tets)
    if boundary_faces.size == 0:
        raise RuntimeError("Rest mesh has no boundary faces")

    boundary_edges = np.concatenate(
        (
            boundary_faces[:, (0, 1)],
            boundary_faces[:, (1, 2)],
            boundary_faces[:, (2, 0)],
        ),
        axis=0,
    )
    boundary_edges = np.unique(np.sort(boundary_edges, axis=1), axis=0)
    boundary_edge_lengths = np.linalg.norm(
        verts_m[boundary_edges[:, 0]] - verts_m[boundary_edges[:, 1]],
        axis=1,
    )
    median_boundary_edge_m = float(np.median(boundary_edge_lengths))

    base_z = float(np.min(verts_m[:, 2]))
    mount_face_mask = (
        np.max(verts_m[boundary_faces, 2], axis=1) <= base_z + MOUNT_PLANE_TOLERANCE_M
    )
    mount_faces = boundary_faces[mount_face_mask]
    if mount_faces.size == 0:
        raise RuntimeError(
            "No bottom mounting faces found; check mesh orientation and mount tolerance"
        )
    mount_vertex_indices = np.unique(mount_faces)

    z_max = float(np.max(verts_m[:, 2]))
    tip_mask = np.max(verts_m[tets, 2], axis=1) >= z_max - TIP_BAND_M
    tip_abs = abs_volumes[tip_mask]

    print(
        f"[fingertip][mesh] tets={len(volumes_m3)}  "
        f"min signed volume={float(np.min(volumes_m3)):.3e} m^3  "
        f"min |volume|={float(np.min(abs_volumes)):.3e} m^3  "
        f"negative={negative_count}"
    )
    print(
        f"[fingertip][mesh] condition number median={float(np.median(condition_numbers)):.2f}  "
        f"p99={float(np.quantile(condition_numbers, 0.99)):.2f}  "
        f"max={float(np.max(condition_numbers)):.2f}"
    )
    print(
        f"[fingertip][mesh] tip-band tets={int(tip_abs.size)}  "
        f"tip min |volume|={float(np.min(tip_abs)):.3e} m^3  "
        f"median boundary edge={median_boundary_edge_m * 1000.0:.3f} mm  "
        f"mount faces={len(mount_faces)}  mount vertices={len(mount_vertex_indices)}"
    )

    return (
        volumes_m3.astype(np.float64),
        tets.astype(np.int32),
        mount_vertex_indices.astype(np.int32),
        median_boundary_edge_m,
    )


def _get_newton_revision():
    try:
        result = subprocess.run(
            ["git", "-C", _LOCAL_NEWTON_ROOT, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


# ---------------------------------------------------------------------------
# Asset + placement
# ---------------------------------------------------------------------------
# The wildmeshing output lives alongside the surface OBJ; it is expressed in
# the same units as the source OBJ (centimetres), so we scale to meters.
ASSET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "fingertip", "fingertip_tet_.msh"
)
FINGERTIP_SCALE = 0.01  # cm -> m
FINGERTIP_POS = wp.vec3(0.0, 0.0, 0.0)  # base flat on the plane
FINGERTIP_ROT = wp.quat_identity()  # tip points +Z


class FingertipController:
    def __init__(self, viewer, args=None):
        # --- simulation timing (meter scale) -------------------------------
        # Perf budget per frame = sim_substeps * iterations VBD sweeps. The
        # wall is only a few millimeters thick, so smaller substeps are more
        # important here than raw frame throughput.
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SIM_SUBSTEPS
        self.iterations = VBD_ITERATIONS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.debug_substep = 0

        if MAX_COMPRESSION_M > WALL_THICKNESS_M * LINEAR_STRAIN_LIMIT:
            raise ValueError(
                "MAX_COMPRESSION_M exceeds the configured engineering-strain limit"
            )
        if INDENTER_SPEED_M_S > INDENTER_MAX_SPEED_M_S:
            raise ValueError("INDENTER_SPEED_M_S exceeds INDENTER_MAX_SPEED_M_S")

        # -------------------------------------------------------------------
        # Step 3a: Load the tet mesh and register it as a soft body with
        # Ecoflex neo-Hookean material parameters (Lame form).
        # newton.TetMesh.create_from_file reads .msh/.vtk/.vtu via meshio and
        # .npz natively; see newton/_src/geometry/types.py:1313.
        # -------------------------------------------------------------------
        tetmesh = newton.TetMesh.create_from_file(ASSET_PATH)
        (
            self._rest_tet_volumes_np,
            local_tet_indices_np,
            mount_vertex_indices_np,
            median_boundary_edge_m,
        ) = _validate_rest_mesh(tetmesh, FINGERTIP_SCALE)

        # Use the actual boundary resolution rather than a fixed radius. The
        # previous 0.2 mm radius was much smaller than the surface edge spacing
        # and allowed rigid contact to be detected only after deep penetration.
        self.particle_radius = PARTICLE_RADIUS_EDGE_FRACTION * median_boundary_edge_m

        # The base is already pinned, so a ground plane would only create a
        # redundant layer of persistent contacts around the mounting surface.
        builder = newton.ModelBuilder(gravity=-9.81)

        particle_start = len(builder.particle_q)

        builder.add_soft_mesh(
            pos=FINGERTIP_POS,
            rot=FINGERTIP_ROT,
            scale=FINGERTIP_SCALE,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,
            density=ECOFLEX_DENSITY,
            k_mu=ECOFLEX_MU,
            k_lambda=ECOFLEX_LAMBDA,
            k_damp=ECOFLEX_K_DAMP,
            particle_radius=self.particle_radius,
        )

        particle_end = len(builder.particle_q)
        self._tet_indices_np = local_tet_indices_np + particle_start
        if len(self._rest_tet_volumes_np) != self._tet_indices_np.shape[0]:
            raise RuntimeError("Rest-volume and tetrahedron-index counts do not match")

        particle_indices_np = np.arange(particle_start, particle_end, dtype=np.int32)
        particle_pos_np = np.asarray(
            [
                [
                    float(builder.particle_q[i][0]),
                    float(builder.particle_q[i][1]),
                    float(builder.particle_q[i][2]),
                ]
                for i in range(particle_start, particle_end)
            ],
            dtype=np.float64,
        )
        tip_order = np.argsort(particle_pos_np[:, 2])[-TIP_DEBUG_PARTICLE_COUNT:][::-1]
        self.tip_particle_indices = particle_indices_np[tip_order]

        # -------------------------------------------------------------------
        # Step 3b: Anchor the actual bottom mounting surface. Selecting boundary
        # faces avoids the abrupt, arbitrary 0.8 mm z-slab that previously cut
        # through tetrahedra and concentrated strain at its first free ring.
        # -------------------------------------------------------------------
        anchored_particle_indices = mount_vertex_indices_np + particle_start
        for particle_index in anchored_particle_indices:
            builder.particle_mass[int(particle_index)] = 0.0
        num_anchored = len(anchored_particle_indices)

        # -------------------------------------------------------------------
        # Step 3c: Kinematic rigid sphere following a prescribed, speed-limited
        # normal-indentation trajectory. This makes the loading repeatable and
        # prevents viewer mouse springs from injecting an unbounded wrench.
        # -------------------------------------------------------------------
        dome_apex_z = float(np.max(particle_pos_np[:, 2]))
        apex_mask = particle_pos_np[:, 2] >= dome_apex_z - TIP_BAND_M
        apex_positions = particle_pos_np[apex_mask]
        if apex_positions.size == 0:
            raise RuntimeError("No particles found in the fingertip apex band")
        self.indenter_center_x = float(np.mean(apex_positions[:, 0]))
        self.indenter_center_y = float(np.mean(apex_positions[:, 1]))

        indenter_radius = INDENTER_RADIUS_M
        effective_contact_z = dome_apex_z + self.particle_radius
        self.indenter_start_center_z = (
            effective_contact_z + INITIAL_CONTACT_CLEARANCE_M + indenter_radius
        )
        self.indenter_max_travel_m = INITIAL_CONTACT_CLEARANCE_M + MAX_COMPRESSION_M
        self.current_nominal_penetration_m = 0.0
        self.current_indenter_speed_m_s = 0.0

        indenter_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(
                    self.indenter_center_x,
                    self.indenter_center_y,
                    self.indenter_start_center_z,
                ),
                wp.quat_identity(),
            ),
            label="indenter",
            is_kinematic=True,
        )

        indenter_cfg = newton.ModelBuilder.ShapeConfig()
        indenter_cfg.density = 0.0
        indenter_cfg.ke = SOFT_CONTACT_KE
        indenter_cfg.kd = SOFT_CONTACT_KD
        indenter_cfg.mu = STATIC_FRICTION
        builder.add_shape_sphere(
            indenter_body,
            wp.transform_identity(),
            radius=indenter_radius,
            cfg=indenter_cfg,
        )
        self.indenter_body = indenter_body

        # VBD requires graph coloring before finalize().
        builder.color()

        # -------------------------------------------------------------------
        # Finalize model + configure contact material properties
        # -------------------------------------------------------------------
        self.model = builder.finalize()
        if self.model.tet_count != len(self._rest_tet_volumes_np):
            raise RuntimeError(
                "Finalized model tetrahedron count differs from the validated rest mesh"
            )
        self._tet_signed_volumes = None
        if ENABLE_INVERSION_DEBUG and self.model.tet_count > 0:
            self._tet_signed_volumes = wp.empty(
                self.model.tet_count,
                dtype=float,
                device=self.model.device,
            )

        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = DYNAMIC_FRICTION
        self.model.shape_material_ke.fill_(SOFT_CONTACT_KE)
        self.model.shape_material_kd.fill_(SOFT_CONTACT_KD)
        self.model.shape_material_mu.fill_(STATIC_FRICTION)

        # Unified VBD/AVBD solver: neo-Hookean FEM for the tet mesh plus
        # rigid-body integration/contact for the indenter in the same step.
        self.solver = SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=True,
            particle_collision_detection_interval=-1,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=SOFT_CONTACT_MARGIN,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.08, 0.08, 0.04), -135.0, -20.0)
        self.newton_version = str(getattr(newton, "__version__", "unknown"))
        self.newton_revision = _get_newton_revision()
        damping_semantics = (
            "absolute (Pa*s, N*s/m)"
            if NEWTON_USES_ABSOLUTE_DAMPING
            else "stiffness-relative (dimensionless)"
        )

        # Worked sensitivity estimate (paper):
        #   For t0=3 mm, E=50 kPa, A=78.5 mm^2: d(d_rep)/dF ~= 1.07 mm/N
        print(
            "[material] LEGACY Ecoflex 00-30 regression runner - "
            "not the active TouchIQ material"
        )
        print(
            f"[fingertip] particles={particle_end - particle_start} "
            f"anchored={num_anchored}"
        )
        print(
            f"[fingertip] neo-Hookean: mu={ECOFLEX_MU:.1f} Pa  "
            f"lambda={ECOFLEX_LAMBDA:.1f} Pa  rho={ECOFLEX_DENSITY} kg/m^3"
        )
        print(
            f"[fingertip] wall t={WALL_THICKNESS_M * 1000:.1f} mm  "
            f"indenter D={INDENTER_DIAM_M * 1000:.1f} mm  "
            f"A={INDENTER_AREA_M2 * 1e6:.1f} mm^2"
        )
        print(
            f"[fingertip] kinematic indenter sphere: "
            f"radius={indenter_radius * 1000:.1f} mm  "
            f"center xy=({self.indenter_center_x * 1000:.2f},"
            f"{self.indenter_center_y * 1000:.2f}) mm  "
            f"start bottom z={self.indenter_start_center_z - indenter_radius:.4f} m  "
            f"(apex z={dome_apex_z:.4f} m, particle radius={self.particle_radius * 1000:.3f} mm)"
        )
        print(
            f"[fingertip] contact ke={SOFT_CONTACT_KE:g}  "
            f"kd={SOFT_CONTACT_KD:g}  "
            f"margin={SOFT_CONTACT_MARGIN * 1000:.1f} mm  "
            f"substeps={self.sim_substeps}  iterations={self.iterations}  "
            f"tet_check_every={TET_CHECK_INTERVAL_SUBSTEPS} substeps  "
            f"trajectory={INDENTER_SPEED_M_S * 1000:.1f} mm/s to "
            f"{MAX_COMPRESSION_M * 1000:.2f} mm"
        )
        print(
            f"[fingertip] Newton version={self.newton_version}  "
            f"commit={self.newton_revision}  package={newton.__file__}  "
            f"damping={damping_semantics}"
        )
        print(
            f"[fingertip] inversion circuit breaker: J < {MIN_RELATIVE_TET_VOLUME:.2f}"
        )

        self.capture()

    def capture(self):
        # The prescribed trajectory advances from a Python-side substep count,
        # and the circuit breaker copies diagnostics to the host. Keep this
        # validation executable out of CUDA graph capture.
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self._update_kinematic_indenter(self.state_0)

            self.collision_pipeline.collide(self.state_0, self.contacts)

            # Unified solve advances both the rigid indenter and the soft
            # fingertip together, including rigid-particle contact coupling.
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.debug_substep += 1

            should_check_tets = (
                ENABLE_INVERSION_DEBUG
                and self.debug_substep % TET_CHECK_INTERVAL_SUBSTEPS == 0
            )
            if should_check_tets and not self._debug_tet_state(self.state_1):
                raise FloatingPointError(
                    "Fingertip tet compression/inversion threshold reached; "
                    f"state saved to {FAILURE_STATE_PATH}"
                )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def _update_kinematic_indenter(self, state):
        elapsed_s = self.debug_substep * self.sim_dt
        travel_m = min(
            self.indenter_max_travel_m,
            INDENTER_SPEED_M_S * elapsed_s,
        )
        is_moving = travel_m < self.indenter_max_travel_m
        velocity_z = -INDENTER_SPEED_M_S if is_moving else 0.0
        center_z = self.indenter_start_center_z - travel_m

        self.current_nominal_penetration_m = max(
            0.0,
            travel_m - INITIAL_CONTACT_CLEARANCE_M,
        )
        self.current_indenter_speed_m_s = abs(velocity_z)

        wp.launch(
            _set_kinematic_body_pose,
            dim=1,
            inputs=[
                self.indenter_body,
                self.indenter_center_x,
                self.indenter_center_y,
                center_z,
                velocity_z,
            ],
            outputs=[state.body_q, state.body_qd],
            device=self.model.device,
        )

    def _debug_tet_state(self, state):
        if not ENABLE_INVERSION_DEBUG or self._tet_signed_volumes is None:
            return True

        wp.launch(
            _compute_tet_signed_volumes,
            dim=self.model.tet_count,
            inputs=[
                self.model.tet_indices,
                state.particle_q,
            ],
            outputs=[self._tet_signed_volumes],
            device=self.model.device,
        )

        current_volumes = self._tet_signed_volumes.numpy().astype(
            np.float64,
            copy=False,
        )
        relative_j = current_volumes / self._rest_tet_volumes_np
        bad_tet_indices = np.flatnonzero(
            (~np.isfinite(relative_j))
            | (relative_j <= 0.0)
            | (relative_j < MIN_RELATIVE_TET_VOLUME)
        )
        finite_relative_j = relative_j[np.isfinite(relative_j)]
        min_relative_j = (
            float(np.min(finite_relative_j)) if finite_relative_j.size else float("nan")
        )

        sample_interval = max(1, self.sim_substeps * TIP_DEBUG_FRAME_INTERVAL)
        periodic_sample = self.debug_substep % sample_interval == 0
        if not periodic_sample and bad_tet_indices.size == 0:
            return True

        particle_q = state.particle_q.numpy()
        if bad_tet_indices.size:
            first_bad = bad_tet_indices[:8]
            bad_centroids = particle_q[self._tet_indices_np[first_bad]].mean(axis=1)
            os.makedirs(os.path.dirname(FAILURE_STATE_PATH), exist_ok=True)
            np.savez_compressed(
                FAILURE_STATE_PATH,
                particle_q=particle_q,
                current_tet_volumes=current_volumes,
                rest_tet_volumes=self._rest_tet_volumes_np,
                relative_j=relative_j,
                bad_tet_indices=bad_tet_indices,
                bad_tet_centroids=bad_centroids,
                relative_j_threshold=np.float64(MIN_RELATIVE_TET_VOLUME),
                substep=np.int64(self.debug_substep),
                sim_time_s=np.float64(self.debug_substep * self.sim_dt),
                indenter_speed_m_s=np.float64(self.current_indenter_speed_m_s),
                nominal_penetration_m=np.float64(self.current_nominal_penetration_m),
                newton_version=np.asarray(self.newton_version),
                newton_revision=np.asarray(self.newton_revision),
            )
            print(
                "[fingertip][fatal] "
                f"substep={self.debug_substep}  "
                f"min relative volume J={min_relative_j:.6f}  "
                f"bad tets={bad_tet_indices.size}  "
                f"penetration={self.current_nominal_penetration_m * 1000.0:.3f} mm  "
                f"first centroids={bad_centroids * 1000.0} mm"
            )
            return False

        tip_positions = particle_q[self.tip_particle_indices]
        finite_particles = np.isfinite(particle_q).all(axis=1)
        if finite_particles.any():
            bounds_lo = np.min(particle_q[finite_particles], axis=0)
            bounds_hi = np.max(particle_q[finite_particles], axis=0)
        else:
            bounds_lo = np.array([np.nan, np.nan, np.nan])
            bounds_hi = np.array([np.nan, np.nan, np.nan])

        print(
            f"[fingertip][debug] substep={self.debug_substep}  "
            f"min relative volume J={min_relative_j:.6f}  "
            f"penetration={self.current_nominal_penetration_m * 1000.0:.3f} mm  "
            f"bounds_mm=({bounds_lo[0] * 1000:.2f},{bounds_lo[1] * 1000:.2f},{bounds_lo[2] * 1000:.2f})"
            f"..({bounds_hi[0] * 1000:.2f},{bounds_hi[1] * 1000:.2f},{bounds_hi[2] * 1000:.2f})"
        )
        tip_text = ", ".join(
            f"{int(idx)}:({pos[0] * 1000:.2f},{pos[1] * 1000:.2f},{pos[2] * 1000:.2f})"
            for idx, pos in zip(
                self.tip_particle_indices[:6], tip_positions[:6], strict=False
            )
        )
        print(f"[fingertip][debug] top particles mm: {tip_text}")
        return True

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if RENDER_CONTACTS:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    example = FingertipController(viewer, args)
    newton.examples.run(example, args)
