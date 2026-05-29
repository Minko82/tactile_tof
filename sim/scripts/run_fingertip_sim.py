# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
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

import warp as wp

import newton
import newton.examples
from newton.solvers import SolverVBD


# ---------------------------------------------------------------------------
# Ecoflex 00-30 mechanical / constitutive parameters
# ---------------------------------------------------------------------------
ECOFLEX_YOUNGS = 50.0e3          # Young's modulus E [Pa]
ECOFLEX_POISSON = 0.40           # Poisson's ratio nu
ECOFLEX_DENSITY = 1070.0         # density rho [kg/m^3]
ECOFLEX_K_DAMP = 1.0e-3          # Rayleigh damping on neo-Hookean elements

# Lame parameters derived from E, nu
#   neo-Hookean strain energy:  W = (mu/2) (I1 - 3) - mu ln(J) + (lambda/2) ln(J)^2
ECOFLEX_MU = ECOFLEX_YOUNGS / (2.0 * (1.0 + ECOFLEX_POISSON))
ECOFLEX_LAMBDA = (
    ECOFLEX_YOUNGS * ECOFLEX_POISSON
    / ((1.0 + ECOFLEX_POISSON) * (1.0 - 2.0 * ECOFLEX_POISSON))
)

# Ecoflex wall thickness / contact geometry (paper design space)
WALL_THICKNESS_M = 0.003         # 3 mm nominal (design-space range 3-5 mm)
INDENTER_RADIUS_M = 0.009        # conforming sphere radius for broader apex contact
INDENTER_DIAM_M = INDENTER_RADIUS_M * 2.0
INDENTER_AREA_M2 = 3.141592653589793 * INDENTER_RADIUS_M**2
MAX_COMPRESSION_M = 0.003        # 0-3 mm expected compression for 0-5 N
LINEAR_STRAIN_LIMIT = 0.30       # neo-Hookean valid to ~30% engineering strain

# ---------------------------------------------------------------------------
# Contact / friction parameters (tuned for Ecoflex against a rigid indenter)
# ---------------------------------------------------------------------------
SOFT_CONTACT_KE = 5.0e4          # stiff, but avoids impulse spikes at the dome apex
# SolverVBD treats kd as a multiplier on contact stiffness:
# damping_coeff = kd * ke. A value like 10.0 would be enormous here.
SOFT_CONTACT_KD = 1.0e-4
STATIC_FRICTION = 1.2            # static mu for silicone vs. smooth rigid surface
DYNAMIC_FRICTION = 0.9           # dynamic mu
SOFT_CONTACT_MARGIN = 0.002      # explicit particle-shape contact generation margin
INDENTER_MAX_SPEED_M_S = 0.2     # safety net for very fast viewer dragging

# Runtime diagnostics. Disable after diagnosing if you need CUDA graph capture
# and maximum interactive frame rate.
ENABLE_INVERSION_DEBUG = True
TIP_DEBUG_PARTICLE_COUNT = 12
TIP_DEBUG_FRAME_INTERVAL = 30
MIN_TET_VOLUME_WARN_M3 = 1.0e-14
TIP_BAND_M = 0.002


@wp.kernel
def _limit_body_linear_velocity(
    body_index: int,
    max_speed: float,
    body_qd: wp.array[wp.spatial_vector],
):
    qd = body_qd[body_index]
    v = wp.spatial_top(qd)
    speed = wp.length(v)

    if speed > max_speed:
        v = v * (max_speed / speed)

    # The indenter is a sphere, so angular velocity has no useful visual or
    # contact effect here; clearing it avoids edge-case rotational energy.
    body_qd[body_index] = wp.spatial_vector(v, wp.vec3(0.0))


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


def _print_rest_mesh_quality(tetmesh, scale):
    volumes_m3 = _tet_signed_volumes_np(tetmesh.vertices, tetmesh.tet_indices, scale)
    abs_volumes = np.abs(volumes_m3)
    finite = np.isfinite(volumes_m3)

    if not finite.any():
        print("[fingertip][mesh] no finite tetrahedron volumes found")
        return

    finite_volumes = volumes_m3[finite]
    finite_abs = abs_volumes[finite]
    negative_count = int(np.count_nonzero(finite_volumes <= 0.0))
    tiny_count = int(np.count_nonzero(finite_abs < MIN_TET_VOLUME_WARN_M3))

    verts_m = np.asarray(tetmesh.vertices, dtype=np.float64) * scale
    tets = np.asarray(tetmesh.tet_indices, dtype=np.int64).reshape(-1, 4)
    z_max = float(np.max(verts_m[:, 2]))
    tip_mask = np.max(verts_m[tets, 2], axis=1) >= z_max - TIP_BAND_M
    tip_abs = abs_volumes[tip_mask & finite]

    print(
        f"[fingertip][mesh] tets={len(volumes_m3)}  "
        f"min signed volume={float(np.min(finite_volumes)):.3e} m^3  "
        f"min |volume|={float(np.min(finite_abs)):.3e} m^3  "
        f"negative={negative_count}  tiny(<{MIN_TET_VOLUME_WARN_M3:.0e})={tiny_count}"
    )

    if tip_abs.size > 0:
        print(
            f"[fingertip][mesh] tip-band tets={int(tip_abs.size)}  "
            f"tip min |volume|={float(np.min(tip_abs)):.3e} m^3"
        )

    if negative_count > 0 or tiny_count > 0:
        print(
            "[fingertip][mesh] WARNING: rest mesh has inverted or tiny tets; "
            "remeshing/refining the apex is likely required for robust contact."
        )


# ---------------------------------------------------------------------------
# Asset + placement
# ---------------------------------------------------------------------------
# The wildmeshing output lives alongside the surface OBJ; it is expressed in
# the same units as the source OBJ (centimetres), so we scale to meters.
ASSET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "fingertip", "fingertip_tet_.msh"
)
FINGERTIP_SCALE = 0.01                                  # cm -> m
FINGERTIP_POS = wp.vec3(0.0, 0.0, 0.0)                  # base flat on the plane
FINGERTIP_ROT = wp.quat_identity()                      # tip points +Z


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
        self.sim_substeps = 30
        self.iterations = 40
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.debug_substep = 0

        # particle / contact sizing (meter scale)
        self.particle_radius = 0.0002  # 0.2 mm; small enough for the dome tip mesh

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_ground_plane()

        # -------------------------------------------------------------------
        # Step 3a: Load the tet mesh and register it as a soft body with
        # Ecoflex neo-Hookean material parameters (Lame form).
        # newton.TetMesh.create_from_file reads .msh/.vtk/.vtu via meshio and
        # .npz natively; see newton/_src/geometry/types.py:1313.
        # -------------------------------------------------------------------
        tetmesh = newton.TetMesh.create_from_file(ASSET_PATH)
        _print_rest_mesh_quality(tetmesh, FINGERTIP_SCALE)

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
        # Step 3b: Anchor the rigid mounting ring at the base of the fingertip.
        # The base plane after the world transform sits at FINGERTIP_POS.z, so
        # we pin every particle within a thin z-slab above that plane. Setting
        # particle_mass = 0 makes those particles kinematic (infinite mass) in
        # VBD. The dome tip above stays free to deform.
        # -------------------------------------------------------------------
        base_z = FINGERTIP_POS[2]
        anchor_slab = 0.0008  # 0.8 mm-thick pinned band at the base
        num_anchored = 0
        for i in range(particle_start, particle_end):
            q = builder.particle_q[i]
            if q[2] <= base_z + anchor_slab:
                builder.particle_mass[i] = 0.0  # zero mass -> fully pinned
                num_anchored += 1

        if num_anchored == 0:
            raise RuntimeError(
                "No particles pinned at the base - check mesh orientation: "
                "the fingertip base should be the z-min side of the tet mesh."
            )

        # -------------------------------------------------------------------
        # Step 3c: Dynamic rigid indenter sphere that can be picked up and
        # dragged by the user (right-click drag in the Newton viewer), or
        # left to fall under gravity onto the dome. The fingertip is ~19 mm
        # tall (source .msh Z-extent 1.9 cm), so the undeformed apex sits
        # at FINGERTIP_POS.z + 0.019 m. A sphere spreads initial apex contact
        # over a smoother patch than a flat box and avoids corner/edge digging
        # while the user drags laterally.
        # -------------------------------------------------------------------
        dome_height_m = 0.019
        dome_apex_z = base_z + dome_height_m

        indenter_radius = INDENTER_RADIUS_M
        # Start the sphere just barely above the apex so it makes contact on
        # the first or second substep with almost zero impact velocity. The
        # fingertip is a hollow shell only ~3 mm thick, so any significant
        # impact energy will tunnel the indenter through the wall before the
        # contact penalty can absorb it.
        plate_clearance = 0.0005            # 0.5 mm above the apex at rest
        indenter_center_z = dome_apex_z + plate_clearance + indenter_radius

        # Small mass keeps impact KE tiny and yields a gentle default load
        # (m*g ~= 0.05 * 9.81 ~= 0.49 N), well inside the paper's 0-5 N
        # operating range. Drag the sphere with right-click to press harder.
        plate_mass = 0.05
        sphere_inertia = (2.0 / 5.0) * plate_mass * indenter_radius**2

        indenter_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, indenter_center_z), wp.quat_identity()
            ),
            mass=plate_mass,
            inertia=wp.mat33(
                sphere_inertia, 0.0, 0.0,
                0.0, sphere_inertia, 0.0,
                0.0, 0.0, sphere_inertia,
            ),
            label="indenter",
            lock_inertia=True,
        )

        builder.add_shape_sphere(
            indenter_body,
            wp.transform_identity(),
            radius=indenter_radius,
        )
        self.indenter_body = indenter_body

        # VBD requires graph coloring before finalize().
        builder.color()

        # -------------------------------------------------------------------
        # Finalize model + configure contact material properties
        # -------------------------------------------------------------------
        self.model = builder.finalize()
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
            particle_enable_tile_solve=False,
            particle_collision_detection_interval=-1
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

        # Worked sensitivity estimate (paper):
        #   For t0=3 mm, E=50 kPa, A=78.5 mm^2: d(d_rep)/dF ~= 1.07 mm/N
        print(
            f"[fingertip] particles={particle_end - particle_start} "
            f"anchored={num_anchored}"
        )
        print(
            f"[fingertip] neo-Hookean: mu={ECOFLEX_MU:.1f} Pa  "
            f"lambda={ECOFLEX_LAMBDA:.1f} Pa  rho={ECOFLEX_DENSITY} kg/m^3"
        )
        print(
            f"[fingertip] wall t={WALL_THICKNESS_M*1000:.1f} mm  "
            f"indenter D={INDENTER_DIAM_M*1000:.1f} mm  "
            f"A={INDENTER_AREA_M2*1e6:.1f} mm^2"
        )
        print(
            f"[fingertip] indenter sphere: mass={plate_mass:.2f} kg  "
            f"radius={indenter_radius*1000:.1f} mm  "
            f"bottom z={indenter_center_z - indenter_radius:.4f} m  "
            f"(apex z={dome_apex_z:.4f} m, clearance={plate_clearance*1000:.1f} mm)"
        )
        print(
            f"[fingertip] contact ke={SOFT_CONTACT_KE:g}  "
            f"kd={SOFT_CONTACT_KD:g}  "
            f"margin={SOFT_CONTACT_MARGIN*1000:.1f} mm  "
            f"substeps={self.sim_substeps}  iterations={self.iterations}  "
            f"max indenter speed={INDENTER_MAX_SPEED_M_S:.2f} m/s"
        )
        print(
            "[fingertip] right-click + drag the blue sphere in the viewer "
            "to press it into the dome, or let gravity do the work."
        )

        self.capture()

    def capture(self):
        if ENABLE_INVERSION_DEBUG:
            self.graph = None
        elif wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Right-click drag forces from the viewer land on state_0.body_f,
            # and the unified VBD solver consumes them for the indenter.
            self.viewer.apply_forces(self.state_0)
            self._limit_indenter_velocity(self.state_0)

            self.collision_pipeline.collide(self.state_0, self.contacts)

            # Unified solve advances both the rigid indenter and the soft
            # fingertip together, including rigid-particle contact coupling.
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self._limit_indenter_velocity(self.state_1)
            self._debug_tet_state(self.state_1)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.debug_substep += 1

    def _limit_indenter_velocity(self, state):
        wp.launch(
            _limit_body_linear_velocity,
            dim=1,
            inputs=[
                self.indenter_body,
                INDENTER_MAX_SPEED_M_S,
            ],
            outputs=[state.body_qd],
            device=self.model.device,
        )

    def _debug_tet_state(self, state):
        if not ENABLE_INVERSION_DEBUG or self._tet_signed_volumes is None:
            return

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

        volumes = self._tet_signed_volumes.numpy()
        finite = np.isfinite(volumes)
        finite_volumes = volumes[finite]
        min_volume = float(np.min(finite_volumes)) if finite_volumes.size else float("nan")
        nonfinite_count = int(volumes.size - finite_volumes.size)
        inverted_count = int(np.count_nonzero(finite_volumes <= 0.0))

        sample_interval = max(1, self.sim_substeps * TIP_DEBUG_FRAME_INTERVAL)
        periodic_sample = self.debug_substep % sample_interval == 0
        bad_state = nonfinite_count > 0 or inverted_count > 0

        if not periodic_sample and not bad_state:
            return

        particle_q = state.particle_q.numpy()
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
            f"min tet volume={min_volume:.3e} m^3  "
            f"inverted={inverted_count}  nonfinite_tets={nonfinite_count}  "
            f"bounds_mm=({bounds_lo[0]*1000:.2f},{bounds_lo[1]*1000:.2f},{bounds_lo[2]*1000:.2f})"
            f"..({bounds_hi[0]*1000:.2f},{bounds_hi[1]*1000:.2f},{bounds_hi[2]*1000:.2f})"
        )
        tip_text = ", ".join(
            f"{int(idx)}:({pos[0]*1000:.2f},{pos[1]*1000:.2f},{pos[2]*1000:.2f})"
            for idx, pos in zip(self.tip_particle_indices[:6], tip_positions[:6], strict=False)
        )
        print(f"[fingertip][debug] top particles mm: {tip_text}")

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
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    example = FingertipController(viewer, args)
    newton.examples.run(example, args)
