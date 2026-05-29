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
INDENTER_DIAM_M = 0.010          # reference indenter diameter (10 mm)
INDENTER_AREA_M2 = 3.141592653589793 * (INDENTER_DIAM_M * 0.5) ** 2  # ~78.5 mm^2
MAX_COMPRESSION_M = 0.003        # 0-3 mm expected compression for 0-5 N
LINEAR_STRAIN_LIMIT = 0.30       # neo-Hookean valid to ~30% engineering strain

# ---------------------------------------------------------------------------
# Contact / friction parameters (tuned for Ecoflex against a rigid indenter)
# ---------------------------------------------------------------------------
SOFT_CONTACT_KE = 1.0e3          # penalty contact stiffness (keeps penetration small)
SOFT_CONTACT_KD = 1.0e1          # contact damping (raised to absorb impact KE)
STATIC_FRICTION = 1.2            # static mu for silicone vs. smooth rigid surface
DYNAMIC_FRICTION = 0.9           # dynamic mu
CONTACT_OFFSET = 0.0001           # small contact offset to ensure robust contact detection at meter scale
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
        # Perf budget per frame = sim_substeps * iterations VBD sweeps. For a
        # quasi-static press against a pinned dome we do not need 600 Hz with
        # 10 iterations; 240 Hz with 5 iterations is substantially faster.
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.iterations = 30
        self.sim_dt = self.frame_dt / self.sim_substeps

        # particle / contact sizing (meter scale)
        self.particle_radius = 0.0005  # 0.5 mm

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_ground_plane()

        # -------------------------------------------------------------------
        # Step 3a: Load the tet mesh and register it as a soft body with
        # Ecoflex neo-Hookean material parameters (Lame form).
        # newton.TetMesh.create_from_file reads .msh/.vtk/.vtu via meshio and
        # .npz natively; see newton/_src/geometry/types.py:1313.
        # -------------------------------------------------------------------
        tetmesh = newton.TetMesh.create_from_file(ASSET_PATH)

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
        # Step 3c: Dynamic rigid indenter plate that can be picked up and
        # dragged by the user (right-click drag in the Newton viewer), or
        # left to fall under gravity onto the dome. The fingertip is ~19 mm
        # tall (source .msh Z-extent 1.9 cm), so the undeformed apex sits
        # at FINGERTIP_POS.z + 0.019 m. We spawn the plate just above the
        # apex (3 mm clearance) so it drops into contact on the first few
        # frames and gives visible compression without exploding the solver.
        # -------------------------------------------------------------------
        dome_height_m = 0.019
        dome_apex_z = base_z + dome_height_m

        plate_hx = 0.020                    # 40 mm wide
        plate_hy = 0.020                    # 40 mm long
        plate_hz = 0.002                    # 4 mm thick
        # Start the plate just barely above the apex so it makes contact on
        # the first or second substep with almost zero impact velocity. The
        # fingertip is a hollow shell only ~3 mm thick, so any significant
        # impact energy will tunnel the plate through the wall before the
        # contact penalty can absorb it.
        plate_clearance = 0.0005            # 0.5 mm above the apex at rest
        plate_center_z = dome_apex_z + plate_clearance + plate_hz

        # Small mass keeps impact KE tiny and yields a gentle default load
        # (m*g ~= 0.05 * 9.81 ~= 0.49 N), well inside the paper's 0-5 N
        # operating range. Drag the plate with right-click to press harder.
        plate_mass = 0.05

        indenter_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, plate_center_z), wp.quat_identity()
            ),
            mass=plate_mass,
            inertia=wp.mat33(
                (plate_mass / 3.0) * (plate_hy**2 + plate_hz**2), 0.0, 0.0,
                0.0, (plate_mass / 3.0) * (plate_hx**2 + plate_hz**2), 0.0,
                0.0, 0.0, (plate_mass / 3.0) * (plate_hx**2 + plate_hy**2),
            ),
            label="indenter",
            lock_inertia=True,
        )

        builder.add_shape_box(
            indenter_body,
            wp.transform_identity(),
            hx=plate_hx,
            hy=plate_hy,
            hz=plate_hz,
        )
        self.indenter_body = indenter_body

        # VBD requires graph coloring before finalize().
        builder.color()

        # -------------------------------------------------------------------
        # Finalize model + configure contact material properties
        # -------------------------------------------------------------------
        self.model = builder.finalize()

        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = DYNAMIC_FRICTION
        self.model.soft_contact_offset = CONTACT_OFFSET
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
        self.contacts = self.model.contacts()

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
            f"[fingertip] indenter plate: mass={plate_mass:.2f} kg  "
            f"bottom z={plate_center_z - plate_hz:.4f} m  "
            f"(apex z={dome_apex_z:.4f} m, clearance={plate_clearance*1000:.1f} mm)"
        )
        print(
            "[fingertip] right-click + drag the blue plate in the viewer "
            "to press it into the dome, or let gravity do the work."
        )

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
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

            self.model.collide(self.state_0, self.contacts)

            # Unified solve advances both the rigid indenter and the soft
            # fingertip together, including rigid-particle contact coupling.
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

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
