"""Host-side Newton soft-contact reaction and surface contact metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactSummary:
    contact_flag: bool
    active_contact_indices: np.ndarray
    active_particle_indices: np.ndarray
    particle_forces_n: np.ndarray
    normal_force_n: float
    tangential_force_n: np.ndarray
    slip_velocity_m_s: np.ndarray


def quaternion_rotate_xyzw(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    xyz = q[:3]
    w = q[3]
    return v + 2.0 * np.cross(xyz, np.cross(xyz, v) + w * v)


def estimate_contact_summary(
    *,
    particle_positions_m: np.ndarray,
    previous_particle_positions_m: np.ndarray,
    particle_radii_m: np.ndarray,
    contact_particles: np.ndarray,
    contact_normals: np.ndarray,
    contact_body_positions_m: np.ndarray,
    penalty_stiffness_n_m: np.ndarray,
    damping_ratio: np.ndarray,
    friction_coefficients: np.ndarray,
    body_position_m: np.ndarray,
    body_quaternion_xyzw: np.ndarray,
    body_linear_velocity_m_s: np.ndarray,
    body_angular_velocity_rad_s: np.ndarray,
    loading_direction: np.ndarray,
    dt_s: float,
    force_threshold_n: float,
    friction_epsilon_m_s: float,
) -> ContactSummary:
    """Reproduce Newton VBD's particle/rigid penalty force on the host.

    Contact candidates inside the collision margin are not considered contact
    unless the computed reaction exceeds ``force_threshold_n``.
    """

    particles = np.asarray(contact_particles, dtype=np.int64)
    if len(particles) == 0:
        return ContactSummary(
            contact_flag=False,
            active_contact_indices=np.empty(0, dtype=np.int32),
            active_particle_indices=np.empty(0, dtype=np.int32),
            particle_forces_n=np.zeros((0, 3), dtype=np.float64),
            normal_force_n=0.0,
            tangential_force_n=np.zeros(3, dtype=np.float64),
            slip_velocity_m_s=np.zeros(3, dtype=np.float64),
        )
    positions = np.asarray(particle_positions_m, dtype=np.float64)[particles]
    previous = np.asarray(previous_particle_positions_m, dtype=np.float64)[particles]
    radii = np.asarray(particle_radii_m, dtype=np.float64)[particles]
    normals = np.asarray(contact_normals, dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-30)
    local_body_points = np.asarray(contact_body_positions_m, dtype=np.float64)
    rotated = quaternion_rotate_xyzw(body_quaternion_xyzw, local_body_points)
    body_points = np.asarray(body_position_m, dtype=np.float64) + rotated
    penetration = -(np.einsum("ij,ij->i", normals, positions - body_points) - radii)
    stiffness = np.asarray(penalty_stiffness_n_m, dtype=np.float64)
    damping = np.asarray(damping_ratio, dtype=np.float64)
    friction = np.asarray(friction_coefficients, dtype=np.float64)
    dx = positions - previous
    normal_displacement = np.einsum("ij,ij->i", normals, dx)
    normal_force = np.maximum(penetration, 0.0) * stiffness
    damping_force = np.where(
        (penetration > 0.0) & (normal_displacement < 0.0),
        -(damping * stiffness / dt_s) * normal_displacement,
        0.0,
    )
    normal_force += damping_force

    body_velocity = np.asarray(body_linear_velocity_m_s, dtype=np.float64) + np.cross(
        np.asarray(body_angular_velocity_rad_s, dtype=np.float64), rotated
    )
    relative_velocity = dx / dt_s - body_velocity
    tangent_velocity = (
        relative_velocity
        - normals * np.einsum("ij,ij->i", normals, relative_velocity)[:, None]
    )
    tangent_speed = np.linalg.norm(tangent_velocity, axis=1)
    regularizer = np.maximum(tangent_speed, float(friction_epsilon_m_s))
    friction_forces = (
        -(friction * normal_force / np.maximum(regularizer, 1.0e-30))[:, None]
        * tangent_velocity
    )
    forces = normals * normal_force[:, None] + friction_forces
    active = np.flatnonzero(normal_force > force_threshold_n).astype(np.int32)
    if not len(active):
        return ContactSummary(
            contact_flag=False,
            active_contact_indices=active,
            active_particle_indices=np.empty(0, dtype=np.int32),
            particle_forces_n=np.zeros((0, 3), dtype=np.float64),
            normal_force_n=0.0,
            tangential_force_n=np.zeros(3, dtype=np.float64),
            slip_velocity_m_s=np.zeros(3, dtype=np.float64),
        )
    direction = np.asarray(loading_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    total = np.sum(forces[active], axis=0)
    axial = max(0.0, float(np.dot(total, direction)))
    tangential = total - direction * float(np.dot(total, direction))
    weights = normal_force[active]
    slip = np.average(tangent_velocity[active], axis=0, weights=weights)
    return ContactSummary(
        contact_flag=True,
        active_contact_indices=active,
        active_particle_indices=np.unique(particles[active]).astype(np.int32),
        particle_forces_n=forces[active],
        normal_force_n=axial,
        tangential_force_n=tangential,
        slip_velocity_m_s=slip,
    )


def contact_face_mask(
    deformed_surface_vertices_m: np.ndarray,
    surface_faces: np.ndarray,
    outer_face_indices: np.ndarray,
    active_particle_positions_m: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    outer_face_indices = np.asarray(outer_face_indices, dtype=np.int64)
    if not len(outer_face_indices) or not len(active_particle_positions_m):
        return np.zeros(len(outer_face_indices), dtype=bool)
    triangles = np.asarray(deformed_surface_vertices_m)[
        np.asarray(surface_faces, dtype=np.int64)[outer_face_indices]
    ]
    centroids = np.mean(triangles, axis=1)
    contacts = np.asarray(active_particle_positions_m, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(contacts).query(centroids, k=1, workers=1)
    except ImportError:
        distances = np.min(
            np.linalg.norm(centroids[:, None, :] - contacts[None, :, :], axis=2), axis=1
        )
    return np.asarray(distances <= distance_m, dtype=bool)


def masked_triangle_area(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = np.asarray(face_indices, dtype=np.int64)[np.asarray(mask, dtype=bool)]
    if not len(selected):
        return 0.0
    triangles = np.asarray(vertices_m, dtype=np.float64)[np.asarray(faces)[selected]]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    return float(np.sum(areas))
