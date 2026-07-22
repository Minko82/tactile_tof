"""Indenter orientation validation and support geometry."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalized_vector(values: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def normalized_quaternion_xyzw(
    values: Any, *, name: str = "quaternion_xyzw"
) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"{name} must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return quaternion / norm


def quaternion_rotate_xyzw(quaternion: Any, vectors: Any) -> np.ndarray:
    q = normalized_quaternion_xyzw(quaternion)
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("vectors must be finite 3-vectors")
    xyz = q[:3]
    return values + 2.0 * np.cross(xyz, np.cross(xyz, values) + q[3] * values)


def indenter_support_distance(
    indenter: dict[str, Any], loading_direction: Any
) -> float:
    """Return the centered primitive's support distance along the loading axis."""

    direction = normalized_vector(loading_direction, name="contact direction")
    kind = str(indenter["type"])
    if kind == "sphere":
        return float(indenter["radius_m"])

    quaternion = normalized_quaternion_xyzw(indenter["quaternion_xyzw"])
    world_axes = quaternion_rotate_xyzw(quaternion, np.eye(3, dtype=np.float64))
    if kind == "flat_plate":
        half_extents = 0.5 * np.asarray(
            [indenter["width_m"], indenter["depth_m"], indenter["thickness_m"]],
            dtype=np.float64,
        )
        return float(np.sum(half_extents * np.abs(world_axes @ direction)))
    if kind == "cylinder":
        cylinder_axis = world_axes[2]
        axial_cosine = float(np.clip(np.dot(direction, cylinder_axis), -1.0, 1.0))
        radial = float(indenter["radius_m"]) * np.sqrt(max(0.0, 1.0 - axial_cosine**2))
        axial = 0.5 * float(indenter["height_m"]) * abs(axial_cosine)
        return float(radial + axial)
    raise ValueError(f"support distance is not defined for indenter type {kind!r}")


def indenter_contact_translation(
    config: dict[str, Any], loading_direction: Any
) -> np.ndarray:
    direction = normalized_vector(loading_direction, name="contact direction")
    location = np.asarray(config["contact"]["location_m"], dtype=np.float64)
    indenter = config["indenter"]
    if indenter["type"] == "rigid_stl":
        local_point = np.asarray(indenter["contact_point_local_m"], dtype=np.float64)
        world_point = quaternion_rotate_xyzw(indenter["quaternion_xyzw"], local_point)
        return location - world_point
    return location - direction * indenter_support_distance(indenter, direction)
