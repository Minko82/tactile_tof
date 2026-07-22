"""Explicit, configuration-driven surface and mount region selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh import AssetValidationError, boundary_faces


@dataclass(frozen=True)
class Regions:
    mount_vertices: np.ndarray
    inner_coating_faces: np.ndarray
    outer_contact_faces: np.ndarray
    inner_coating_vertices: np.ndarray
    outer_contact_vertices: np.ndarray
    minimum_wall_thickness_m: float
    median_wall_thickness_m: float
    maximum_wall_thickness_m: float


def _axis_index(axis: str) -> int:
    try:
        return {"x": 0, "y": 1, "z": 2}[str(axis).lower()]
    except KeyError as exc:
        raise AssetValidationError(f"Unknown selector axis {axis!r}") from exc


def _length_m(
    selector: dict[str, Any], base: str, default: float | None = None
) -> float:
    if f"{base}_m" in selector:
        return float(selector[f"{base}_m"])
    if f"{base}_mm" in selector:
        return float(selector[f"{base}_mm"]) * 1.0e-3
    if default is not None:
        return default
    raise AssetValidationError(f"selector requires {base}_m or {base}_mm")


def _point_mask(points: np.ndarray, selector: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if "all" in selector:
        children = selector["all"]
        if not children:
            return np.ones(len(points), dtype=bool)
        return np.logical_and.reduce([_point_mask(points, child) for child in children])
    if "any" in selector:
        children = selector["any"]
        if not children:
            return np.zeros(len(points), dtype=bool)
        return np.logical_or.reduce([_point_mask(points, child) for child in children])
    if "not" in selector:
        return ~_point_mask(points, selector["not"])

    kind = selector.get("type")
    if kind == "all":
        return np.ones(len(points), dtype=bool)
    if kind == "indices":
        result = np.zeros(len(points), dtype=bool)
        indices = np.asarray(selector.get("values", []), dtype=np.int64)
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= len(points)):
            raise AssetValidationError("region selector contains out-of-range indices")
        result[indices] = True
        return result
    if kind == "coordinate_band":
        axis = _axis_index(selector["axis"])
        center = _length_m(selector, "center")
        half_width = _length_m(selector, "half_width")
        return np.abs(points[:, axis] - center) <= half_width
    if kind == "axis_extreme_band":
        axis = _axis_index(selector["axis"])
        width = _length_m(selector, "width")
        side = selector.get("side")
        if side == "min":
            return points[:, axis] <= np.min(points[:, axis]) + width
        if side == "max":
            return points[:, axis] >= np.max(points[:, axis]) - width
        raise AssetValidationError("axis_extreme_band side must be 'min' or 'max'")
    if kind == "halfspace":
        origin = np.asarray(selector["origin_m"], dtype=np.float64)
        normal = np.asarray(selector["normal"], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        signed = (points - origin) @ normal
        offset = _length_m(selector, "offset", 0.0)
        side = selector.get("side", "positive")
        return signed >= offset if side == "positive" else signed <= offset
    if kind == "box":
        lower = np.asarray(selector["min_m"], dtype=np.float64)
        upper = np.asarray(selector["max_m"], dtype=np.float64)
        return np.all((points >= lower) & (points <= upper), axis=1)
    if kind == "sphere":
        center = np.asarray(selector["center_m"], dtype=np.float64)
        radius = _length_m(selector, "radius")
        return np.linalg.norm(points - center, axis=1) <= radius
    if kind == "radial_range":
        axis = _axis_index(selector.get("axis", "z"))
        center = np.asarray(selector.get("center_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        radial_axes = [index for index in range(3) if index != axis]
        radius = np.linalg.norm(points[:, radial_axes] - center[radial_axes], axis=1)
        minimum = _length_m(selector, "min_radius", 0.0)
        maximum = _length_m(selector, "max_radius", float("inf"))
        return (radius >= minimum) & (radius <= maximum)
    raise AssetValidationError(f"Unknown or missing region selector type {kind!r}")


def select_vertices(points: np.ndarray, selector: dict[str, Any]) -> np.ndarray:
    return np.flatnonzero(_point_mask(points, selector)).astype(np.int32)


def select_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    selector: dict[str, Any],
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    centroids = np.mean(triangles, axis=1)
    if "normal" in selector:
        normal_rule = selector["normal"]
        direction = np.asarray(normal_rule["direction"], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(norms[:, None], 1.0e-30)
        normal_mask = normals @ direction >= float(normal_rule.get("min_dot", 0.0))
        selector = {key: value for key, value in selector.items() if key != "normal"}
    else:
        normal_mask = np.ones(len(faces), dtype=bool)
    if selector.get("type") == "indices":
        mask = np.zeros(len(faces), dtype=bool)
        indices = np.asarray(selector.get("values", []), dtype=np.int64)
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= len(faces)):
            raise AssetValidationError("face selector contains out-of-range indices")
        mask[indices] = True
    else:
        mask = _point_mask(centroids, selector)
    return np.flatnonzero(mask & normal_mask).astype(np.int32)


def _nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(b).query(a, k=1, workers=1)
        return np.asarray(distances, dtype=np.float64)
    except ImportError:
        result = np.empty(len(a), dtype=np.float64)
        block = 512
        for start in range(0, len(a), block):
            delta = a[start : start + block, None, :] - b[None, :, :]
            result[start : start + block] = np.sqrt(
                np.min(np.einsum("ijk,ijk->ij", delta, delta), axis=1)
            )
        return result


def select_regions(
    surface_vertices_m: np.ndarray,
    surface_faces: np.ndarray,
    tet_vertices_m: np.ndarray,
    tets: np.ndarray,
    config: dict[str, Any],
) -> Regions:
    required = ("mount_vertices", "inner_coating_faces", "outer_contact_faces")
    missing = [name for name in required if name not in config]
    if missing:
        raise AssetValidationError("regions config is missing: " + ", ".join(missing))

    volume_boundary = boundary_faces(tets)
    boundary_vertex_ids = np.unique(volume_boundary)
    selected_local = select_vertices(
        np.asarray(tet_vertices_m)[boundary_vertex_ids], config["mount_vertices"]
    )
    mount = boundary_vertex_ids[selected_local].astype(np.int32)
    inner_faces = select_faces(
        surface_vertices_m, surface_faces, config["inner_coating_faces"]
    )
    outer_faces = select_faces(
        surface_vertices_m, surface_faces, config["outer_contact_faces"]
    )
    if not len(mount):
        raise AssetValidationError("mount_vertices selector produced an empty region")
    if not len(inner_faces):
        raise AssetValidationError(
            "inner_coating_faces selector produced an empty region"
        )
    if not len(outer_faces):
        raise AssetValidationError(
            "outer_contact_faces selector produced an empty region"
        )
    overlap = np.intersect1d(inner_faces, outer_faces)
    if len(overlap):
        raise AssetValidationError(
            f"inner and outer coating regions overlap on {len(overlap)} faces"
        )

    inner_vertices = np.unique(np.asarray(surface_faces)[inner_faces]).astype(np.int32)
    outer_vertices = np.unique(np.asarray(surface_faces)[outer_faces]).astype(np.int32)
    shared = np.intersect1d(inner_vertices, outer_vertices)
    inner_for_thickness = np.setdiff1d(inner_vertices, shared)
    outer_for_thickness = np.setdiff1d(outer_vertices, shared)
    if not len(inner_for_thickness) or not len(outer_for_thickness):
        raise AssetValidationError(
            "inner and outer regions do not contain distinct vertices"
        )
    surface_vertices_m = np.asarray(surface_vertices_m, dtype=np.float64)
    distances = np.concatenate(
        (
            _nearest_distances(
                surface_vertices_m[inner_for_thickness],
                surface_vertices_m[outer_for_thickness],
            ),
            _nearest_distances(
                surface_vertices_m[outer_for_thickness],
                surface_vertices_m[inner_for_thickness],
            ),
        )
    )
    minimum = float(np.min(distances))
    median = float(np.median(distances))
    maximum = float(np.max(distances))
    declared_minimum = float(config["minimum_wall_thickness_mm"]) * 1.0e-3
    declared_maximum = float(config["maximum_wall_thickness_mm"]) * 1.0e-3
    if minimum <= 0.0 or minimum < declared_minimum:
        raise AssetValidationError(
            f"wall thickness is {minimum * 1000.0:.6g} mm, below the configured "
            f"minimum {declared_minimum * 1000.0:.6g} mm"
        )
    if median > declared_maximum:
        raise AssetValidationError(
            f"median inner/outer separation is {median * 1000.0:.6g} mm, above the "
            f"configured maximum {declared_maximum * 1000.0:.6g} mm; check region labels"
        )
    return Regions(
        mount_vertices=mount,
        inner_coating_faces=inner_faces,
        outer_contact_faces=outer_faces,
        inner_coating_vertices=inner_vertices,
        outer_contact_vertices=outer_vertices,
        minimum_wall_thickness_m=minimum,
        median_wall_thickness_m=median,
        maximum_wall_thickness_m=maximum,
    )
