"""Rest-state high-resolution surface to tetrahedral mesh mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import AssetValidationError


@dataclass(frozen=True)
class SurfaceMapping:
    tet_ids: np.ndarray
    barycentric_weights: np.ndarray
    reconstruction_error_m: np.ndarray


def _barycentric(
    point: np.ndarray, vertices: np.ndarray, tet: np.ndarray
) -> np.ndarray:
    p0 = vertices[tet[0]]
    dm = np.stack(
        (vertices[tet[1]] - p0, vertices[tet[2]] - p0, vertices[tet[3]] - p0),
        axis=1,
    )
    try:
        tail = np.linalg.solve(dm, point - p0)
    except np.linalg.LinAlgError:
        return np.full(4, np.nan, dtype=np.float64)
    return np.asarray([1.0 - np.sum(tail), tail[0], tail[1], tail[2]])


def build_surface_mapping(
    surface_vertices_m: np.ndarray,
    tet_vertices_m: np.ndarray,
    tets: np.ndarray,
    *,
    barycentric_tolerance: float = 1.0e-6,
    nearest_vertex_tolerance_m: float = 1.0e-9,
    candidate_count: int = 64,
    allow_projection: bool = False,
    projection_tolerance_m: float = 1.0e-7,
) -> SurfaceMapping:
    surface = np.asarray(surface_vertices_m, dtype=np.float64)
    vertices = np.asarray(tet_vertices_m, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    if not len(surface) or not len(vertices) or not len(tets):
        raise AssetValidationError("surface mapping inputs must be nonempty")

    centroids = np.mean(vertices[tets], axis=1)
    incident: list[list[int]] = [[] for _ in range(len(vertices))]
    for tet_id, tet in enumerate(tets):
        for vertex_id in tet:
            incident[int(vertex_id)].append(tet_id)

    try:
        from scipy.spatial import cKDTree

        vertex_tree = cKDTree(vertices)
        centroid_tree = cKDTree(centroids)
        vertex_distances, nearest_vertices = vertex_tree.query(surface, k=1, workers=1)
        _, nearest_tets = centroid_tree.query(
            surface, k=min(candidate_count, len(tets)), workers=1
        )
    except ImportError:
        vertex_distances = np.empty(len(surface), dtype=np.float64)
        nearest_vertices = np.empty(len(surface), dtype=np.int64)
        nearest_tets = np.empty(
            (len(surface), min(candidate_count, len(tets))), dtype=np.int64
        )
        for index, point in enumerate(surface):
            vd = np.linalg.norm(vertices - point, axis=1)
            nearest_vertices[index] = int(np.argmin(vd))
            vertex_distances[index] = float(np.min(vd))
            cd = np.linalg.norm(centroids - point, axis=1)
            nearest_tets[index] = np.argsort(cd)[: nearest_tets.shape[1]]
    if nearest_tets.ndim == 1:
        nearest_tets = nearest_tets[:, None]

    tet_ids = np.full(len(surface), -1, dtype=np.int32)
    weights = np.zeros((len(surface), 4), dtype=np.float64)
    errors = np.full(len(surface), np.inf, dtype=np.float64)
    unresolved: list[int] = []
    for point_id, point in enumerate(surface):
        nearest_vertex = int(nearest_vertices[point_id])
        candidates = list(incident[nearest_vertex])
        candidates.extend(int(value) for value in nearest_tets[point_id])
        candidates = list(dict.fromkeys(candidates))
        best: tuple[float, float, int, np.ndarray] | None = None
        for tet_id in candidates:
            bary = _barycentric(point, vertices, tets[tet_id])
            if not np.isfinite(bary).all():
                continue
            reconstructed = bary @ vertices[tets[tet_id]]
            error = float(np.linalg.norm(reconstructed - point))
            minimum = float(np.min(bary))
            score = (minimum, -error)
            if best is None or score > (best[0], best[1]):
                best = (minimum, -error, tet_id, bary)
        if best is None:
            unresolved.append(point_id)
            continue
        minimum, negative_error, best_tet, bary = best
        error = -negative_error
        contained = (
            minimum >= -barycentric_tolerance
            and np.max(bary) <= 1.0 + barycentric_tolerance
        )
        if contained or vertex_distances[point_id] <= nearest_vertex_tolerance_m:
            bary[np.abs(bary) <= barycentric_tolerance] = 0.0
            bary /= np.sum(bary)
        elif allow_projection:
            bary = np.clip(bary, 0.0, 1.0)
            bary /= np.sum(bary)
            projected = bary @ vertices[tets[best_tet]]
            error = float(np.linalg.norm(projected - point))
            if error > projection_tolerance_m:
                unresolved.append(point_id)
                continue
        else:
            unresolved.append(point_id)
            continue
        tet_ids[point_id] = best_tet
        weights[point_id] = bary
        errors[point_id] = error

    if unresolved:
        sample = unresolved[:10]
        raise AssetValidationError(
            f"Could not map {len(unresolved)} surface vertices into the tet mesh; "
            f"first unresolved indices: {sample}. Ensure the volume mesh conforms to surface.stl."
        )
    return SurfaceMapping(
        tet_ids=tet_ids, barycentric_weights=weights, reconstruction_error_m=errors
    )


def reconstruct_surface(
    tet_particle_positions: np.ndarray,
    tets: np.ndarray,
    mapping: SurfaceMapping | tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    positions = np.asarray(tet_particle_positions, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    if isinstance(mapping, SurfaceMapping):
        tet_ids = mapping.tet_ids
        weights = mapping.barycentric_weights
    else:
        tet_ids, weights = mapping
    corners = positions[tets[np.asarray(tet_ids, dtype=np.int64)]]
    return np.einsum("ni,nij->nj", np.asarray(weights, dtype=np.float64), corners)
