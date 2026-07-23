"""Mesh validation and small, deterministic mesh I/O helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


class AssetValidationError(ValueError):
    """Raised when an asset cannot safely be used by the mechanics solver."""


@dataclass(frozen=True)
class SurfaceReport:
    vertex_count: int
    face_count: int
    component_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    inconsistent_edge_count: int
    degenerate_face_count: int
    duplicate_face_count: int
    bounds_min_m: list[float]
    bounds_max_m: list[float]
    dimensions_m: list[float]
    area_m2: float
    enclosed_volume_m3: float
    finite_coordinates: bool
    positive_enclosed_volume: bool
    watertight: bool
    manifold: bool
    winding_consistent: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TetReport:
    vertex_count: int
    tet_count: int
    boundary_face_count: int
    negative_volume_count: int
    zero_volume_count: int
    min_volume_m3: float
    median_volume_m3: float
    max_volume_m3: float
    min_quality: float
    median_quality: float
    min_condition_number: float
    median_condition_number: float
    max_condition_number: float
    median_boundary_edge_m: float
    bounds_min_m: list[float]
    bounds_max_m: list[float]
    dimensions_m: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    result = np.asarray(vertices, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise AssetValidationError("vertices must have shape (N, 3)")
    if len(result) == 0 or not np.isfinite(result).all():
        raise AssetValidationError("vertices must be nonempty and finite")
    return result


def _as_indices(indices: np.ndarray, width: int, vertex_count: int) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != width:
        raise AssetValidationError(f"indices must have shape (N, {width})")
    if len(result) == 0:
        raise AssetValidationError("index array must be nonempty")
    if np.min(result) < 0 or np.max(result) >= vertex_count:
        raise AssetValidationError("mesh contains out-of-range vertex indices")
    return result


def weld_stl_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    decimal_digits: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct shared STL topology without repairing or deleting faces.

    STL stores three independent vertices per triangle.  Welding coordinates is
    necessary to inspect topology and is not considered a geometry repair.
    Duplicate, degenerate, open, or non-manifold faces remain and are reported.
    """

    vertices = _as_vertices(vertices)
    faces = _as_indices(faces, 3, len(vertices))
    rounded = np.round(vertices, decimals=decimal_digits)
    _, first, inverse = np.unique(
        rounded, axis=0, return_index=True, return_inverse=True
    )
    # Use rounding only as the equivalence key; preserve an original coordinate
    # for every welded vertex so preparation does not quantize the STL.
    return vertices[first].astype(np.float64), inverse[faces].astype(np.int64)


def _component_count(vertex_count: int, edges: np.ndarray) -> int:
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for a, b in edges:
        root_a = find(int(a))
        root_b = find(int(b))
        if root_a != root_b:
            parent[root_b] = root_a
    used = np.unique(edges)
    return len({find(int(index)) for index in used})


def inspect_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    area_epsilon_m2: float = 1.0e-16,
) -> SurfaceReport:
    """Return surface diagnostics without repairing or rejecting the mesh."""

    vertices = _as_vertices(vertices)
    faces = _as_indices(faces, 3, len(vertices))

    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    degenerate_count = int(np.count_nonzero(areas <= area_epsilon_m2))

    canonical_faces = np.sort(faces, axis=1)
    _, face_counts = np.unique(canonical_faces, axis=0, return_counts=True)
    duplicate_count = int(np.sum(np.maximum(face_counts - 1, 0)))

    directed = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    canonical_edges = np.sort(directed, axis=1)
    unique_edges, inverse, edge_counts = np.unique(
        canonical_edges, axis=0, return_inverse=True, return_counts=True
    )
    boundary_count = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_count = int(np.count_nonzero(edge_counts > 2))
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    signed_sums = np.bincount(inverse, weights=signs, minlength=len(unique_edges))
    inconsistent_count = int(np.count_nonzero((edge_counts == 2) & (signed_sums != 0)))
    components = _component_count(len(vertices), unique_edges)

    bounds_min = np.min(vertices, axis=0)
    bounds_max = np.max(vertices, axis=0)
    enclosed_volume = float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            )
        )
        / 6.0
    )
    return SurfaceReport(
        vertex_count=len(vertices),
        face_count=len(faces),
        component_count=components,
        boundary_edge_count=boundary_count,
        nonmanifold_edge_count=nonmanifold_count,
        inconsistent_edge_count=inconsistent_count,
        degenerate_face_count=degenerate_count,
        duplicate_face_count=duplicate_count,
        bounds_min_m=bounds_min.tolist(),
        bounds_max_m=bounds_max.tolist(),
        dimensions_m=(bounds_max - bounds_min).tolist(),
        area_m2=float(np.sum(areas)),
        enclosed_volume_m3=enclosed_volume,
        finite_coordinates=True,
        positive_enclosed_volume=enclosed_volume > 0.0,
        watertight=boundary_count == 0 and nonmanifold_count == 0,
        manifold=boundary_count == 0 and nonmanifold_count == 0,
        winding_consistent=inconsistent_count == 0,
    )


def surface_validation_problems(
    report: SurfaceReport, *, require_single_component: bool = True
) -> list[str]:
    """Return every reason a surface is rejected by the mechanics preflight."""

    problems: list[str] = []
    if report.degenerate_face_count:
        problems.append(f"{report.degenerate_face_count} degenerate faces")
    if report.duplicate_face_count:
        problems.append(f"{report.duplicate_face_count} duplicate faces")
    if report.boundary_edge_count:
        problems.append(f"{report.boundary_edge_count} open boundary edges")
    if report.nonmanifold_edge_count:
        problems.append(f"{report.nonmanifold_edge_count} non-manifold edges")
    if not report.winding_consistent:
        problems.append(
            f"{report.inconsistent_edge_count} edges have inconsistent face winding"
        )
    if require_single_component and report.component_count != 1:
        problems.append(f"{report.component_count} disconnected surface components")
    if not report.positive_enclosed_volume:
        problems.append(
            "enclosed volume is not positive "
            f"({report.enclosed_volume_m3:.6g} m^3)"
        )
    return problems


def validate_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    area_epsilon_m2: float = 1.0e-16,
    require_single_component: bool = True,
) -> SurfaceReport:
    report = inspect_surface(vertices, faces, area_epsilon_m2=area_epsilon_m2)
    problems = surface_validation_problems(
        report, require_single_component=require_single_component
    )
    if problems:
        raise AssetValidationError("Invalid STL surface: " + "; ".join(problems))
    return report

def tet_signed_volumes(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    vertices = _as_vertices(vertices)
    tets = _as_indices(tets, 4, len(vertices))
    p0 = vertices[tets[:, 0]]
    return (
        np.einsum(
            "ij,ij->i",
            vertices[tets[:, 1]] - p0,
            np.cross(vertices[tets[:, 2]] - p0, vertices[tets[:, 3]] - p0),
        )
        / 6.0
    )


def orient_tets_positive(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return tetrahedra with Newton-compatible positive rest winding."""

    result = np.asarray(tets, dtype=np.int64).copy()
    volumes = tet_signed_volumes(vertices, result)
    negative = volumes < 0.0
    if np.any(negative):
        first = result[negative, 0].copy()
        result[negative, 0] = result[negative, 1]
        result[negative, 1] = first
    return result


def boundary_faces(tets: np.ndarray) -> np.ndarray:
    tets = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    candidates = np.concatenate(
        (
            tets[:, (1, 2, 3)],
            tets[:, (0, 3, 2)],
            tets[:, (0, 1, 3)],
            tets[:, (0, 2, 1)],
        ),
        axis=0,
    )
    canonical = np.sort(candidates, axis=1)
    _, first, counts = np.unique(
        canonical, axis=0, return_index=True, return_counts=True
    )
    return candidates[first[counts == 1]]


def tet_quality(
    vertices: np.ndarray, tets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean-ratio quality (1 is regular) and Dm condition numbers."""

    vertices = _as_vertices(vertices)
    tets = _as_indices(tets, 4, len(vertices))
    volumes = np.abs(tet_signed_volumes(vertices, tets))
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_sq = np.zeros(len(tets), dtype=np.float64)
    for a, b in pairs:
        delta = vertices[tets[:, a]] - vertices[tets[:, b]]
        edge_sq += np.einsum("ij,ij->i", delta, delta)
    with np.errstate(invalid="ignore", divide="ignore"):
        quality = 12.0 * np.power(3.0 * volumes, 2.0 / 3.0) / edge_sq
    p0 = vertices[tets[:, 0]]
    dm = np.stack(
        (
            vertices[tets[:, 1]] - p0,
            vertices[tets[:, 2]] - p0,
            vertices[tets[:, 3]] - p0,
        ),
        axis=-1,
    )
    condition = np.linalg.cond(dm)
    return quality, condition


def validate_tets(
    vertices: np.ndarray,
    tets: np.ndarray,
    *,
    min_volume_m3: float,
    min_quality: float,
    max_condition_number: float,
) -> TetReport:
    vertices = _as_vertices(vertices)
    tets = _as_indices(tets, 4, len(vertices))
    volumes = tet_signed_volumes(vertices, tets)
    quality, condition = tet_quality(vertices, tets)
    negative = int(np.count_nonzero(volumes < 0.0))
    zero = int(np.count_nonzero(volumes <= min_volume_m3))
    bad_quality = int(np.count_nonzero(~np.isfinite(quality) | (quality < min_quality)))
    bad_condition = int(
        np.count_nonzero(~np.isfinite(condition) | (condition > max_condition_number))
    )
    faces = boundary_faces(tets)
    edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    bounds_min = np.min(vertices, axis=0)
    bounds_max = np.max(vertices, axis=0)
    report = TetReport(
        vertex_count=len(vertices),
        tet_count=len(tets),
        boundary_face_count=len(faces),
        negative_volume_count=negative,
        zero_volume_count=zero,
        min_volume_m3=float(np.min(volumes)),
        median_volume_m3=float(np.median(volumes)),
        max_volume_m3=float(np.max(volumes)),
        min_quality=float(np.nanmin(quality)),
        median_quality=float(np.nanmedian(quality)),
        min_condition_number=float(np.nanmin(condition)),
        median_condition_number=float(np.nanmedian(condition)),
        max_condition_number=float(np.nanmax(condition)),
        median_boundary_edge_m=float(np.median(lengths)),
        bounds_min_m=bounds_min.tolist(),
        bounds_max_m=bounds_max.tolist(),
        dimensions_m=(bounds_max - bounds_min).tolist(),
    )
    problems: list[str] = []
    if negative:
        problems.append(f"{negative} negatively wound tetrahedra")
    if zero:
        problems.append(f"{zero} tetrahedra at or below {min_volume_m3:g} m^3")
    if bad_quality:
        problems.append(f"{bad_quality} tetrahedra below quality {min_quality:g}")
    if bad_condition:
        problems.append(
            f"{bad_condition} tetrahedra above condition number {max_condition_number:g}"
        )
    if problems:
        raise AssetValidationError("Invalid volume mesh: " + "; ".join(problems))
    return report


def validate_dimensions(
    dimensions_m: Iterable[float],
    expected_dimensions_m: Iterable[float],
    tolerance_m: float,
) -> None:
    actual = np.asarray(list(dimensions_m), dtype=np.float64)
    expected = np.asarray(list(expected_dimensions_m), dtype=np.float64)
    if actual.shape != (3,) or expected.shape != (3,):
        raise AssetValidationError(
            "physical dimensions must contain exactly three values"
        )
    error = np.abs(actual - expected)
    if np.any(error > tolerance_m):
        raise AssetValidationError(
            "Physical dimensions do not match the declared asset: "
            f"actual={actual.tolist()} m, expected={expected.tolist()} m, "
            f"tolerance={tolerance_m:g} m"
        )


def write_gmsh_v2(
    path: str | Path,
    vertices: np.ndarray,
    tets: np.ndarray,
    *,
    surface_faces: np.ndarray | None = None,
) -> None:
    """Write a deterministic ASCII Gmsh 2.2 mesh understood by meshio/Newton."""

    path = Path(path)
    vertices = _as_vertices(vertices)
    tets = _as_indices(tets, 4, len(vertices))
    faces = boundary_faces(tets) if surface_faces is None else np.asarray(surface_faces)
    faces = _as_indices(faces, 3, len(vertices))
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        stream.write(f"$Nodes\n{len(vertices)}\n")
        for index, point in enumerate(vertices, start=1):
            stream.write(f"{index} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n")
        stream.write("$EndNodes\n")
        stream.write(f"$Elements\n{len(faces) + len(tets)}\n")
        element = 1
        for face in faces:
            nodes = " ".join(str(int(value) + 1) for value in face)
            stream.write(f"{element} 2 2 1 1 {nodes}\n")
            element += 1
        for tet in tets:
            nodes = " ".join(str(int(value) + 1) for value in tet)
            stream.write(f"{element} 4 2 2 1 {nodes}\n")
            element += 1
        stream.write("$EndElements\n")


def read_gmsh_v2(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read nodes and linear tetrahedra from an ASCII Gmsh 2.x file."""

    lines = Path(path).read_text(encoding="ascii").splitlines()
    try:
        node_start = lines.index("$Nodes")
        node_end = lines.index("$EndNodes")
        element_start = lines.index("$Elements")
        element_end = lines.index("$EndElements")
    except ValueError as exc:
        raise AssetValidationError("Only ASCII Gmsh 2.x meshes are supported") from exc
    node_lines = lines[node_start + 2 : node_end]
    tags: list[int] = []
    points: list[list[float]] = []
    for line in node_lines:
        fields = line.split()
        tags.append(int(fields[0]))
        points.append([float(fields[1]), float(fields[2]), float(fields[3])])
    tag_to_index = {tag: index for index, tag in enumerate(tags)}
    tets: list[list[int]] = []
    for line in lines[element_start + 2 : element_end]:
        fields = line.split()
        element_type = int(fields[1])
        tag_count = int(fields[2])
        if element_type == 4:
            node_tags = fields[3 + tag_count :]
            if len(node_tags) != 4:
                raise AssetValidationError("volume.msh contains non-linear tetrahedra")
            tets.append([tag_to_index[int(tag)] for tag in node_tags])
    if not tets:
        raise AssetValidationError("volume.msh contains no linear tetrahedra")
    return np.asarray(points, dtype=np.float64), np.asarray(tets, dtype=np.int64)
