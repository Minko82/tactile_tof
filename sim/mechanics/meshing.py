"""Gmsh-backed surface-conforming tetrahedralization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mesh import AssetValidationError, orient_tets_positive, write_gmsh_v2


def _surface_components(gmsh, surface_tags: list[int]) -> list[list[int]]:
    curve_to_surfaces: dict[int, list[int]] = {}
    for surface in surface_tags:
        curves = gmsh.model.getBoundary([(2, surface)], combined=False, oriented=False)
        for dimension, curve in curves:
            if dimension == 1:
                curve_to_surfaces.setdefault(curve, []).append(surface)
    adjacency = {surface: set() for surface in surface_tags}
    for members in curve_to_surfaces.values():
        for surface in members:
            adjacency[surface].update(members)
    components: list[list[int]] = []
    unseen = set(surface_tags)
    while unseen:
        seed = unseen.pop()
        component = [seed]
        stack = [seed]
        while stack:
            current = stack.pop()
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            component.extend(neighbors)
            stack.extend(neighbors)
        components.append(sorted(component))
    return components


def tetrahedralize_gmsh(
    surface_stl: str | Path,
    output_msh: str | Path,
    *,
    target_edge_m: float,
    classification_angle_degrees: float,
    optimize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "prepare_fingertip.py requires the gmsh Python package; install sim/mechanics-requirements.txt"
        ) from exc

    surface_stl = Path(surface_stl).resolve()
    gmsh.initialize(["prepare_fingertip", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("fingertip")
        gmsh.merge(str(surface_stl))
        angle = np.deg2rad(classification_angle_degrees)
        gmsh.model.mesh.classifySurfaces(angle, True, False, np.pi)
        gmsh.model.mesh.createGeometry()
        gmsh.model.geo.synchronize()
        surface_tags = [
            tag for dimension, tag in gmsh.model.getEntities(2) if dimension == 2
        ]
        if not surface_tags:
            raise AssetValidationError("Gmsh did not create any surface entities")
        components = _surface_components(gmsh, surface_tags)
        loops: list[tuple[float, int]] = []
        for component in components:
            loop = gmsh.model.geo.addSurfaceLoop(component)
            boxes = [gmsh.model.getBoundingBox(2, surface) for surface in component]
            lower = np.min(np.asarray(boxes)[:, :3], axis=0)
            upper = np.max(np.asarray(boxes)[:, 3:], axis=0)
            loops.append((float(np.prod(upper - lower)), loop))
        loops.sort(reverse=True)
        gmsh.model.geo.addVolume([loop for _, loop in loops])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_edge_m)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_edge_m)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)
        if optimize:
            gmsh.model.mesh.optimize("Netgen")

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        node_tags = np.asarray(node_tags, dtype=np.int64)
        order = np.argsort(node_tags)
        sorted_tags = node_tags[order]
        tets: list[np.ndarray] = []
        element_types, _, element_nodes = gmsh.model.mesh.getElements(3)
        for element_type, flattened in zip(element_types, element_nodes, strict=True):
            _, dimension, _, node_count, _, primary_count = (
                gmsh.model.mesh.getElementProperties(element_type)
            )
            if dimension != 3:
                continue
            if node_count != 4 or primary_count != 4:
                raise AssetValidationError(
                    "Gmsh generated non-linear or non-tetrahedral volume cells"
                )
            tags = np.asarray(flattened, dtype=np.int64).reshape(-1, 4)
            locations = np.searchsorted(sorted_tags, tags)
            if np.any(locations >= len(sorted_tags)) or np.any(
                sorted_tags[locations] != tags
            ):
                raise AssetValidationError("Gmsh returned an unknown node tag")
            tets.append(order[locations])
        if not tets:
            raise AssetValidationError("Gmsh generated no tetrahedra")
        tetrahedra = orient_tets_positive(coordinates, np.concatenate(tets, axis=0))
        write_gmsh_v2(output_msh, coordinates, tetrahedra)
        return coordinates, tetrahedra
    finally:
        gmsh.finalize()
