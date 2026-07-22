#!/usr/bin/env python3
"""Validate and tetrahedralize a positive, hollow silicone fingertip STL."""

# ruff: noqa: E402 -- repository-local packages are added before importing them.

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sim.mechanics.mapping import build_surface_mapping
from sim.mechanics.mesh import (
    AssetValidationError,
    validate_dimensions,
    validate_surface,
    validate_tets,
    weld_stl_vertices,
)
from sim.mechanics.meshing import tetrahedralize_gmsh
from sim.mechanics.regions import select_regions


UNIT_TO_METERS = {"mm": 1.0e-3, "cm": 1.0e-2, "m": 1.0}
DELIVERABLES = (
    "surface.stl",
    "volume.msh",
    "asset.json",
    "regions.npz",
    "surface_mapping.npz",
)


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetValidationError(
            f"Could not read regions config {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AssetValidationError("regions config must contain a JSON object")
    return value


def _load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "prepare_fingertip.py requires trimesh; install sim/mechanics-requirements.txt"
        ) from exc
    loaded = trimesh.load_mesh(path, process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise AssetValidationError(
            "--stl must contain one triangular mesh, not a scene"
        )
    return np.asarray(loaded.vertices), np.asarray(loaded.faces)


def _write_surface(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False, validate=False
    )
    mesh.export(path, file_type="stl")


def prepare_asset(args: argparse.Namespace) -> Path:
    source = Path(args.stl).resolve()
    regions_path = Path(args.regions_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source.is_file():
        raise AssetValidationError(f"STL does not exist: {source}")
    if not regions_path.is_file():
        raise AssetValidationError(f"regions config does not exist: {regions_path}")
    if args.target_edge_mm <= 0.0:
        raise AssetValidationError("--target-edge-mm must be positive")
    existing = [name for name in DELIVERABLES if (output_dir / name).exists()]
    if existing and not args.force:
        raise AssetValidationError(
            f"output directory already contains {existing}; pass --force to replace only these files"
        )

    config = _load_json(regions_path)
    if config.get("asset_semantics") != "positive_hollow_silicone_body":
        raise AssetValidationError(
            "regions config must declare asset_semantics='positive_hollow_silicone_body'; "
            "a negative manufacturing mold is not a mechanics volume"
        )
    required = (
        "expected_dimensions_mm",
        "dimension_tolerance_mm",
        "minimum_wall_thickness_mm",
        "maximum_wall_thickness_mm",
        "mount_vertices",
        "inner_coating_faces",
        "outer_contact_faces",
        "tet_validation",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise AssetValidationError("regions config is missing: " + ", ".join(missing))

    raw_vertices, raw_faces = _load_stl(source)
    welded_vertices, faces = weld_stl_vertices(
        raw_vertices,
        raw_faces,
        decimal_digits=int(config.get("weld_decimal_digits", 7)),
    )
    vertices_m = welded_vertices * UNIT_TO_METERS[args.units]
    surface_report = validate_surface(
        vertices_m,
        faces,
        area_epsilon_m2=float(config.get("minimum_face_area_mm2", 1.0e-8)) * 1.0e-6,
        require_single_component=bool(
            config.get("require_single_surface_component", True)
        ),
    )
    validate_dimensions(
        surface_report.dimensions_m,
        np.asarray(config["expected_dimensions_mm"], dtype=np.float64) * 1.0e-3,
        float(config["dimension_tolerance_mm"]) * 1.0e-3,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="prepare_fingertip_", dir=output_dir.parent
    ) as temporary:
        temp = Path(temporary)
        surface_path = temp / "surface.stl"
        volume_path = temp / "volume.msh"
        _write_surface(surface_path, vertices_m, faces)
        tet_vertices, tets = tetrahedralize_gmsh(
            surface_path,
            volume_path,
            target_edge_m=float(args.target_edge_mm) * 1.0e-3,
            classification_angle_degrees=float(
                config.get("classification_angle_degrees", 40.0)
            ),
            optimize=bool(config.get("optimize_tets", True)),
        )
        tet_rules = config["tet_validation"]
        tet_report = validate_tets(
            tet_vertices,
            tets,
            min_volume_m3=float(tet_rules["minimum_volume_m3"]),
            min_quality=float(tet_rules["minimum_mean_ratio_quality"]),
            max_condition_number=float(tet_rules["maximum_condition_number"]),
        )
        validate_dimensions(
            tet_report.dimensions_m,
            np.asarray(config["expected_dimensions_mm"], dtype=np.float64) * 1.0e-3,
            float(config["dimension_tolerance_mm"]) * 1.0e-3,
        )
        regions = select_regions(vertices_m, faces, tet_vertices, tets, config)
        mapping_rules = config.get("surface_mapping", {})
        mapping = build_surface_mapping(
            vertices_m,
            tet_vertices,
            tets,
            barycentric_tolerance=float(
                mapping_rules.get("barycentric_tolerance", 1.0e-5)
            ),
            nearest_vertex_tolerance_m=float(
                mapping_rules.get("nearest_vertex_tolerance_mm", 1.0e-5)
            )
            * 1.0e-3,
            candidate_count=int(mapping_rules.get("candidate_tet_count", 96)),
            allow_projection=bool(mapping_rules.get("allow_projection", False)),
            projection_tolerance_m=float(
                mapping_rules.get("projection_tolerance_mm", 1.0e-4)
            )
            * 1.0e-3,
        )

        np.savez_compressed(
            temp / "regions.npz",
            mount_vertices=regions.mount_vertices,
            inner_coating_faces=regions.inner_coating_faces,
            outer_contact_faces=regions.outer_contact_faces,
            inner_coating_vertices=regions.inner_coating_vertices,
            outer_contact_vertices=regions.outer_contact_vertices,
        )
        np.savez_compressed(
            temp / "surface_mapping.npz",
            surface_rest_vertices_m=vertices_m,
            surface_faces=faces.astype(np.int32),
            tet_indices=tets.astype(np.int32),
            surface_tet_ids=mapping.tet_ids,
            barycentric_weights=mapping.barycentric_weights,
            reconstruction_error_m=mapping.reconstruction_error_m,
        )
        asset = {
            "schema_version": 1,
            "asset_id": config.get("asset_id", source.stem),
            "asset_semantics": config["asset_semantics"],
            "source_stl": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_units": args.units,
            "mesh_units": "m",
            "surface_stl": "surface.stl",
            "volume_msh": "volume.msh",
            "regions_npz": "regions.npz",
            "surface_mapping_npz": "surface_mapping.npz",
            "target_edge_m": float(args.target_edge_mm) * 1.0e-3,
            "dimensions_m": surface_report.dimensions_m,
            "wall_thickness_m": {
                "minimum": regions.minimum_wall_thickness_m,
                "median": regions.median_wall_thickness_m,
                "maximum": regions.maximum_wall_thickness_m,
            },
            "validation": {
                "surface": surface_report.to_dict(),
                "tetrahedra": tet_report.to_dict(),
                "surface_mapping_max_error_m": float(
                    np.max(mapping.reconstruction_error_m)
                ),
            },
            "preparer": {
                "script": str(Path(__file__).resolve()),
                "regions_config": str(regions_path),
            },
        }
        (temp / "asset.json").write_text(
            json.dumps(asset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for name in DELIVERABLES:
            destination = output_dir / name
            if destination.exists() and not args.force:
                raise AssetValidationError(f"refusing to replace {destination}")
            os.replace(temp / name, destination)

    print(
        f"Prepared {output_dir / 'asset.json'}: {surface_report.vertex_count} surface vertices, "
        f"{tet_report.tet_count} tetrahedra, wall median "
        f"{regions.median_wall_thickness_m * 1000.0:.3f} mm"
    )
    return output_dir / "asset.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stl", required=True, help="positive hollow silicone body STL"
    )
    parser.add_argument("--units", required=True, choices=tuple(UNIT_TO_METERS))
    parser.add_argument("--target-edge-mm", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regions-config", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the five generated deliverables",
    )
    return parser


def main() -> int:
    try:
        prepare_asset(build_parser().parse_args())
    except (AssetValidationError, RuntimeError) as exc:
        print(f"prepare_fingertip: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
