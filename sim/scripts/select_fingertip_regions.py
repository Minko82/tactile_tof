#!/usr/bin/env python3
"""Rebuild explicit fingertip regions without rerunning tetrahedralization."""

# ruff: noqa: E402 -- repository-local packages are added before importing them.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sim.mechanics.mesh import AssetValidationError, read_gmsh_v2, weld_stl_vertices
from sim.mechanics.regions import select_regions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface-stl", required=True, help="prepared meter-scale surface.stl"
    )
    parser.add_argument("--volume-msh", required=True, help="prepared volume.msh")
    parser.add_argument("--regions-config", required=True)
    parser.add_argument("--output", required=True, help="regions.npz destination")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and not args.force:
        parser.error(f"refusing to replace {output}; pass --force")
    try:
        import trimesh

        surface = trimesh.load_mesh(Path(args.surface_stl).resolve(), process=False)
        vertices, faces = weld_stl_vertices(surface.vertices, surface.faces)
        tet_vertices, tets = read_gmsh_v2(Path(args.volume_msh).resolve())
        config = json.loads(Path(args.regions_config).read_text(encoding="utf-8"))
        regions = select_regions(vertices, faces, tet_vertices, tets, config)
    except (AssetValidationError, OSError, ValueError, ImportError) as exc:
        print(f"select_fingertip_regions: error: {exc}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        mount_vertices=regions.mount_vertices,
        inner_coating_faces=regions.inner_coating_faces,
        outer_contact_faces=regions.outer_contact_faces,
        inner_coating_vertices=regions.inner_coating_vertices,
        outer_contact_vertices=regions.outer_contact_vertices,
    )
    print(
        f"Wrote {output}: mount={len(regions.mount_vertices)}, "
        f"inner_faces={len(regions.inner_coating_faces)}, "
        f"outer_faces={len(regions.outer_contact_faces)}, "
        f"median_wall={regions.median_wall_thickness_m * 1000.0:.3f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
