"""Configuration-driven fingertip mechanics helpers.

The package is intentionally dependency-light.  Newton, Warp, Gmsh, and
Trimesh are imported only by the command-line entry points that need them, so
the geometry, trajectory, mapping, contact, and export logic can be unit tested
with NumPy alone.
"""

from .config import ConfigError, load_run_config
from .contact import ContactSummary, estimate_contact_summary
from .mapping import SurfaceMapping, build_surface_mapping, reconstruct_surface
from .mesh import AssetValidationError, SurfaceReport, TetReport
from .trajectory import DeterministicTrajectory, PrescribedTrajectory, TrajectorySample

__all__ = [
    "AssetValidationError",
    "ConfigError",
    "ContactSummary",
    "DeterministicTrajectory",
    "PrescribedTrajectory",
    "SurfaceMapping",
    "SurfaceReport",
    "TetReport",
    "TrajectorySample",
    "build_surface_mapping",
    "estimate_contact_summary",
    "load_run_config",
    "reconstruct_surface",
]
