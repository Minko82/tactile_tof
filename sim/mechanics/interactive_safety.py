"""Safety policy and probe geometry helpers for interactive mechanics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .indenter import (
    normalized_quaternion_xyzw,
    normalized_vector,
    quaternion_rotate_xyzw,
)


@dataclass(frozen=True)
class ProbePose:
    """A normalized probe pose used for both mouse targets and accepted states."""

    position_m: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("probe position must be a finite 3-vector")
        object.__setattr__(self, "position_m", position.copy())
        object.__setattr__(
            self,
            "quaternion_xyzw",
            normalized_quaternion_xyzw(self.quaternion_xyzw).copy(),
        )

    def copy(self) -> ProbePose:
        return ProbePose(self.position_m.copy(), self.quaternion_xyzw.copy())


@dataclass(frozen=True)
class CandidateSafety:
    """Result of evaluating one simulated candidate substep."""

    fatal_reason: str | None
    stop_reason: str | None
    warning_reasons: tuple[str, ...]
    minimum_relative_j: float
    affected_tet_indices: np.ndarray
    estimated_force_magnitude_n: float
    commanded_indentation_m: float

    @property
    def fatal(self) -> bool:
        return self.fatal_reason is not None

    @property
    def stopped(self) -> bool:
        return self.stop_reason is not None

    @property
    def warned(self) -> bool:
        return bool(self.warning_reasons)


def probe_support_distance(
    probe: dict[str, Any],
    direction: Any,
    quaternion_xyzw: Any,
    *,
    mesh_vertices_m: np.ndarray | None = None,
) -> float:
    """Return probe support along ``direction`` for its actual orientation."""

    direction_array = normalized_vector(direction, name="contact direction")
    quaternion = normalized_quaternion_xyzw(quaternion_xyzw)
    kind = str(probe["type"])
    if kind == "sphere":
        return float(probe["radius_m"])
    if kind in {"rounded_block", "custom_rigid_stl"}:
        if mesh_vertices_m is None or not len(mesh_vertices_m):
            raise ValueError("mesh probe support requires local mesh vertices")
        rotated = quaternion_rotate_xyzw(quaternion, mesh_vertices_m)
        return float(np.max(rotated @ direction_array))

    world_axis = quaternion_rotate_xyzw(
        quaternion, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    )
    axial_cosine = float(np.clip(np.dot(direction_array, world_axis), -1.0, 1.0))
    half_height = 0.5 * float(probe["height_m"])
    radius = float(probe["radius_m"])
    if kind == "capsule":
        return float(radius + half_height * abs(axial_cosine))
    if kind == "cylinder":
        radial = radius * math.sqrt(max(0.0, 1.0 - axial_cosine**2))
        return float(radial + half_height * abs(axial_cosine))
    raise ValueError(f"unsupported probe support geometry {kind!r}")


def commanded_indentation(
    pose: ProbePose,
    *,
    probe: dict[str, Any],
    contact_location_m: Any,
    contact_direction: Any,
    mesh_vertices_m: np.ndarray | None = None,
) -> float:
    """Return support-point command beyond the configured contact reference.

    Negative values are clearance. Positive values are commanded indentation;
    this is not a measurement of literal object penetration.
    """

    direction = normalized_vector(contact_direction, name="contact direction")
    location = np.asarray(contact_location_m, dtype=np.float64)
    if location.shape != (3,) or not np.isfinite(location).all():
        raise ValueError("contact location must be a finite 3-vector")
    support = probe_support_distance(
        probe,
        direction,
        pose.quaternion_xyzw,
        mesh_vertices_m=mesh_vertices_m,
    )
    return float(np.dot(pose.position_m - location, direction) + support)


def evaluate_candidate_safety(
    *,
    relative_j: Any,
    estimated_force_magnitude_n: float,
    commanded_indentation_m: float,
    circuit_breaker_minimum_j: float,
    stop_minimum_j: float,
    warning_minimum_j: float,
    stop_estimated_force_n: float,
    warning_estimated_force_n: float,
    maximum_commanded_indentation_m: float,
    warning_commanded_indentation_m: float,
) -> CandidateSafety:
    """Evaluate primary tet-volume and secondary force/command limits."""

    relative = np.asarray(relative_j, dtype=np.float64)
    finite = np.isfinite(relative)
    minimum_j = float(np.min(relative[finite])) if np.any(finite) else float("-inf")
    fatal_mask = ~finite | (relative < float(circuit_breaker_minimum_j))
    stop_mask = ~finite | (relative < float(stop_minimum_j))
    warning_mask = ~finite | (relative < float(warning_minimum_j))

    fatal_reason = "tet_volume_circuit_breaker" if np.any(fatal_mask) else None
    stop_reason = None
    if fatal_reason is None:
        if np.any(stop_mask):
            stop_reason = "minimum_relative_tet_volume"
        elif commanded_indentation_m > maximum_commanded_indentation_m:
            stop_reason = "commanded_indentation_limit"
        elif estimated_force_magnitude_n > stop_estimated_force_n:
            stop_reason = "estimated_force_limit"

    warning_reasons: list[str] = []
    if np.any(warning_mask):
        warning_reasons.append("minimum_relative_tet_volume")
    if commanded_indentation_m > warning_commanded_indentation_m:
        warning_reasons.append("commanded_indentation")
    if estimated_force_magnitude_n > warning_estimated_force_n:
        warning_reasons.append("estimated_force")

    affected_mask = fatal_mask if fatal_reason else stop_mask
    if not np.any(affected_mask) and warning_reasons:
        affected_mask = warning_mask
    affected = np.flatnonzero(affected_mask).astype(np.int32)
    if not len(affected) and (fatal_reason or stop_reason or warning_reasons):
        affected = np.asarray([int(np.nanargmin(relative))], dtype=np.int32)

    return CandidateSafety(
        fatal_reason=fatal_reason,
        stop_reason=stop_reason,
        warning_reasons=tuple(warning_reasons),
        minimum_relative_j=minimum_j,
        affected_tet_indices=affected,
        estimated_force_magnitude_n=float(estimated_force_magnitude_n),
        commanded_indentation_m=float(commanded_indentation_m),
    )
