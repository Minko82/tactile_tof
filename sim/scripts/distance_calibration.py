"""Ascending-only VL53L5CX distance calibration for shape replays.

This module deliberately operates without Isaac Sim.  It consumes immutable
shape-replay-v2 ``off`` captures, canonical real ascending logs, and exact STL
geometry.  No code path resolves or opens a descending input.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import shape_replay


MANIFEST_SCHEMA = "shape-distance-training-manifest-v1"
ARTIFACT_SCHEMA = "vl53l5cx-distance-calibration-v1"
TRAINING_REGIME = "cup_spoon_ascending_v1"
LAMBDA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass(frozen=True)
class TrainingSettings:
    max_timestamp_gap_s: float = 0.25
    max_adjacent_tcp_delta_mm: float = 0.75
    min_plateau_frames: int = 4
    max_plateau_range_mm: float = 2.5
    max_abs_plateau_slope_mm_s: float = 1.0
    table_gate_mm: float = 18.0
    object_clearance_m: float = 0.005
    min_cell_frames: int = 4
    min_heights_per_shape: int = 3
    min_total_height_groups: int = 8
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainingSettings":
        known = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in known if key in values})


@dataclass(frozen=True)
class ManifestEntry:
    name: str
    direction: str
    profile_path: Path
    robot_csv: Path
    real_tof_csv: Path
    simulation_csv: Path


@dataclass(frozen=True)
class TrainingManifest:
    path: Path
    training_regime_id: str
    entries: tuple[ManifestEntry, ...]
    settings: TrainingSettings
    output_artifact: Path
    coverage_report: Path

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingManifest":
        manifest_path = Path(path).resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported training manifest schema in {manifest_path}")
        regime = str(data.get("training_regime_id", ""))
        if regime != TRAINING_REGIME:
            raise ValueError(f"this fitter requires training_regime_id={TRAINING_REGIME!r}")

        def resolve(value: Any, field: str) -> Path:
            if not value:
                raise ValueError(f"{field} is required")
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            candidate = candidate.resolve()
            if any("descending" in part.lower() for part in candidate.parts):
                raise ValueError(f"{field} must not reference a descending path: {candidate}")
            return candidate

        entries: list[ManifestEntry] = []
        for index, item in enumerate(data.get("entries", ())):
            direction = str(item.get("direction", ""))
            if direction != "ascending":
                raise ValueError(f"entries[{index}].direction must be exactly 'ascending'")
            entries.append(
                ManifestEntry(
                    str(item.get("name", "")),
                    direction,
                    resolve(item.get("profile"), f"entries[{index}].profile"),
                    resolve(item.get("robot_csv"), f"entries[{index}].robot_csv"),
                    resolve(item.get("real_tof_csv"), f"entries[{index}].real_tof_csv"),
                    resolve(item.get("simulation_csv"), f"entries[{index}].simulation_csv"),
                )
            )
        if {entry.name for entry in entries} != {"cup", "spoon"} or len(entries) != 2:
            raise ValueError("cup_spoon_ascending_v1 requires exactly one cup and one spoon entry")
        output = resolve(data.get("output_artifact"), "output_artifact")
        report_value = data.get("coverage_report", "cup_spoon_ascending_v1_coverage.json")
        report = resolve(report_value, "coverage_report")
        return cls(
            manifest_path,
            regime,
            tuple(entries),
            TrainingSettings.from_mapping(dict(data.get("settings", {}))),
            output,
            report,
        )


@dataclass(frozen=True)
class Plateau:
    shape: str
    plateau_id: int
    indices: tuple[int, ...]
    median_tcp_z_mm: float
    tcp_range_mm: float
    slope_mm_s: float


@dataclass(frozen=True)
class SimulationRow:
    timestamp: str
    ranges_m: tuple[float | None, ...]
    valid: tuple[bool, ...]


@dataclass(frozen=True)
class TrainingDataset:
    entry: ManifestEntry
    profile: shape_replay.ShapeExperimentProfile
    sensor_config: dict[str, Any]
    samples: tuple[shape_replay.ReplaySample, ...]
    simulation_rows: tuple[SimulationRow, ...]
    plateaus: tuple[Plateau, ...]


@dataclass(frozen=True)
class CalibrationFit:
    provisional_global_mm: float
    final_global_mm: float
    common_component_mm: float
    residuals_mm: tuple[float, ...]
    support_mask: tuple[bool, ...]
    effective_sample_sizes: tuple[float, ...]
    coverage: tuple[dict[str, Any], ...]
    retained_cell_medians: dict[tuple[str, int, int], float]
    retained_cell_projected_medians: dict[tuple[str, int, int], float]


class SupportPreflightError(ValueError):
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        unsupported = report.get("unsupported_zones", [])
        super().__init__(
            "cup_spoon_ascending_v1 support preflight failed for zones: "
            + ", ".join(str(value) for value in unsupported)
            + "; collect a separately versioned empty-table or flat-target regime"
        )


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot calculate a median of no values")
    return float(statistics.median(values))


def theil_sen_slope(times: Sequence[float], values: Sequence[float]) -> float:
    slopes = [
        (float(values[right]) - float(values[left])) / (float(times[right]) - float(times[left]))
        for left in range(len(values))
        for right in range(left + 1, len(values))
        if float(times[right]) > float(times[left])
    ]
    return _median(slopes) if slopes else 0.0


def segment_plateaus(
    shape: str,
    samples: Sequence[shape_replay.ReplaySample],
    settings: TrainingSettings,
) -> tuple[Plateau, ...]:
    """Find stable maximal runs; rejected slow motion is never subdivided."""

    if not samples:
        return ()
    candidates: list[list[int]] = [[0]]
    for index in range(1, len(samples)):
        previous = samples[index - 1]
        current = samples[index]
        gap = current.elapsed_s - previous.elapsed_s
        delta_mm = abs(current.tcp_z_m - previous.tcp_z_m) * 1000.0
        if gap <= settings.max_timestamp_gap_s and delta_mm <= settings.max_adjacent_tcp_delta_mm:
            candidates[-1].append(index)
        else:
            candidates.append([index])

    plateaus: list[Plateau] = []
    for candidate in candidates:
        if len(candidate) < settings.min_plateau_frames:
            continue
        z_values = [samples[index].tcp_z_m * 1000.0 for index in candidate]
        times = [samples[index].elapsed_s for index in candidate]
        complete_range = max(z_values) - min(z_values)
        slope = theil_sen_slope(times, z_values)
        if complete_range > settings.max_plateau_range_mm or abs(slope) > settings.max_abs_plateau_slope_mm_s:
            continue
        plateaus.append(
            Plateau(shape, len(plateaus), tuple(candidate), _median(z_values), complete_range, slope)
        )
    return tuple(plateaus)


def _load_simulation_rows(path: Path) -> tuple[SimulationRow, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"schema_version", "reference_timestamp"}
        required.update(f"valid_{zone:02d}" for zone in range(64))
        required.update(f"rtx_range_m_{zone:02d}" for zone in range(64))
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path} is not an immutable shape-replay-v2 capture; missing {missing[0]}")
        rows: list[SimulationRow] = []
        for row_number, row in enumerate(reader, start=2):
            if row.get("schema_version") != shape_replay.OUTPUT_SCHEMA_VERSION:
                raise ValueError(f"{path}:{row_number} has an unsupported schema version")
            timestamp = str(row.get("reference_timestamp", ""))
            ranges: list[float | None] = []
            valid: list[bool] = []
            for zone in range(64):
                is_valid = str(row[f"valid_{zone:02d}"]).strip() in ("1", "true", "True")
                value = str(row[f"rtx_range_m_{zone:02d}"]).strip()
                valid.append(is_valid)
                ranges.append(float(value) if is_valid and value else None)
            rows.append(SimulationRow(timestamp, tuple(ranges), tuple(valid)))
    if not rows:
        raise ValueError(f"{path} contains no simulation rows")
    return tuple(rows)


def _load_dataset(entry: ManifestEntry, settings: TrainingSettings) -> TrainingDataset:
    for path in (entry.profile_path, entry.robot_csv, entry.real_tof_csv, entry.simulation_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = shape_replay.ShapeExperimentProfile.from_json(
        entry.profile_path,
        require_files=False,
        selected_directions=("ascending",),
    )
    profile = replace(
        profile,
        directions={"ascending": shape_replay.DirectionFiles(entry.robot_csv, entry.real_tof_csv)},
    )
    with profile.sensor_profile.open("r", encoding="utf-8") as handle:
        sensor_config = json.load(handle)
    samples = tuple(shape_replay.load_replay_samples(profile, "ascending"))
    loaded_rows = _load_simulation_rows(entry.simulation_csv)
    rows_by_timestamp = {row.timestamp: row for row in loaded_rows}
    if len(rows_by_timestamp) != len(loaded_rows):
        raise ValueError(f"{entry.simulation_csv} contains duplicate reference timestamps")
    missing = [sample.reference_timestamp for sample in samples if sample.reference_timestamp not in rows_by_timestamp]
    if missing:
        raise ValueError(f"simulation capture is missing ascending timestamp {missing[0]}")
    rows = tuple(rows_by_timestamp[sample.reference_timestamp] for sample in samples)
    plateaus = segment_plateaus(entry.name, samples, settings)
    return TrainingDataset(entry, profile, sensor_config, samples, rows, plateaus)


def validate_shared_compatibility(datasets: Sequence[TrainingDataset]) -> dict[str, Any]:
    if len(datasets) != 2:
        raise ValueError("shared compatibility requires cup and spoon datasets")
    reference = datasets[0]
    reference_hash = shape_replay.sha256_file(reference.profile.sensor_profile)
    rows = int(reference.sensor_config["rows"])
    cols = int(reference.sensor_config["cols"])
    reference_angles = shape_replay.canonical_emitter_angles(
        rows,
        cols,
        float(reference.sensor_config["fov_h_deg"]),
        float(reference.sensor_config["fov_v_deg"]),
    )
    results: dict[str, Any] = {
        "reference_profile": reference.profile.name,
        "sensor_profile_sha256": reference_hash,
        "tcp_to_sensor_z_tolerance_mm": 0.10,
        "orientation_tolerance_deg": 0.01,
        "emitter_angle_tolerance_deg": 1.0e-12,
        "profiles": {},
    }
    for dataset in datasets:
        profile = dataset.profile
        if shape_replay.sha256_file(profile.sensor_profile) != reference_hash:
            raise ValueError("cup and spoon must use identical fixed sensor-profile content")
        angles = shape_replay.canonical_emitter_angles(
            int(dataset.sensor_config["rows"]),
            int(dataset.sensor_config["cols"]),
            float(dataset.sensor_config["fov_h_deg"]),
            float(dataset.sensor_config["fov_v_deg"]),
        )
        if len(angles) != len(reference_angles) or any(
            abs(left.azimuth_deg - right.azimuth_deg) > 1.0e-12
            or abs(left.elevation_deg - right.elevation_deg) > 1.0e-12
            for left, right in zip(angles, reference_angles)
        ):
            raise ValueError("cup and spoon emitter geometries are incompatible")
        mount_difference_mm = abs(profile.tcp_to_sensor_z_m - reference.profile.tcp_to_sensor_z_m) * 1000.0
        orientation_difference = shape_replay.quaternion_angular_difference_deg(
            profile.sensor_quat_wxyz, reference.profile.sensor_quat_wxyz
        )
        if mount_difference_mm > 0.10:
            raise ValueError("cup and spoon tcp_to_sensor_z_m values differ by more than 0.10 mm")
        if orientation_difference > 0.01:
            raise ValueError("cup and spoon mount orientations differ by more than 0.01 degrees")
        results["profiles"][profile.name] = {
            "tcp_to_sensor_z_m": profile.tcp_to_sensor_z_m,
            "sensor_quat_wxyz": list(profile.sensor_quat_wxyz),
            "difference_from_reference_mm": mount_difference_mm,
            "orientation_difference_from_reference_deg": orientation_difference,
        }
    return results


@dataclass
class _BvhNode:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    indices: tuple[int, ...] = ()
    left: "_BvhNode | None" = None
    right: "_BvhNode | None" = None


class WorldMeshIndex:
    def __init__(self, profile: shape_replay.ShapeExperimentProfile) -> None:
        mesh = shape_replay.load_stl(profile.stl_path)
        points = [shape_replay.transform_stl_vertex_to_world(vertex, profile) for vertex in mesh.vertices]
        self.triangles = tuple(tuple(points[index : index + 3]) for index in range(0, len(points), 3))
        centroids = [tuple(sum(vertex[axis] for vertex in triangle) / 3.0 for axis in range(3)) for triangle in self.triangles]

        def build(indices: list[int]) -> _BvhNode:
            minimum = tuple(min(self.triangles[index][vertex][axis] for index in indices for vertex in range(3)) for axis in range(3))
            maximum = tuple(max(self.triangles[index][vertex][axis] for index in indices for vertex in range(3)) for axis in range(3))
            if len(indices) <= 32:
                return _BvhNode(minimum, maximum, tuple(indices))
            spans = [maximum[axis] - minimum[axis] for axis in range(3)]
            axis = max(range(3), key=lambda value: spans[value])
            indices.sort(key=lambda index: centroids[index][axis])
            middle = len(indices) // 2
            return _BvhNode(minimum, maximum, left=build(indices[:middle]), right=build(indices[middle:]))

        self.root = build(list(range(len(self.triangles))))
        self._footprint_cell_m = 0.002
        min_x = min(point[0] for point in points) - 0.008
        min_y = min(point[1] for point in points) - 0.008
        max_x = max(point[0] for point in points) + 0.008
        max_y = max(point[1] for point in points) + 0.008
        self._footprint_origin = (min_x, min_y)
        self._footprint_width = max(1, int(math.ceil((max_x - min_x) / self._footprint_cell_m)))
        self._footprint_height = max(1, int(math.ceil((max_y - min_y) / self._footprint_cell_m)))
        occupied: set[tuple[int, int]] = set()
        for triangle in self.triangles:
            low_col = max(0, int(math.floor((min(point[0] for point in triangle) - min_x) / self._footprint_cell_m)))
            high_col = min(
                self._footprint_width - 1,
                int(math.floor((max(point[0] for point in triangle) - min_x) / self._footprint_cell_m)),
            )
            low_row = max(0, int(math.floor((min(point[1] for point in triangle) - min_y) / self._footprint_cell_m)))
            high_row = min(
                self._footprint_height - 1,
                int(math.floor((max(point[1] for point in triangle) - min_y) / self._footprint_cell_m)),
            )
            a, b, c = ((point[0], point[1]) for point in triangle)
            for row in range(low_row, high_row + 1):
                py = min_y + (row + 0.5) * self._footprint_cell_m
                for col in range(low_col, high_col + 1):
                    px = min_x + (col + 0.5) * self._footprint_cell_m
                    if shape_replay._point_in_triangle_2d(px, py, a, b, c):
                        occupied.add((col, row))
            for point in triangle:
                occupied.add(
                    (
                        int(math.floor((point[0] - min_x) / self._footprint_cell_m)),
                        int(math.floor((point[1] - min_y) / self._footprint_cell_m)),
                    )
                )
        self._footprint = occupied

    @staticmethod
    def _ray_box(origin: Sequence[float], direction: Sequence[float], node: _BvhNode, maximum_t: float) -> bool:
        low, high = 0.0, maximum_t
        for axis in range(3):
            if abs(direction[axis]) < 1.0e-12:
                if origin[axis] < node.minimum[axis] or origin[axis] > node.maximum[axis]:
                    return False
                continue
            inverse = 1.0 / direction[axis]
            first = (node.minimum[axis] - origin[axis]) * inverse
            second = (node.maximum[axis] - origin[axis]) * inverse
            if first > second:
                first, second = second, first
            low = max(low, first)
            high = min(high, second)
            if high < low:
                return False
        return True

    @staticmethod
    def _ray_triangle(origin: Sequence[float], direction: Sequence[float], triangle: Sequence[Sequence[float]]) -> float | None:
        a, b, c = triangle
        edge1 = tuple(b[axis] - a[axis] for axis in range(3))
        edge2 = tuple(c[axis] - a[axis] for axis in range(3))
        cross = (
            direction[1] * edge2[2] - direction[2] * edge2[1],
            direction[2] * edge2[0] - direction[0] * edge2[2],
            direction[0] * edge2[1] - direction[1] * edge2[0],
        )
        determinant = sum(edge1[axis] * cross[axis] for axis in range(3))
        if abs(determinant) < 1.0e-12:
            return None
        inverse = 1.0 / determinant
        offset = tuple(origin[axis] - a[axis] for axis in range(3))
        u = sum(offset[axis] * cross[axis] for axis in range(3)) * inverse
        if u < 0.0 or u > 1.0:
            return None
        q = (
            offset[1] * edge1[2] - offset[2] * edge1[1],
            offset[2] * edge1[0] - offset[0] * edge1[2],
            offset[0] * edge1[1] - offset[1] * edge1[0],
        )
        v = sum(direction[axis] * q[axis] for axis in range(3)) * inverse
        if v < 0.0 or u + v > 1.0:
            return None
        distance = sum(edge2[axis] * q[axis] for axis in range(3)) * inverse
        return distance if distance > 1.0e-9 else None

    def first_hit(self, origin: Sequence[float], direction: Sequence[float], maximum_t: float) -> float | None:
        best = maximum_t
        hit = False
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not self._ray_box(origin, direction, node, best):
                continue
            if node.indices:
                for index in node.indices:
                    distance = self._ray_triangle(origin, direction, self.triangles[index])
                    if distance is not None and distance < best:
                        best, hit = distance, True
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)
        return best if hit else None

    def footprint_within(self, x_m: float, y_m: float, clearance_m: float) -> bool:
        col = int(math.floor((x_m - self._footprint_origin[0]) / self._footprint_cell_m))
        row = int(math.floor((y_m - self._footprint_origin[1]) / self._footprint_cell_m))
        radius = int(math.ceil(clearance_m / self._footprint_cell_m))
        return any(
            (test_col, test_row) in self._footprint
            for test_col in range(col - radius, col + radius + 1)
            for test_row in range(row - radius, row + radius + 1)
            if (test_col - col) ** 2 + (test_row - row) ** 2 <= radius * radius
        )


def _classification_cells(
    datasets: Sequence[TrainingDataset],
    settings: TrainingSettings,
) -> tuple[
    dict[tuple[str, int, int], list[float]],
    dict[tuple[str, int, int], list[float]],
    dict[str, Any],
]:
    residuals: dict[tuple[str, int, int], list[float]] = {}
    projected_values: dict[tuple[str, int, int], list[float]] = {}
    plateau_report: dict[str, Any] = {}
    for dataset in datasets:
        config = dataset.sensor_config
        local_rays = shape_replay.canonical_local_rays(
            int(config["rows"]), int(config["cols"]), float(config["fov_h_deg"]), float(config["fov_v_deg"])
        )
        factors = shape_replay.projection_factors(
            int(config["rows"]), int(config["cols"]), float(config["fov_h_deg"]), float(config["fov_v_deg"])
        )
        world_forward = shape_replay.quaternion_rotate_vector(dataset.profile.sensor_quat_wxyz, (1.0, 0.0, 0.0))
        mesh_index = WorldMeshIndex(dataset.profile)
        plateau_report[dataset.entry.name] = []
        for plateau in dataset.plateaus:
            plateau_report[dataset.entry.name].append(
                {
                    "plateau_id": plateau.plateau_id,
                    "median_tcp_z_mm": plateau.median_tcp_z_mm,
                    "frames": len(plateau.indices),
                    "tcp_range_mm": plateau.tcp_range_mm,
                    "theil_sen_slope_mm_s": plateau.slope_mm_s,
                }
            )
            for sample_index in plateau.indices:
                sample = dataset.samples[sample_index]
                sim = dataset.simulation_rows[sample_index]
                origin = shape_replay.sensor_world_position(dataset.profile, sample.tcp_z_m)
                for zone, local_ray in enumerate(local_rays):
                    real = sample.real_zones_mm[zone]
                    range_m = sim.ranges_m[zone]
                    if real <= 0 or not sim.valid[zone] or range_m is None:
                        continue
                    world_ray = shape_replay.quaternion_rotate_vector(dataset.profile.sensor_quat_wxyz, local_ray)
                    if world_ray[2] >= -1.0e-12:
                        continue
                    table_t = (dataset.profile.table_top_z_m - origin[2]) / world_ray[2]
                    if table_t <= 0.0:
                        continue
                    table_point = tuple(origin[axis] + table_t * world_ray[axis] for axis in range(3))
                    object_hit = mesh_index.first_hit(origin, world_ray, table_t)
                    if object_hit is not None and object_hit < table_t - 1.0e-6:
                        continue
                    if mesh_index.footprint_within(table_point[0], table_point[1], settings.object_clearance_m):
                        continue
                    axial_factor_world = sum(world_ray[axis] * world_forward[axis] for axis in range(3))
                    expected_table_mm = table_t * axial_factor_world * 1000.0
                    projected_mm = range_m * 1000.0 * factors[zone]
                    if abs(float(real) - expected_table_mm) > settings.table_gate_mm:
                        continue
                    if abs(projected_mm - expected_table_mm) > settings.table_gate_mm:
                        continue
                    key = (dataset.entry.name, plateau.plateau_id, zone)
                    residuals.setdefault(key, []).append(float(real) - projected_mm)
                    projected_values.setdefault(key, []).append(projected_mm)
    return residuals, projected_values, plateau_report


def _coverage_report(
    cells: dict[tuple[str, int, int], Sequence[float]],
    settings: TrainingSettings,
    *,
    stage: str,
) -> dict[str, Any]:
    coverage: list[dict[str, Any]] = []
    unsupported: list[int] = []
    for zone in range(64):
        heights: dict[str, list[int]] = {}
        frame_counts: dict[str, dict[str, int]] = {}
        for shape in ("cup", "spoon"):
            supported_plateaus = sorted(
                plateau
                for (cell_shape, plateau, cell_zone), values in cells.items()
                if cell_shape == shape and cell_zone == zone and len(values) >= settings.min_cell_frames
            )
            heights[shape] = supported_plateaus
            frame_counts[shape] = {
                str(plateau): len(cells[(shape, plateau, zone)]) for plateau in supported_plateaus
            }
        supported = (
            len(heights["cup"]) >= settings.min_heights_per_shape
            and len(heights["spoon"]) >= settings.min_heights_per_shape
            and len(heights["cup"]) + len(heights["spoon"]) >= settings.min_total_height_groups
        )
        if not supported:
            unsupported.append(zone)
        coverage.append(
            {
                "zone": zone,
                "supported": supported,
                "supported_plateaus": heights,
                "frame_counts": frame_counts,
            }
        )
    return {
        "schema_version": "shape-distance-coverage-v1",
        "training_regime_id": TRAINING_REGIME,
        "stage": stage,
        "support_policy": {
            "min_frames_per_shape_zone_height": settings.min_cell_frames,
            "min_heights_per_shape": settings.min_heights_per_shape,
            "min_total_height_groups": settings.min_total_height_groups,
        },
        "supported_all_64": not unsupported,
        "unsupported_zones": unsupported,
        "zones": coverage,
    }


def _reject_outliers(values: Sequence[float]) -> list[float]:
    center = _median(values)
    mad = _median([abs(value - center) for value in values])
    threshold = max(3.0, 3.0 * 1.4826 * mad)
    return [float(value) for value in values if abs(float(value) - center) <= threshold]


def _retained_outlier_indices(values: Sequence[float]) -> list[int]:
    center = _median(values)
    mad = _median([abs(float(value) - center) for value in values])
    threshold = max(3.0, 3.0 * 1.4826 * mad)
    return [index for index, value in enumerate(values) if abs(float(value) - center) <= threshold]


def _fit_from_cells(
    residual_cells: dict[tuple[str, int, int], Sequence[float]],
    projected_cells: dict[tuple[str, int, int], Sequence[float]],
    settings: TrainingSettings,
    shrinkage_lambda: float,
    *,
    strict: bool,
) -> CalibrationFit:
    medians: dict[tuple[str, int, int], float] = {}
    projected_medians: dict[tuple[str, int, int], float] = {}
    retained_values: dict[tuple[str, int, int], list[float]] = {}
    for key, values in residual_cells.items():
        retained_indices = _retained_outlier_indices(values)
        retained = [float(values[index]) for index in retained_indices]
        if len(retained) < settings.min_cell_frames:
            continue
        retained_values[key] = retained
        medians[key] = _median(retained)
        projected_medians[key] = _median([float(projected_cells[key][index]) for index in retained_indices])
    post_report = _coverage_report(retained_values, settings, stage="post_outlier")
    if strict and not post_report["supported_all_64"]:
        raise SupportPreflightError(post_report)

    shape_globals: list[float] = []
    for shape in ("cup", "spoon"):
        height_values: list[float] = []
        plateaus = sorted({plateau for cell_shape, plateau, _zone in medians if cell_shape == shape})
        for plateau in plateaus:
            zone_values = [value for (cell_shape, cell_plateau, _zone), value in medians.items() if cell_shape == shape and cell_plateau == plateau]
            if zone_values:
                height_values.append(_median(zone_values))
        if height_values:
            shape_globals.append(sum(height_values) / len(height_values))
    if len(shape_globals) != 2:
        raise ValueError("both cup and spoon require supported training cells for the global offset")
    provisional_global = sum(shape_globals) / 2.0

    support: list[bool] = []
    effective: list[float] = []
    shrunk: list[float] = []
    for zone in range(64):
        shape_estimates: list[float] = []
        heights: dict[str, int] = {}
        for shape in ("cup", "spoon"):
            values = [
                value - provisional_global
                for (cell_shape, _plateau, cell_zone), value in medians.items()
                if cell_shape == shape and cell_zone == zone
            ]
            heights[shape] = len(values)
            if values:
                shape_estimates.append(sum(values) / len(values))
        zone_supported = (
            heights["cup"] >= settings.min_heights_per_shape
            and heights["spoon"] >= settings.min_heights_per_shape
            and heights["cup"] + heights["spoon"] >= settings.min_total_height_groups
        )
        support.append(zone_supported)
        if zone_supported:
            n_eff = 4.0 / (1.0 / heights["cup"] + 1.0 / heights["spoon"])
            unshrunk = sum(shape_estimates) / 2.0
            shrunk.append(unshrunk * n_eff / (n_eff + shrinkage_lambda))
            effective.append(n_eff)
        else:
            shrunk.append(0.0)
            effective.append(0.0)
    if strict and not all(support):
        raise SupportPreflightError(post_report)
    supported_values = [value for value, is_supported in zip(shrunk, support) if is_supported]
    common = sum(supported_values) / len(supported_values) if supported_values else 0.0
    residuals = tuple(value - common if is_supported else 0.0 for value, is_supported in zip(shrunk, support))
    return CalibrationFit(
        provisional_global,
        provisional_global + common,
        common,
        residuals,
        tuple(support),
        tuple(effective),
        tuple(post_report["zones"]),
        medians,
        projected_medians,
    )


def _cross_validate(
    residual_cells: dict[tuple[str, int, int], list[float]],
    projected_cells: dict[tuple[str, int, int], list[float]],
    settings: TrainingSettings,
) -> tuple[float, dict[str, Any]]:
    folds = sorted({(shape, plateau) for shape, plateau, _zone in residual_cells})
    candidate_results: dict[str, Any] = {}
    for candidate in LAMBDA_GRID:
        scores_by_shape: dict[str, list[float]] = {"cup": [], "spoon": []}
        fold_reports: list[dict[str, Any]] = []
        for held_shape, held_plateau in folds:
            training_residuals = {
                key: values
                for key, values in residual_cells.items()
                if not (key[0] == held_shape and key[1] == held_plateau)
            }
            training_projected = {
                key: values
                for key, values in projected_cells.items()
                if key in training_residuals
            }
            fit = _fit_from_cells(training_residuals, training_projected, settings, candidate, strict=False)
            zone_scores: list[float] = []
            unsupported_scored: list[int] = []
            for zone in range(64):
                key = (held_shape, held_plateau, zone)
                values = residual_cells.get(key)
                if not values:
                    continue
                correction = fit.final_global_mm + (fit.residuals_mm[zone] if fit.support_mask[zone] else 0.0)
                zone_scores.append(_median([abs(correction - value) for value in values]))
                if not fit.support_mask[zone]:
                    unsupported_scored.append(zone)
            if not zone_scores:
                continue
            height_score = sum(zone_scores) / len(zone_scores)
            scores_by_shape[held_shape].append(height_score)
            fold_reports.append(
                {
                    "held_shape": held_shape,
                    "held_plateau": held_plateau,
                    "height_score_mm": height_score,
                    "scored_zones": len(zone_scores),
                    "fold_unsupported_zones_scored_with_zero_residual": unsupported_scored,
                }
            )
        shape_scores = {
            shape: (sum(values) / len(values) if values else None) for shape, values in scores_by_shape.items()
        }
        if any(value is None for value in shape_scores.values()):
            raise ValueError("cross-validation requires scored cup and spoon plateaus")
        overall = (float(shape_scores["cup"]) + float(shape_scores["spoon"])) / 2.0
        candidate_results[str(candidate)] = {
            "overall_score_mm": overall,
            "shape_scores_mm": shape_scores,
            "folds": fold_reports,
        }
    selected = min(
        LAMBDA_GRID,
        key=lambda value: (round(float(candidate_results[str(value)]["overall_score_mm"]), 12), -value),
    )
    return selected, {
        "lambda_grid": list(LAMBDA_GRID),
        "score_hierarchy": ["median_frames_to_zone_height", "mean_zones_to_height", "mean_heights_to_shape", "mean_cup_spoon"],
        "larger_lambda_tie_break_after_decimal_places": 12,
        "selected_lambda": selected,
        "candidates": candidate_results,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - position) + ordered[high] * (position - low))


def _slope_diagnostics(fit: CalibrationFit, settings: TrainingSettings) -> dict[str, Any]:
    rng = random.Random(settings.bootstrap_seed)
    result: dict[str, Any] = {}
    trigger_counts: dict[str, int] = {}
    for shape in ("cup", "spoon"):
        zones: list[dict[str, Any]] = []
        triggers = 0
        for zone in range(64):
            points = sorted(
                (
                    fit.retained_cell_projected_medians[key],
                    value - fit.final_global_mm - fit.residuals_mm[zone],
                )
                for key, value in fit.retained_cell_medians.items()
                if key[0] == shape and key[2] == zone
            )
            if len(points) < 6:
                zones.append({"zone": zone, "supported": False, "plateau_groups": len(points)})
                continue
            distances = [point[0] for point in points]
            residuals = [point[1] for point in points]
            slope = theil_sen_slope(distances, residuals)
            bootstraps: list[float] = []
            attempts = 0
            while len(bootstraps) < settings.bootstrap_samples and attempts < settings.bootstrap_samples * 10:
                attempts += 1
                sample = [points[rng.randrange(len(points))] for _ in points]
                if len({point[0] for point in sample}) < 2:
                    continue
                bootstraps.append(theil_sen_slope([point[0] for point in sample], [point[1] for point in sample]))
            low = _percentile(bootstraps, 0.025)
            high = _percentile(bootstraps, 0.975)
            flagged = (low > 0.0 or high < 0.0) and abs(slope) >= 0.02
            triggers += int(flagged)
            zones.append(
                {
                    "zone": zone,
                    "supported": True,
                    "plateau_groups": len(points),
                    "theil_sen_slope_mm_per_mm": slope,
                    "unadjusted_bootstrap_95_percent_interval": [low, high],
                    "engineering_range_dependence_flag": flagged,
                }
            )
        trigger_counts[shape] = triggers
        result[shape] = zones
    inadequate = any(value >= 8 for value in trigger_counts.values())
    return {
        "bootstrap_seed": settings.bootstrap_seed,
        "bootstrap_samples": settings.bootstrap_samples,
        "multiple_testing_correction": None,
        "interpretation": "engineering adequacy heuristic; not a formal statistical conclusion",
        "per_shape_zone": result,
        "trigger_counts": trigger_counts,
        "constant_additive_engineering_inadequacy_flag": inadequate,
    }


def _git_version(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, text=True, capture_output=True, check=True
            ).stdout.strip()
        )
        return {"git_revision": revision, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "git_dirty": None}


def fit_manifest(
    manifest: TrainingManifest,
    *,
    mode: str = "strict",
    preflight_only: bool = False,
) -> dict[str, Any]:
    if mode not in ("strict", "diagnostic"):
        raise ValueError("fitter mode must be strict or diagnostic")
    datasets = tuple(_load_dataset(entry, manifest.settings) for entry in manifest.entries)
    compatibility = validate_shared_compatibility(datasets)
    residual_cells, projected_cells, plateau_report = _classification_cells(datasets, manifest.settings)
    preflight = _coverage_report(residual_cells, manifest.settings, stage="preflight")
    preflight.update({"plateaus": plateau_report, "compatibility": compatibility})
    manifest.coverage_report.parent.mkdir(parents=True, exist_ok=True)
    with manifest.coverage_report.open("w", encoding="utf-8") as handle:
        json.dump(preflight, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if preflight_only:
        return preflight
    if not preflight["supported_all_64"]:
        # cup_spoon_ascending_v1 is terminal on failed support.  Diagnostic
        # zero-residual artifacts are only meaningful for a separately named
        # training regime; they may not bypass this workflow's preflight.
        raise SupportPreflightError(preflight)

    selected_lambda, cv = _cross_validate(residual_cells, projected_cells, manifest.settings)
    fit = _fit_from_cells(
        residual_cells,
        projected_cells,
        manifest.settings,
        selected_lambda,
        strict=mode == "strict",
    )
    diagnostics = _slope_diagnostics(fit, manifest.settings)
    reference = datasets[0]
    config = reference.sensor_config
    angles = shape_replay.canonical_emitter_angles(
        int(config["rows"]), int(config["cols"]), float(config["fov_h_deg"]), float(config["fov_v_deg"])
    )
    rays = shape_replay.canonical_local_rays(
        int(config["rows"]), int(config["cols"]), float(config["fov_h_deg"]), float(config["fov_v_deg"])
    )
    factors = shape_replay.projection_factors(
        int(config["rows"]), int(config["cols"]), float(config["fov_h_deg"]), float(config["fov_v_deg"])
    )
    repo_root = Path(__file__).resolve().parents[2]
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "training_regime_id": manifest.training_regime_id,
        "distance_parameter_provenance": "new distance parameters fitted only from cup and spoon ascending inputs",
        "fixed_profile_provenance": "pre-existing fixed values may contain historical descending provenance",
        "descending_inputs_accessed": False,
        "sensor_profile": {
            "path": str(reference.profile.sensor_profile),
            "sha256": shape_replay.sha256_file(reference.profile.sensor_profile),
            "config": config,
        },
        "emitter_geometry": {
            "canonical_order": "row-major authored emitter order",
            "sensor_forward_axis_local": [1.0, 0.0, 0.0],
            "angles_deg": [
                {"zone": emitter.zone, "azimuth_deg": emitter.azimuth_deg, "elevation_deg": emitter.elevation_deg}
                for emitter in angles
            ],
            "local_spherical_unit_rays": [list(ray) for ray in rays],
            "projection_factors": list(factors),
        },
        "compatibility_tolerances": {
            "emitter_angle_deg": 1.0e-12,
            "tcp_to_sensor_z_mm": 0.10,
            "orientation_deg": 0.01,
            "sensor_xy_m_is_not_mount_compatibility_input": True,
        },
        "compatibility": compatibility,
        "fixed_profiles": {
            dataset.profile.name: {
                "profile_sha256": shape_replay.sha256_file(dataset.entry.profile_path),
                "stl_sha256": shape_replay.sha256_file(dataset.profile.stl_path),
                "tcp_to_sensor_z_m": dataset.profile.tcp_to_sensor_z_m,
                "sensor_quat_wxyz": list(dataset.profile.sensor_quat_wxyz),
                "zone_transform_applied_once_at_ingestion": dataset.profile.zone_transform,
            }
            for dataset in datasets
        },
        "training_inputs": {
            "manifest": {"path": str(manifest.path), "sha256": shape_replay.sha256_file(manifest.path)},
            "entries": {
                entry.name: {
                    "direction": "ascending",
                    "robot_csv": {"path": str(entry.robot_csv), "sha256": shape_replay.sha256_file(entry.robot_csv)},
                    "real_tof_csv": {"path": str(entry.real_tof_csv), "sha256": shape_replay.sha256_file(entry.real_tof_csv)},
                    "simulation_csv": {"path": str(entry.simulation_csv), "sha256": shape_replay.sha256_file(entry.simulation_csv)},
                }
                for entry in manifest.entries
            },
        },
        "classification": {
            **manifest.settings.__dict__,
            "table_gate_definition": "independent absolute difference from analytically projected table intersection",
            "world_ray_origin": "(sensor_xy_m.x, sensor_xy_m.y, tcp_z_m - tcp_to_sensor_z_m)",
            "stl_transform_order": ["subtract_source_origin", "source_units_to_m", "rotate_z_yaw", "translate_shape_pose"],
        },
        "plateaus": plateau_report,
        "preflight": preflight,
        "parameters": {
            "newly_fitted_parameter_scope": "ascending-only distance layer",
            "sign_convention": "comparison_mm = projected_mm + global_offset_mm + zone_residual_mm",
            "target_residual_definition": "real_mm - projected_mm",
            "rounding_rule": "round-to-nearest ties-to-even, then clamp; modes derive independently from rtx_range_m",
            "selected_lambda": selected_lambda,
            "provisional_global_offset_mm": fit.provisional_global_mm,
            "common_component_transferred_to_global_mm": fit.common_component_mm,
            "global_offset_mm": fit.final_global_mm,
            "zone_residuals_mm": list(fit.residuals_mm),
            "support_mask": list(fit.support_mask),
            "effective_sample_sizes_from_plateau_groups": list(fit.effective_sample_sizes),
            "plateau_coverage": list(fit.coverage),
        },
        "cross_validation": cv,
        "range_dependence_diagnostics": diagnostics,
        "validity_policy": "raw, projected, and comparison preserve the exact same selected-return validity mask",
        "units": {"rtx_range": "m", "reported_distance": "mm", "offsets": "mm"},
        "code_version": {
            **_git_version(repo_root),
            "implementation_version": 1,
            "source_sha256": {
                "distance_calibration.py": shape_replay.sha256_file(__file__),
                "shape_replay.py": shape_replay.sha256_file(shape_replay.__file__),
            },
        },
    }
    manifest.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    with manifest.output_artifact.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return artifact
