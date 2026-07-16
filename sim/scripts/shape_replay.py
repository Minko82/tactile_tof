"""Dependency-light helpers for replaying recorded shape experiments.

The Isaac Sim entrypoint imports this module before starting SimulationApp, so
all geometry, timestamp alignment, calibration, and result comparison logic is
kept independent of Isaac/Omniverse.  Only the optional heatmap writer imports
NumPy and Matplotlib.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ZONE_TRANSFORMS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "mirror",
    "mirror_rot90",
    "mirror_rot180",
    "mirror_rot270",
)


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _vec(values: Any, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{field_name} must contain {length} numbers")
    return tuple(_finite_float(value, field_name) for value in values)


def _resolve_path(profile_path: Path, value: Any, field_name: str) -> Path:
    if not value:
        raise ValueError(f"{field_name} is required")
    path = Path(str(value))
    if not path.is_absolute():
        path = profile_path.parent / path
    return path.resolve()


@dataclass(frozen=True)
class DirectionFiles:
    robot_csv: Path
    tof_csv: Path


@dataclass(frozen=True)
class MeshPose:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class ShapeExperimentProfile:
    name: str
    profile_path: Path
    stl_path: Path
    sensor_profile: Path
    directions: dict[str, DirectionFiles]
    stl_units_to_m: float = 0.001
    mesh_origin_source_units: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_pose: MeshPose = MeshPose()
    table_top_z_m: float = 0.0
    tcp_to_sensor_z_m: float = 0.09
    timestamp_lag_ms: float = 65.0
    sensor_xy_m: tuple[float, float] = (0.0, 0.0)
    sensor_quat_wxyz: tuple[float, float, float, float] = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
    zone_transform: str = "identity"
    visual_rgb: tuple[float, float, float] = (0.008, 0.008, 0.01)
    nonvisual_material: str = "RubberStandard"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if self.stl_units_to_m <= 0.0:
            raise ValueError("stl_units_to_m must be positive")
        if self.tcp_to_sensor_z_m <= 0.0:
            raise ValueError("tcp_to_sensor_z_m must be positive")
        if self.timestamp_lag_ms < 0.0:
            raise ValueError("timestamp_lag_ms must be non-negative")
        if self.zone_transform not in ZONE_TRANSFORMS:
            raise ValueError(f"unsupported zone_transform {self.zone_transform!r}")
        if set(self.directions) != {"ascending", "descending"}:
            raise ValueError("directions must define exactly ascending and descending")
        norm = math.sqrt(sum(value * value for value in self.sensor_quat_wxyz))
        if abs(norm - 1.0) > 1.0e-5:
            raise ValueError("sensor_quat_wxyz must be normalized")

    @classmethod
    def from_json(cls, path: str | Path, *, require_files: bool = True) -> "ShapeExperimentProfile":
        profile_path = Path(path).resolve()
        with profile_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        pose_data = data.get("mesh_pose", {})
        material_data = data.get("material", {})
        direction_data = data.get("directions", {})
        directions: dict[str, DirectionFiles] = {}
        for direction in ("ascending", "descending"):
            values = direction_data.get(direction, {})
            directions[direction] = DirectionFiles(
                robot_csv=_resolve_path(profile_path, values.get("robot_csv"), f"directions.{direction}.robot_csv"),
                tof_csv=_resolve_path(profile_path, values.get("tof_csv"), f"directions.{direction}.tof_csv"),
            )

        profile = cls(
            name=str(data.get("name", "")).strip(),
            profile_path=profile_path,
            stl_path=_resolve_path(profile_path, data.get("stl_path"), "stl_path"),
            sensor_profile=_resolve_path(profile_path, data.get("sensor_profile"), "sensor_profile"),
            directions=directions,
            stl_units_to_m=_finite_float(data.get("stl_units_to_m", 0.001), "stl_units_to_m"),
            mesh_origin_source_units=_vec(data.get("mesh_origin_source_units", (0, 0, 0)), 3, "mesh_origin_source_units"),
            mesh_pose=MeshPose(
                x_m=_finite_float(pose_data.get("x_m", 0.0), "mesh_pose.x_m"),
                y_m=_finite_float(pose_data.get("y_m", 0.0), "mesh_pose.y_m"),
                yaw_deg=_finite_float(pose_data.get("yaw_deg", 0.0), "mesh_pose.yaw_deg"),
            ),
            table_top_z_m=_finite_float(data.get("table_top_z_m", 0.0), "table_top_z_m"),
            tcp_to_sensor_z_m=_finite_float(data.get("tcp_to_sensor_z_m", 0.09), "tcp_to_sensor_z_m"),
            timestamp_lag_ms=_finite_float(data.get("timestamp_lag_ms", 65.0), "timestamp_lag_ms"),
            sensor_xy_m=_vec(data.get("sensor_xy_m", (0, 0)), 2, "sensor_xy_m"),
            sensor_quat_wxyz=_vec(
                data.get("sensor_quat_wxyz", (math.sqrt(0.5), 0, math.sqrt(0.5), 0)),
                4,
                "sensor_quat_wxyz",
            ),
            zone_transform=str(data.get("zone_transform", "identity")),
            visual_rgb=_vec(material_data.get("visual_rgb", (0.008, 0.008, 0.01)), 3, "material.visual_rgb"),
            nonvisual_material=str(material_data.get("nonvisual_type", "RubberStandard")),
        )
        if require_files:
            paths = [profile.stl_path, profile.sensor_profile]
            for values in profile.directions.values():
                paths.extend((values.robot_csv, values.tof_csv))
            missing = [str(candidate) for candidate in paths if not candidate.is_file()]
            if missing:
                raise FileNotFoundError("missing experiment files: " + ", ".join(missing))
        return profile


@dataclass(frozen=True)
class StlMesh:
    vertices: tuple[tuple[float, float, float], ...]
    triangle_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(self.bounds_max[index] - self.bounds_min[index] for index in range(3))


def load_stl(path: str | Path) -> StlMesh:
    """Read binary or ASCII STL without welding or changing any vertices."""

    path = Path(path)
    data = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    is_binary = False
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        is_binary = 84 + triangle_count * 50 == len(data)
    if is_binary:
        offset = 84
        for _ in range(triangle_count):
            values = struct.unpack_from("<12fH", data, offset)
            vertices.extend(
                (
                    (float(values[3]), float(values[4]), float(values[5])),
                    (float(values[6]), float(values[7]), float(values[8])),
                    (float(values[9]), float(values[10]), float(values[11])),
                )
            )
            offset += 50
    else:
        for line in data.decode("utf-8", errors="strict").splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append(tuple(float(value) for value in fields[1:4]))
        if len(vertices) % 3:
            raise ValueError(f"{path} contains an incomplete ASCII STL triangle")
        triangle_count = len(vertices) // 3
    if not vertices:
        raise ValueError(f"{path} contains no STL triangles")

    bounds_min = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    bounds_max = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    return StlMesh(tuple(vertices), int(triangle_count), bounds_min, bounds_max)


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp {value!r}") from exc


def _seconds(value: datetime) -> float:
    midnight = value.replace(hour=0, minute=0, second=0, microsecond=0)
    return value.toordinal() * 86400.0 + (value - midnight).total_seconds()


@dataclass(frozen=True)
class RobotZSample:
    timestamp: str
    time_s: float
    z_m: float


@dataclass(frozen=True)
class ToFReferenceFrame:
    timestamp: str
    time_s: float
    zones_mm: tuple[int, ...]


@dataclass(frozen=True)
class ReplaySample:
    reference_timestamp: str
    elapsed_s: float
    tcp_z_m: float
    sensor_z_m: float
    real_zones_mm: tuple[int, ...]


def load_robot_z_csv(path: str | Path) -> list[RobotZSample]:
    samples: list[RobotZSample] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError(f"{path} is missing its timestamp/z header")
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            timestamp = row[0].strip()
            samples.append(RobotZSample(timestamp, _seconds(_parse_timestamp(timestamp)), float(row[1])))
    if len(samples) < 2:
        raise ValueError(f"{path} must contain at least two robot samples")
    if any(right.time_s <= left.time_s for left, right in zip(samples, samples[1:])):
        raise ValueError(f"{path} robot timestamps must be strictly increasing")
    return samples


def load_tof_csv(path: str | Path, zones: int = 64) -> list[ToFReferenceFrame]:
    columns = [f"zone_{index:02d}" for index in range(zones)]
    frames: list[ToFReferenceFrame] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "timestamp" not in reader.fieldnames:
            raise ValueError(f"{path} is missing a timestamp column")
        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing ToF columns: {', '.join(missing)}")
        for row in reader:
            timestamp = str(row["timestamp"]).strip()
            frames.append(
                ToFReferenceFrame(
                    timestamp,
                    _seconds(_parse_timestamp(timestamp)),
                    tuple(int(float(row[column])) for column in columns),
                )
            )
    if not frames:
        raise ValueError(f"{path} contains no ToF frames")
    return frames


def interpolate_robot_z(samples: Sequence[RobotZSample], time_s: float) -> float:
    times = [sample.time_s for sample in samples]
    if time_s < times[0] or time_s > times[-1]:
        raise ValueError("requested robot timestamp is outside the recorded range")
    right = bisect.bisect_left(times, time_s)
    if right == 0:
        return samples[0].z_m
    if right == len(samples):
        return samples[-1].z_m
    if times[right] == time_s:
        return samples[right].z_m
    left = right - 1
    duration = times[right] - times[left]
    alpha = (time_s - times[left]) / duration
    return samples[left].z_m + alpha * (samples[right].z_m - samples[left].z_m)


def load_replay_samples(profile: ShapeExperimentProfile, direction: str) -> list[ReplaySample]:
    if direction not in profile.directions:
        raise ValueError(f"unsupported experiment direction {direction!r}")
    files = profile.directions[direction]
    robot = load_robot_z_csv(files.robot_csv)
    tof = load_tof_csv(files.tof_csv)
    lag_s = profile.timestamp_lag_ms / 1000.0
    overlapping: list[tuple[ToFReferenceFrame, float]] = []
    for frame in tof:
        robot_time = frame.time_s - lag_s
        if robot[0].time_s <= robot_time <= robot[-1].time_s:
            overlapping.append((frame, interpolate_robot_z(robot, robot_time)))
    if not overlapping:
        raise ValueError(f"{direction} has no robot/ToF timestamp overlap after applying {profile.timestamp_lag_ms:g} ms lag")
    start_time = overlapping[0][0].time_s
    return [
        ReplaySample(
            reference_timestamp=frame.timestamp,
            elapsed_s=frame.time_s - start_time,
            tcp_z_m=tcp_z,
            sensor_z_m=tcp_z - profile.tcp_to_sensor_z_m,
            real_zones_mm=frame.zones_mm,
        )
        for frame, tcp_z in overlapping
    ]


def transform_zones(values: Sequence[Any], transform: str, rows: int = 8, cols: int = 8) -> list[Any]:
    if transform not in ZONE_TRANSFORMS:
        raise ValueError(f"unsupported zone transform {transform!r}")
    if len(values) != rows * cols:
        raise ValueError(f"expected {rows * cols} zones, got {len(values)}")
    if rows != cols and "rot90" in transform or rows != cols and "rot270" in transform:
        raise ValueError("90-degree transforms require a square grid")
    matrix = [list(values[start : start + cols]) for start in range(0, len(values), cols)]
    if transform.startswith("mirror"):
        matrix = [list(reversed(row)) for row in matrix]
    rotations = {"identity": 0, "mirror": 0, "rot90": 1, "mirror_rot90": 1, "rot180": 2, "mirror_rot180": 2, "rot270": 3, "mirror_rot270": 3}[transform]
    for _ in range(rotations):
        matrix = [list(row) for row in zip(*reversed(matrix))]
    return [value for row in matrix for value in row]


def quaternion_rotate_vector(
    quaternion_wxyz: Sequence[float], vector_xyz: Sequence[float]
) -> tuple[float, float, float]:
    w, x, y, z = quaternion_wxyz
    vx, vy, vz = vector_xyz
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


@dataclass(frozen=True)
class FootprintGrid:
    min_x_m: float
    min_y_m: float
    cell_m: float
    width: int
    height: int
    occupied: bytes

    def contains(self, x_m: float, y_m: float) -> bool:
        col = int(math.floor((x_m - self.min_x_m) / self.cell_m))
        row = int(math.floor((y_m - self.min_y_m) / self.cell_m))
        return 0 <= col < self.width and 0 <= row < self.height and bool(self.occupied[row * self.width + col])

    def centroid(self) -> tuple[float, float]:
        points = [
            (self.min_x_m + (index % self.width + 0.5) * self.cell_m, self.min_y_m + (index // self.width + 0.5) * self.cell_m)
            for index, value in enumerate(self.occupied)
            if value
        ]
        if not points:
            return (0.0, 0.0)
        return (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))


def _point_in_triangle_2d(
    px: float,
    py: float,
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    v0x, v0y = c[0] - a[0], c[1] - a[1]
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = px - a[0], py - a[1]
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(denominator) < 1.0e-14:
        return False
    inverse = 1.0 / denominator
    u = (dot11 * dot02 - dot01 * dot12) * inverse
    v = (dot00 * dot12 - dot01 * dot02) * inverse
    return u >= -1.0e-9 and v >= -1.0e-9 and u + v <= 1.0 + 1.0e-9


def build_footprint_grid(
    mesh: StlMesh,
    units_to_m: float,
    origin_source_units: Sequence[float],
    *,
    cell_m: float = 0.0015,
) -> FootprintGrid:
    """Rasterize the exact triangle projection for fast rigid-pose fitting."""

    origin_x, origin_y = float(origin_source_units[0]), float(origin_source_units[1])
    points = [((vertex[0] - origin_x) * units_to_m, (vertex[1] - origin_y) * units_to_m) for vertex in mesh.vertices]
    min_x = min(point[0] for point in points) - cell_m
    min_y = min(point[1] for point in points) - cell_m
    max_x = max(point[0] for point in points) + cell_m
    max_y = max(point[1] for point in points) + cell_m
    width = max(1, int(math.ceil((max_x - min_x) / cell_m)))
    height = max(1, int(math.ceil((max_y - min_y) / cell_m)))
    occupied = bytearray(width * height)
    for start in range(0, len(points), 3):
        triangle = points[start : start + 3]
        low_col = max(0, int(math.floor((min(point[0] for point in triangle) - min_x) / cell_m)))
        high_col = min(width - 1, int(math.floor((max(point[0] for point in triangle) - min_x) / cell_m)))
        low_row = max(0, int(math.floor((min(point[1] for point in triangle) - min_y) / cell_m)))
        high_row = min(height - 1, int(math.floor((max(point[1] for point in triangle) - min_y) / cell_m)))
        for row in range(low_row, high_row + 1):
            py = min_y + (row + 0.5) * cell_m
            for col in range(low_col, high_col + 1):
                px = min_x + (col + 0.5) * cell_m
                if _point_in_triangle_2d(px, py, triangle[0], triangle[1], triangle[2]):
                    occupied[row * width + col] = 1
    # Preserve thin vertical walls/handles that can project to sub-cell lines.
    for x_m, y_m in points:
        col = int(math.floor((x_m - min_x) / cell_m))
        row = int(math.floor((y_m - min_y) / cell_m))
        if 0 <= col < width and 0 <= row < height:
            occupied[row * width + col] = 1
    return FootprintGrid(min_x, min_y, cell_m, width, height, bytes(occupied))


def estimate_zone_table_offsets(samples: Sequence[ReplaySample], zones: int = 64) -> list[float]:
    """Estimate per-zone table offsets around the robust global mount offset."""

    candidates: list[list[float]] = [[] for _ in range(zones)]
    stable_samples: list[ReplaySample] = []
    for index, sample in enumerate(samples):
        previous_z = samples[max(0, index - 1)].tcp_z_m
        next_z = samples[min(len(samples) - 1, index + 1)].tcp_z_m
        if abs(next_z - previous_z) <= 0.0025:
            stable_samples.append(sample)
    source = stable_samples or list(samples)
    global_offset_mm = _estimate_global_mount_offset_mm(source)
    for sample in source:
        for index, distance in enumerate(sample.real_zones_mm):
            if distance > 0:
                candidates[index].append(sample.tcp_z_m * 1000.0 - distance)
    offsets: list[float] = []
    for values in candidates:
        if not values:
            offsets.append(global_offset_mm)
            continue
        near_table = sorted(value for value in values if abs(value - global_offset_mm) <= 18.0)
        ordered = near_table or sorted(values)
        midpoint = len(ordered) // 2
        offsets.append(ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0)
    return offsets


def _estimate_global_mount_offset_mm(samples: Sequence[ReplaySample]) -> float:
    groups: dict[int, list[ReplaySample]] = {}
    for sample in samples:
        groups.setdefault(int(round(sample.tcp_z_m * 1000.0)), []).append(sample)
    estimates: list[float] = []
    for group in groups.values():
        if len(group) < 4:
            continue
        values = sorted(value for sample in group for value in sample.real_zones_mm if value > 0)
        if not values:
            continue
        midpoint = len(values) // 2
        median_distance = values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2.0
        median_tcp = sorted(sample.tcp_z_m for sample in group)[len(group) // 2] * 1000.0
        estimates.append(median_tcp - median_distance)
    if not estimates:
        return 90.0
    ordered = sorted(estimates)
    midpoint = len(ordered) // 2
    return ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def observed_object_mask(sample: ReplaySample, zone_offsets_mm: Sequence[float], threshold_mm: float = 14.0) -> list[bool]:
    mask: list[bool] = []
    for index, distance in enumerate(sample.real_zones_mm):
        expected_table = sample.tcp_z_m * 1000.0 - zone_offsets_mm[index]
        mask.append(distance <= 0 or distance < expected_table - threshold_mm)
    return mask


def stable_plateau_masks(samples: Sequence[ReplaySample], *, min_frames: int = 4) -> list[tuple[ReplaySample, list[bool]]]:
    offsets = estimate_zone_table_offsets(samples)
    groups: dict[int, list[ReplaySample]] = {}
    for index, sample in enumerate(samples):
        previous_z = samples[max(0, index - 1)].tcp_z_m
        next_z = samples[min(len(samples) - 1, index + 1)].tcp_z_m
        if abs(next_z - previous_z) > 0.0025:
            continue
        key = int(round(sample.tcp_z_m * 1000.0))
        groups.setdefault(key, []).append(sample)
    plateaus: list[tuple[ReplaySample, list[bool]]] = []
    for group in groups.values():
        if len(group) < min_frames:
            continue
        masks = [observed_object_mask(sample, offsets) for sample in group]
        majority = [sum(bool(mask[index]) for mask in masks) >= len(masks) / 2.0 for index in range(64)]
        plateaus.append((group[len(group) // 2], majority))
    return sorted(plateaus, key=lambda item: item[0].elapsed_s)


def _zone_plane_points(
    sample: ReplaySample,
    plane_z_m: float,
    fov_h_deg: float,
    fov_v_deg: float,
    sensor_xy_m: Sequence[float],
    tcp_to_sensor_z_m: float | None = None,
    rows: int = 8,
    cols: int = 8,
) -> list[tuple[float, float]]:
    sensor_z_m = sample.sensor_z_m if tcp_to_sensor_z_m is None else sample.tcp_z_m - tcp_to_sensor_z_m
    vertical_drop = max(sensor_z_m - plane_z_m, 0.0)
    points: list[tuple[float, float]] = []
    for row in range(rows):
        elevation = fov_v_deg / 2.0 - fov_v_deg * row / max(rows - 1, 1)
        for col in range(cols):
            azimuth = -fov_h_deg / 2.0 + fov_h_deg * col / max(cols - 1, 1)
            points.append(
                (
                    float(sensor_xy_m[0]) + vertical_drop * math.tan(math.radians(elevation)),
                    float(sensor_xy_m[1]) + vertical_drop * math.tan(math.radians(azimuth)),
                )
            )
    return points


def _pose_score(points_and_observed: Sequence[tuple[float, float, bool]], footprint: FootprintGrid, pose: MeshPose) -> float:
    radians = math.radians(pose.yaw_deg)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    intersection = 0
    union = 0
    for world_x, world_y, observed in points_and_observed:
        dx = world_x - pose.x_m
        dy = world_y - pose.y_m
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        predicted = footprint.contains(local_x, local_y)
        intersection += int(predicted and observed)
        union += int(predicted or observed)
    return intersection / union if union else 0.0


@dataclass(frozen=True)
class CalibrationResult:
    mesh_pose: MeshPose
    zone_transform: str
    score: float
    centered_baseline_score: float
    tcp_to_sensor_z_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "mesh_pose": {"x_m": self.mesh_pose.x_m, "y_m": self.mesh_pose.y_m, "yaw_deg": self.mesh_pose.yaw_deg},
            "zone_transform": self.zone_transform,
            "silhouette_iou": self.score,
            "centered_baseline_iou": self.centered_baseline_score,
            "tcp_to_sensor_z_m": self.tcp_to_sensor_z_m,
        }


def estimate_tcp_to_sensor_offset(samples: Sequence[ReplaySample]) -> float:
    stable: list[ReplaySample] = []
    for index, sample in enumerate(samples):
        previous_z = samples[max(0, index - 1)].tcp_z_m
        next_z = samples[min(len(samples) - 1, index + 1)].tcp_z_m
        if abs(next_z - previous_z) <= 0.0025:
            stable.append(sample)
    return _estimate_global_mount_offset_mm(stable or samples) / 1000.0


def calibrate_rigid_pose(
    profile: ShapeExperimentProfile,
    samples_by_direction: dict[str, Sequence[ReplaySample]],
    *,
    fov_h_deg: float = 45.0,
    fov_v_deg: float = 45.0,
    footprint_cell_m: float = 0.0015,
) -> CalibrationResult:
    """Fit one XY/yaw/zone transform to stable plateaus in both directions."""

    all_samples = [sample for direction in ("ascending", "descending") for sample in samples_by_direction[direction]]
    offset = estimate_tcp_to_sensor_offset(all_samples)
    mesh = load_stl(profile.stl_path)
    footprint = build_footprint_grid(mesh, profile.stl_units_to_m, profile.mesh_origin_source_units, cell_m=footprint_cell_m)
    plane_z = profile.table_top_z_m + (
        (mesh.bounds_min[2] + mesh.bounds_max[2]) * 0.5 - profile.mesh_origin_source_units[2]
    ) * profile.stl_units_to_m
    plateau_data: list[tuple[list[tuple[float, float]], list[bool]]] = []
    for direction in ("ascending", "descending"):
        for sample, mask in stable_plateau_masks(samples_by_direction[direction]):
            plateau_data.append(
                (
                    _zone_plane_points(
                        sample,
                        plane_z,
                        fov_h_deg,
                        fov_v_deg,
                        profile.sensor_xy_m,
                        offset,
                    ),
                    mask,
                )
            )
    if not plateau_data:
        raise ValueError("no stable plateaus were found for rigid pose calibration")

    footprint_center = footprint.centroid()
    best: tuple[float, MeshPose, str] = (-1.0, MeshPose(), "identity")
    baseline = 0.0
    for transform in ZONE_TRANSFORMS:
        transformed = [(points, transform_zones(mask, transform)) for points, mask in plateau_data]
        evidence = [point for points, mask in transformed for point, observed in zip(points, mask) if observed]
        if evidence:
            evidence_center = (
                sum(point[0] for point in evidence) / len(evidence),
                sum(point[1] for point in evidence) / len(evidence),
            )
        else:
            evidence_center = tuple(profile.sensor_xy_m)
        flat = [(point[0], point[1], observed) for points, mask in transformed for point, observed in zip(points, mask)]
        for yaw in range(0, 360, 15):
            radians = math.radians(yaw)
            rotated_center = (
                math.cos(radians) * footprint_center[0] - math.sin(radians) * footprint_center[1],
                math.sin(radians) * footprint_center[0] + math.cos(radians) * footprint_center[1],
            )
            guess_x = evidence_center[0] - rotated_center[0]
            guess_y = evidence_center[1] - rotated_center[1]
            baseline = max(baseline, _pose_score(flat, footprint, MeshPose(0.0, 0.0, float(yaw))))
            for x_step in range(-4, 5):
                for y_step in range(-4, 5):
                    pose = MeshPose(guess_x + x_step * 0.01, guess_y + y_step * 0.01, float(yaw))
                    score = _pose_score(flat, footprint, pose)
                    if score > best[0]:
                        best = (score, pose, transform)

    transformed = [(points, transform_zones(mask, best[2])) for points, mask in plateau_data]
    flat = [(point[0], point[1], observed) for points, mask in transformed for point, observed in zip(points, mask)]
    coarse_pose = best[1]
    for yaw_delta in range(-15, 16, 3):
        for x_step in range(-5, 6):
            for y_step in range(-5, 6):
                pose = MeshPose(
                    coarse_pose.x_m + x_step * 0.002,
                    coarse_pose.y_m + y_step * 0.002,
                    (coarse_pose.yaw_deg + yaw_delta) % 360.0,
                )
                score = _pose_score(flat, footprint, pose)
                if score > best[0]:
                    best = (score, pose, best[2])

    return CalibrationResult(best[1], best[2], best[0], baseline, offset)


@dataclass
class ShapeReplayRun:
    profile: ShapeExperimentProfile
    direction: str
    samples: list[ReplaySample]
    output_dir: Path
    sim_ticks: list[int] = field(default_factory=list)

    @property
    def sensor_prim_path(self) -> str:
        # Isaac Sim 5.1's Replicator render-product wrapper does not accept a
        # nested RTX lidar prim path reliably, so the sensor stays root-level.
        return "/VL53L8CX"


def _flatten_matrix(matrix: Any) -> list[int]:
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    return [int(value) for row in matrix for value in (row.tolist() if hasattr(row, "tolist") else row)]


def _flatten_auxiliary_matrix(matrix: Any, zones: int = 64) -> list[Any]:
    if matrix is None:
        return [""] * zones
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    values: list[Any] = []
    for row in matrix:
        if hasattr(row, "tolist"):
            row = row.tolist()
        values.extend("" if value is None else value for value in row)
    if len(values) < zones:
        values.extend([""] * (zones - len(values)))
    return values[:zones]


def _frame_metrics(real: Sequence[int], simulated: Sequence[int]) -> dict[str, float | int | None]:
    real_valid_values = [float(value) for value in real if value > 0]
    sim_valid_values = [float(value) for value in simulated if value > 0]
    paired = [(float(s), float(r)) for r, s in zip(real, simulated) if r > 0 and s > 0]
    errors = [sim - reference for sim, reference in paired]
    real_no_return = {index for index, value in enumerate(real) if value <= 0}
    sim_no_return = {index for index, value in enumerate(simulated) if value <= 0}
    union = real_no_return | sim_no_return
    return {
        "real_valid_zones": len(real_valid_values),
        "sim_valid_zones": len(sim_valid_values),
        "real_mean_distance_mm": sum(real_valid_values) / len(real_valid_values) if real_valid_values else None,
        "sim_mean_distance_mm": sum(sim_valid_values) / len(sim_valid_values) if sim_valid_values else None,
        "paired_valid_zones": len(paired),
        "mae_mm": sum(abs(error) for error in errors) / len(errors) if errors else None,
        "bias_mm": sum(errors) / len(errors) if errors else None,
        "no_return_iou": len(real_no_return & sim_no_return) / len(union) if union else 1.0,
    }


def build_comparison(
    samples: Sequence[ReplaySample],
    frames: Sequence[Any],
    zone_transform: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = min(len(samples), len(frames))
    rows: list[dict[str, Any]] = []
    zone_errors: list[list[float]] = [[] for _ in range(64)]
    all_errors: list[float] = []
    real_valid = 0
    sim_valid = 0
    no_return_ious: list[float] = []
    for index in range(count):
        sample = samples[index]
        real = [int(value) for value in transform_zones(sample.real_zones_mm, zone_transform)]
        simulated = _flatten_matrix(frames[index].distances_mm)
        metrics = _frame_metrics(real, simulated)
        rows.append(
            {
                "frame_index": index,
                "reference_timestamp": sample.reference_timestamp,
                "elapsed_s": sample.elapsed_s,
                "tcp_z_m": sample.tcp_z_m,
                "sensor_z_m": sample.sensor_z_m,
                **metrics,
            }
        )
        real_valid += int(metrics["real_valid_zones"])
        sim_valid += int(metrics["sim_valid_zones"])
        no_return_ious.append(float(metrics["no_return_iou"]))
        for zone, (reference, sim) in enumerate(zip(real, simulated)):
            if reference > 0 and sim > 0:
                error = float(sim - reference)
                zone_errors[zone].append(error)
                all_errors.append(error)
    denominator = max(count * 64, 1)
    plateau_groups: dict[int, list[int]] = {}
    for index, sample in enumerate(samples[:count]):
        plateau_groups.setdefault(int(round(sample.tcp_z_m * 1000.0)), []).append(index)
    plateau_summaries: list[dict[str, Any]] = []
    for tcp_z_mm, indices in plateau_groups.items():
        if len(indices) < 4:
            continue
        real_values: list[int] = []
        sim_values: list[int] = []
        for index in indices:
            real_values.extend(int(value) for value in transform_zones(samples[index].real_zones_mm, zone_transform))
            sim_values.extend(_flatten_matrix(frames[index].distances_mm))
        plateau_summaries.append({"tcp_z_mm": tcp_z_mm, "frames": len(indices), **_frame_metrics(real_values, sim_values)})
    summary = {
        "frames": count,
        "real_valid_rate": real_valid / denominator,
        "sim_valid_rate": sim_valid / denominator,
        "distance_mae_mm": sum(abs(value) for value in all_errors) / len(all_errors) if all_errors else None,
        "distance_bias_mm": sum(all_errors) / len(all_errors) if all_errors else None,
        "mean_no_return_iou": sum(no_return_ious) / len(no_return_ious) if no_return_ious else None,
        "per_zone_bias_mm": [sum(values) / len(values) if values else None for values in zone_errors],
        "per_zone_paired_samples": [len(values) for values in zone_errors],
        "plateaus": plateau_summaries,
    }
    return rows, summary


def _write_comparison_graph(path: Path, rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> bool:
    """Plot aligned real/sim distance, return coverage, and frame error."""

    if not rows:
        return False
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return _write_dependency_free_comparison_graph(path, rows)

    elapsed = [float(row["elapsed_s"]) for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(13.0, 10.0), sharex=True)

    axes[0].plot(
        elapsed,
        [row["real_mean_distance_mm"] for row in rows],
        color="#1F2937",
        linewidth=1.8,
        label="Real VL53L5CX",
    )
    axes[0].plot(
        elapsed,
        [row["sim_mean_distance_mm"] for row in rows],
        color="#E76F51",
        linewidth=1.8,
        label="Raw RTX simulation",
    )
    axes[0].plot(
        elapsed,
        [float(row["sensor_z_m"]) * 1000.0 for row in rows],
        color="#94A3B8",
        linewidth=1.0,
        linestyle="--",
        label="Sensor height above table",
    )
    axes[0].set_ylabel("Distance (mm)")
    axes[0].set_title("Mean valid-zone distance")
    axes[0].legend(loc="best", frameon=False, ncol=3)

    axes[1].plot(
        elapsed,
        [row["real_valid_zones"] for row in rows],
        color="#1F2937",
        linewidth=1.8,
        label="Real valid zones",
    )
    axes[1].plot(
        elapsed,
        [row["sim_valid_zones"] for row in rows],
        color="#E76F51",
        linewidth=1.8,
        label="Simulation valid zones",
    )
    axes[1].set_ylabel("Valid zones (of 64)")
    axes[1].set_ylim(-1, 65)
    axes[1].set_title("Return coverage")
    axes[1].legend(loc="best", frameon=False, ncol=2)

    axes[2].plot(
        elapsed,
        [row["mae_mm"] for row in rows],
        color="#D97706",
        linewidth=1.7,
        label="Frame MAE",
    )
    axes[2].plot(
        elapsed,
        [row["bias_mm"] for row in rows],
        color="#2563EB",
        linewidth=1.4,
        label="Signed bias",
    )
    axes[2].axhline(0.0, color="#64748B", linewidth=0.9)
    axes[2].set_xlabel("Elapsed experiment time (s)")
    axes[2].set_ylabel("Error (mm)")
    axes[2].set_title("Distance error where both sensors returned a value")
    axes[2].legend(loc="best", frameon=False, ncol=2)

    for axis in axes:
        axis.grid(True, color="#CBD5E1", alpha=0.55, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    overall_mae = summary.get("distance_mae_mm")
    suffix = f" | overall MAE {overall_mae:.1f} mm" if isinstance(overall_mae, (int, float)) else ""
    figure.suptitle(f"Real versus simulation: {summary.get('direction', 'shape replay')}{suffix}", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)
    return True


def _write_dependency_free_comparison_graph(path: Path, rows: Sequence[dict[str, Any]]) -> bool:
    """Write a compact trend PNG when Matplotlib is unavailable in Isaac Python."""

    width, height = 1200, 780
    pixels = bytearray([255] * width * height * 3)
    left, right = 70, width - 28
    panel_height = 210
    panel_tops = (28, 282, 536)
    series = (
        ([row["real_mean_distance_mm"] for row in rows], [row["sim_mean_distance_mm"] for row in rows]),
        ([row["real_valid_zones"] for row in rows], [row["sim_valid_zones"] for row in rows]),
        ([row["mae_mm"] for row in rows], [row["bias_mm"] for row in rows]),
    )

    def draw_line(start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            if 0 <= x0 < width and 0 <= y0 < height:
                offset = (y0 * width + x0) * 3
                pixels[offset : offset + 3] = bytes(color)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    colors = ((31, 41, 55), (231, 111, 81))
    for top, panel_series in zip(panel_tops, series):
        bottom = top + panel_height
        draw_line((left, top), (left, bottom), (100, 116, 139))
        draw_line((left, bottom), (right, bottom), (100, 116, 139))
        finite = [
            float(value)
            for values in panel_series
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
        if not finite:
            continue
        minimum, maximum = min(finite), max(finite)
        if math.isclose(minimum, maximum):
            minimum -= 1.0
            maximum += 1.0
        padding = 0.08 * (maximum - minimum)
        minimum -= padding
        maximum += padding
        for values, color in zip(panel_series, colors):
            previous: tuple[int, int] | None = None
            for index, value in enumerate(values):
                if value is None or not math.isfinite(float(value)):
                    previous = None
                    continue
                point = (
                    int(round(left + index / max(len(values) - 1, 1) * (right - left))),
                    int(round(bottom - (float(value) - minimum) / (maximum - minimum) * panel_height)),
                )
                if previous is not None:
                    draw_line(previous, point, color)
                previous = point

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)

    scanlines = b"".join(
        b"\x00" + bytes(pixels[row * width * 3 : (row + 1) * width * 3]) for row in range(height)
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(scanlines, 9))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return True


def _write_heatmaps(
    path: Path,
    samples: Sequence[ReplaySample],
    frames: Sequence[Any],
    zone_transform: str,
) -> bool:
    count = min(len(samples), len(frames))
    if not count:
        return False
    groups: dict[int, list[int]] = {}
    for index, sample in enumerate(samples[:count]):
        groups.setdefault(int(round(sample.tcp_z_m * 1000.0)), []).append(index)
    stable = [(level, indices) for level, indices in groups.items() if len(indices) >= 4]
    if not stable:
        return False
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return _write_dependency_free_heatmaps(path, samples, frames, zone_transform, stable)
    columns = len(stable)
    figure, axes = plt.subplots(2, columns, figsize=(max(3.0 * columns, 8.0), 6.0), squeeze=False)
    for col, (level, indices) in enumerate(stable):
        real = np.asarray([transform_zones(samples[index].real_zones_mm, zone_transform) for index in indices], dtype=float)
        sim = np.asarray([_flatten_matrix(frames[index].distances_mm) for index in indices], dtype=float)
        real[real <= 0] = np.nan
        sim[sim <= 0] = np.nan
        for row, (values, label) in enumerate(((real, "real"), (sim, "simulation"))):
            valid = np.isfinite(values)
            totals = np.nansum(values, axis=0)
            counts = valid.sum(axis=0)
            average = np.divide(totals, counts, out=np.full(64, np.nan), where=counts > 0).reshape(8, 8)
            axes[row][col].imshow(average, cmap="viridis", vmin=0, vmax=500)
            axes[row][col].set_title(f"{label}: TCP {level} mm")
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
    figure.suptitle("Cup experiment: real vs raw RTX plateau averages")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _write_dependency_free_heatmaps(
    path: Path,
    samples: Sequence[ReplaySample],
    frames: Sequence[Any],
    zone_transform: str,
    stable: Sequence[tuple[int, list[int]]],
) -> bool:
    """Write a compact RGB PNG when Matplotlib is unavailable in Isaac Python."""

    cell = 12
    panel = 8 * cell
    gap = 8
    width = len(stable) * (panel + gap) + gap
    height = 2 * (panel + gap) + gap
    pixels = bytearray([245] * (width * height * 3))

    def color(value: float | None) -> tuple[int, int, int]:
        if value is None:
            return (25, 25, 25)
        fraction = min(max(value / 500.0, 0.0), 1.0)
        return (
            int(35 + 220 * fraction),
            int(40 + 180 * (1.0 - abs(2.0 * fraction - 1.0))),
            int(220 - 185 * fraction),
        )

    for column, (_level, indices) in enumerate(stable):
        real_rows = [transform_zones(samples[index].real_zones_mm, zone_transform) for index in indices]
        sim_rows = [_flatten_matrix(frames[index].distances_mm) for index in indices]
        for panel_row, rows in enumerate((real_rows, sim_rows)):
            averages: list[float | None] = []
            for zone in range(64):
                valid = [float(row[zone]) for row in rows if row[zone] > 0]
                averages.append(sum(valid) / len(valid) if valid else None)
            origin_x = gap + column * (panel + gap)
            origin_y = gap + panel_row * (panel + gap)
            for zone, value in enumerate(averages):
                zone_x = origin_x + (zone % 8) * cell
                zone_y = origin_y + (zone // 8) * cell
                rgb = color(value)
                for y in range(zone_y, zone_y + cell - 1):
                    for x in range(zone_x, zone_x + cell - 1):
                        offset = (y * width + x) * 3
                        pixels[offset : offset + 3] = bytes(rgb)

    def chunk(chunk_type: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)

    scanlines = b"".join(b"\x00" + bytes(pixels[row * width * 3 : (row + 1) * width * 3]) for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(scanlines, 9))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return True


def write_shape_replay_outputs(run: ShapeReplayRun, frames: Sequence[Any]) -> dict[str, Any]:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows, summary = build_comparison(run.samples, frames, run.profile.zone_transform)
    flat_path = run.output_dir / "sim_flat.csv"
    with flat_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["reference_timestamp", "frame_index", "sim_tick", "elapsed_s", "tcp_z_m", "sensor_z_m", "valid_zones"]
            + [f"zone_{index:02d}" for index in range(64)]
            + [f"intensity_{index:02d}" for index in range(64)]
            + [f"material_{index:02d}" for index in range(64)]
        )
        for index, (sample, frame) in enumerate(zip(run.samples, frames)):
            zones = _flatten_matrix(frame.distances_mm)
            intensities = _flatten_auxiliary_matrix(frame.intensities)
            materials = _flatten_auxiliary_matrix(frame.material_ids)
            writer.writerow(
                [
                    sample.reference_timestamp,
                    index,
                    run.sim_ticks[index] if index < len(run.sim_ticks) else "",
                    sample.elapsed_s,
                    sample.tcp_z_m,
                    sample.sensor_z_m,
                    sum(value > 0 for value in zones),
                ]
                + zones
                + intensities
                + materials
            )

    comparison_path = run.output_dir / "comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(comparison_rows[0]) if comparison_rows else ["frame_index"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary.update(
        {
            "experiment": run.profile.name,
            "direction": run.direction,
            "zone_transform": run.profile.zone_transform,
            "mesh_pose": {
                "x_m": run.profile.mesh_pose.x_m,
                "y_m": run.profile.mesh_pose.y_m,
                "yaw_deg": run.profile.mesh_pose.yaw_deg,
            },
            "tcp_to_sensor_z_m": run.profile.tcp_to_sensor_z_m,
            "raw_rtx": True,
        }
    )
    heatmap_written = _write_heatmaps(run.output_dir / "step_heatmaps.png", run.samples, frames, run.profile.zone_transform)
    summary["step_heatmaps_written"] = heatmap_written
    summary["comparison_graph_written"] = _write_comparison_graph(
        run.output_dir / "comparison_graph.png", comparison_rows, summary
    )
    with (run.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def write_calibration(path: str | Path, result: CalibrationResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def replace_cli_option(arguments: Sequence[str], option: str, value: str) -> list[str]:
    """Replace an argparse-style option in a child-process command line."""

    output: list[str] = []
    index = 0
    replaced = False
    while index < len(arguments):
        token = arguments[index]
        if token == option:
            output.extend((option, value))
            replaced = True
            index += 2
            continue
        if token.startswith(option + "="):
            output.append(option + "=" + value)
            replaced = True
            index += 1
            continue
        output.append(token)
        index += 1
    if not replaced:
        output.extend((option, value))
    return output
