"""Isaac Sim VL53L8CX-style 8x8 ToF prototype.

This module is split into two parts:

* Dependency-free helpers for config validation, emitter mapping, range
  conversion, and CSV compatibility with the existing TouchIQ visualizer.
* A lazy Isaac Sim runtime path that imports Isaac/Omniverse modules only
  after ``SimulationApp`` has been created.

Run with Isaac Sim's Python, for example:

    python.sh sim/scripts/run_vl53l8cx_isaac_tof.py --headless --frames 120

The current project Python environment does not need Isaac Sim or NumPy to
import this file and run the unit tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH = REPO_ROOT / "sim" / "config" / "vl53l8cx_8x8.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "sim" / "output" / "vl53l8cx_isaac_tof.csv"
DEFAULT_FLAT_CSV = REPO_ROOT / "sim" / "output" / "live_readings.csv"
DEFAULT_TEST_RESULTS_DIR = REPO_ROOT / "sim" / "test_results"
DEFAULT_VISUALIZER_SCRIPT = REPO_ROOT / "sim" / "scripts" / "visualizer.py"
TARGET_CUBE_SIZE_M = (0.03989, 0.04015, 0.03991)
TABLE_CUBE_GAP_M = 0.05
TABLE_SENSOR_HOUSING_SIZE_M = (0.06, 0.06, 0.055)
TABLE_SENSOR_FACE_SIZE_M = (0.002, 0.034, 0.022)


@dataclass(frozen=True)
class VL53L8CXConfig:
    """Configuration for a VL53L8CX-like multizone ToF frame."""

    rows: int = 8
    cols: int = 8
    fov_h_deg: float = 45.0
    fov_v_deg: float = 45.0
    min_mm: int = 20
    max_mm: int = 4000
    frame_rate_hz: float = 15.0
    invalid_mm: int = 0

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if self.fov_h_deg <= 0.0 or self.fov_v_deg <= 0.0:
            raise ValueError("FoV values must be positive")
        if self.min_mm < 0 or self.max_mm <= self.min_mm:
            raise ValueError("range must satisfy 0 <= min_mm < max_mm")
        if self.frame_rate_hz <= 0.0:
            raise ValueError("frame_rate_hz must be positive")
        if self.invalid_mm < 0:
            raise ValueError("invalid_mm must be non-negative")

    @property
    def zones(self) -> int:
        return self.rows * self.cols

    @property
    def min_range_m(self) -> float:
        return self.min_mm / 1000.0

    @property
    def max_range_m(self) -> float:
        return self.max_mm / 1000.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "VL53L8CXConfig":
        known = {field for field in cls.__dataclass_fields__}
        return cls(**{key: data[key] for key in known if key in data})

    @classmethod
    def from_json(cls, path: str | Path) -> "VL53L8CXConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))


@dataclass
class VL53L8CXFrame:
    """One timestamped VL53L8CX-style frame.

    ``distances_mm`` may be a nested Python sequence or a NumPy array. It is
    intentionally typed loosely so this class remains importable without NumPy.
    """

    timestamp: str
    distances_mm: Any
    intensities: Any | None = None
    material_ids: Any | None = None

    def csv_row(self) -> list[str]:
        return [self.timestamp, format_matrix_for_csv(self.distances_mm)]


def iter_matrix_rows(matrix: Any) -> list[list[int]]:
    """Return a plain nested integer list from a list-like or NumPy-like matrix."""

    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    rows: list[list[int]] = []
    for row in matrix:
        if hasattr(row, "tolist"):
            row = row.tolist()
        rows.append([int(value) for value in row])
    return rows


def flatten_matrix(matrix: Any) -> list[int]:
    return [value for row in iter_matrix_rows(matrix) for value in row]


def flatten_optional_matrix(matrix: Any, zones: int) -> list[Any]:
    if matrix is None:
        return ["" for _ in range(zones)]
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()

    values: list[Any] = []
    for row in matrix:
        if hasattr(row, "tolist"):
            row = row.tolist()
        for value in row:
            values.append("" if value is None else value)

    if len(values) < zones:
        values.extend("" for _ in range(zones - len(values)))
    return values[:zones]


def format_matrix_for_csv(matrix: Any) -> str:
    """Format an 8x8 matrix like the existing recorded CSV data.

    Example: ``[[20 23], [24 25]]``. Values inside a row are space-separated,
    while rows are comma-separated, causing ``csv.writer`` to quote the field.
    """

    rows = iter_matrix_rows(matrix)
    return "[" + ", ".join("[" + " ".join(str(value) for value in row) + "]" for row in rows) + "]"


def parse_matrix_text(text: str, rows: int = 8, cols: int = 8) -> list[list[int]]:
    """Parse the project CSV matrix text back into a nested integer list."""

    values = [int(match) for match in re.findall(r"-?\d+", text)]
    expected = rows * cols
    if len(values) != expected:
        raise ValueError(f"expected {expected} matrix values, got {len(values)}")
    return [values[start : start + cols] for start in range(0, expected, cols)]


class VL53L8CXCsvWriter:
    """CSV writer compatible with ``examples/press_example.csv``."""

    def __init__(self, path: str | Path, append: bool = False) -> None:
        self.path = Path(path)
        self.append = append
        self._handle: Any | None = None
        self._writer: csv.writer | None = None

    def __enter__(self) -> "VL53L8CXCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        should_write_header = not self.append or not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a" if self.append else "w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)
        if should_write_header:
            self._writer.writerow(["time_stamp", "data"])
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None

    def write_frame(self, frame: VL53L8CXFrame) -> None:
        if self._writer is None:
            raise RuntimeError("CSV writer is not open")
        self._writer.writerow(frame.csv_row())
        if self._handle is not None:
            self._handle.flush()


def write_frames_csv(frames: Iterable[VL53L8CXFrame], path: str | Path, append: bool = False) -> None:
    with VL53L8CXCsvWriter(path, append=append) as writer:
        for frame in frames:
            writer.write_frame(frame)


class VL53L8CXFlatCsvWriter:
    """Flat 64-zone CSV writer for live visualization and offline analysis."""

    def __init__(self, path: str | Path, config: VL53L8CXConfig, append: bool = False) -> None:
        self.path = Path(path)
        self.config = config
        self.append = append
        self._handle: Any | None = None
        self._writer: csv.writer | None = None

    def __enter__(self) -> "VL53L8CXFlatCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        should_write_header = not self.append or not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a" if self.append else "w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)
        if should_write_header:
            self._writer.writerow(
                ["timestamp", "frame_index", "sim_tick", "valid_zones"]
                + [f"zone_{index:02d}" for index in range(self.config.zones)]
                + [f"intensity_{index:02d}" for index in range(self.config.zones)]
                + [f"material_{index:02d}" for index in range(self.config.zones)]
            )
            self._handle.flush()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None

    def write_frame(self, frame_index: int, sim_tick: int, frame: VL53L8CXFrame) -> None:
        if self._writer is None:
            raise RuntimeError("CSV writer is not open")
        values = flatten_matrix(frame.distances_mm)
        valid_zones = sum(1 for value in values if value > 0)
        intensities = flatten_optional_matrix(frame.intensities, self.config.zones)
        material_ids = flatten_optional_matrix(frame.material_ids, self.config.zones)
        self._writer.writerow([frame.timestamp, frame_index, sim_tick, valid_zones] + values + intensities + material_ids)
        if self._handle is not None:
            self._handle.flush()


def distance_m_to_mm(distance_m: Any, config: VL53L8CXConfig) -> int:
    """Convert a finite hit distance in meters to a clipped integer millimeter value."""

    if distance_m is None:
        return config.invalid_mm
    try:
        distance = float(distance_m)
    except (TypeError, ValueError):
        return config.invalid_mm
    if not math.isfinite(distance) or distance <= 0.0:
        return config.invalid_mm
    distance_mm = int(round(distance * 1000.0))
    return min(max(distance_mm, config.min_mm), config.max_mm)


def zone_index_to_row_col(zone_index: int, config: VL53L8CXConfig) -> tuple[int, int]:
    if zone_index < 0 or zone_index >= config.zones:
        raise ValueError(f"zone index {zone_index} outside 0..{config.zones - 1}")
    return divmod(zone_index, config.cols)


def row_col_to_zone_index(row: int, col: int, config: VL53L8CXConfig) -> int:
    if row < 0 or row >= config.rows or col < 0 or col >= config.cols:
        raise ValueError(f"row/col ({row}, {col}) outside {config.rows}x{config.cols}")
    return row * config.cols + col


def emitter_ids_to_zone_indices(emitter_ids: Sequence[Any], config: VL53L8CXConfig) -> list[int | None]:
    """Normalize Isaac emitter IDs to row-major zone indices.

    Isaac profiles commonly expose emitter IDs as either 0-based or 1-based.
    If every returned ID is in ``1..zones`` and none are zero, the IDs are
    treated as 1-based. Otherwise they are treated as 0-based.
    """

    ids: list[int] = []
    for emitter_id in emitter_ids:
        try:
            ids.append(int(emitter_id))
        except (TypeError, ValueError):
            ids.append(-1)

    one_based = bool(ids) and all(1 <= emitter_id <= config.zones for emitter_id in ids)
    indices: list[int | None] = []
    for emitter_id in ids:
        index = emitter_id - 1 if one_based else emitter_id
        indices.append(index if 0 <= index < config.zones else None)
    return indices


def _empty_matrix(rows: int, cols: int, fill: Any = 0) -> list[list[Any]]:
    return [[fill for _ in range(cols)] for _ in range(rows)]


def build_distance_matrix_from_returns(
    distances_m: Sequence[Any],
    config: VL53L8CXConfig,
    emitter_ids: Sequence[Any] | None = None,
    intensities: Sequence[Any] | None = None,
    material_ids: Sequence[Any] | None = None,
) -> tuple[list[list[int]], list[list[float | None]] | None, list[list[int | None]] | None]:
    """Build a row-major VL53L8CX frame from RTX Lidar returns.

    If more than one return lands in the same zone, the closest valid return is
    kept. Missing zones remain ``config.invalid_mm``.
    """

    distances = list(distances_m)
    if emitter_ids is None:
        zone_indices: list[int | None] = list(range(min(len(distances), config.zones)))
    else:
        zone_indices = emitter_ids_to_zone_indices(list(emitter_ids), config)

    intensity_values = list(intensities) if intensities is not None else None
    material_values = list(material_ids) if material_ids is not None else None
    distance_matrix = _empty_matrix(config.rows, config.cols, config.invalid_mm)
    intensity_matrix = _empty_matrix(config.rows, config.cols, None) if intensity_values is not None else None
    material_matrix = _empty_matrix(config.rows, config.cols, None) if material_values is not None else None

    for return_index, zone_index in enumerate(zone_indices):
        if zone_index is None or return_index >= len(distances):
            continue
        distance_mm = distance_m_to_mm(distances[return_index], config)
        if distance_mm == config.invalid_mm:
            continue
        row, col = zone_index_to_row_col(zone_index, config)
        previous = distance_matrix[row][col]
        if previous == config.invalid_mm or distance_mm < previous:
            distance_matrix[row][col] = distance_mm
            if intensity_matrix is not None and intensity_values is not None and return_index < len(intensity_values):
                intensity_matrix[row][col] = float(intensity_values[return_index])
            if material_matrix is not None and material_values is not None and return_index < len(material_values):
                material_matrix[row][col] = int(material_values[return_index])

    return distance_matrix, intensity_matrix, material_matrix


def _array_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _extract_scan_payload(sensor_frame: dict[str, Any]) -> dict[str, Any]:
    """Find the RTX scan-buffer payload in a LidarRtx frame dictionary."""

    for key in ("IsaacCreateRTXLidarScanBuffer", "isaacCreateRTXLidarScanBuffer"):
        payload = sensor_frame.get(key)
        if isinstance(payload, dict):
            return payload
    return sensor_frame


def _has_distance_payload(sensor_frame: Any) -> bool:
    if not isinstance(sensor_frame, dict) or not sensor_frame:
        return False
    payload = _extract_scan_payload(sensor_frame)
    if "distance" not in payload:
        return False
    distances = payload.get("distance")
    if distances is None:
        return False
    if hasattr(distances, "size"):
        return int(distances.size) > 0
    try:
        return len(distances) > 0
    except TypeError:
        return True


def _print_frame_summary(sensor_frame: Any) -> None:
    if not isinstance(sensor_frame, dict):
        print(f"RTX frame: {type(sensor_frame)} {sensor_frame}")
        return

    print(f"RTX frame keys: {list(sensor_frame.keys())}")
    payload = _extract_scan_payload(sensor_frame)
    for key, value in payload.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {type(value)} {value}")


def _format_matrix_for_console(matrix: Any) -> str:
    rows = iter_matrix_rows(matrix)
    return "\n".join(
        ("[[" if index == 0 else " [")
        + " ".join(f"{value:4d}" for value in row)
        + ("]]" if index == len(rows) - 1 else "]")
        for index, row in enumerate(rows)
    )


def _print_distance_matrix(frame_index: int, sim_tick: int, matrix: Any) -> None:
    values = flatten_matrix(matrix)
    valid = [value for value in values if value > 0]
    valid_zones = len(valid)
    mean_mm = sum(valid) / valid_zones if valid_zones else 0.0
    min_mm = min(valid) if valid else 0
    max_mm = max(valid) if valid else 0

    print(
        f"Frame {frame_index:05d} | sim_tick={sim_tick:05d} | "
        f"valid_zones={valid_zones} | mean={mean_mm:.1f} mm | min={min_mm} | max={max_mm}"
    )
    print()
    print(_format_matrix_for_console(matrix))
    print()


def _draw_lidar_rays(
    sensor_frame: dict[str, Any],
    draw: Any,
    stage: Any,
    sensor_prim_path: str,
    Gf: Any,
    Usd: Any,
    UsdGeom: Any,
    color: tuple[float, float, float, float] = (0.1, 0.8, 1.0, 1.0),
    line_width: float = 2.0,
) -> None:
    payload = _extract_scan_payload(sensor_frame)

    points = payload.get("data")
    if points is None or not hasattr(points, "shape") or points.shape[0] == 0:
        return

    prim = stage.GetPrimAtPath(sensor_prim_path)
    world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    origin = world_transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    origin_tuple = (float(origin[0]), float(origin[1]), float(origin[2]))

    start_points = []
    end_points = []
    for point in points:
        local_hit = Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
        world_hit = world_transform.Transform(local_hit)
        start_points.append(origin_tuple)
        end_points.append((float(world_hit[0]), float(world_hit[1]), float(world_hit[2])))

    colors = [color for _ in start_points]
    widths = [line_width for _ in start_points]

    draw.clear_lines()
    draw.draw_lines(start_points, end_points, colors, widths)


def frame_from_rtx_scan(sensor_frame: dict[str, Any], config: VL53L8CXConfig) -> VL53L8CXFrame:
    """Convert a LidarRtx ``get_current_frame()`` dictionary into a ToF frame."""

    payload = _extract_scan_payload(sensor_frame)
    distances = _array_to_list(payload.get("distance"))
    emitter_ids = _array_to_list(payload.get("emitterId")) or None
    intensities = _array_to_list(payload.get("intensity")) or None
    material_ids = _array_to_list(payload.get("materialId")) or None
    matrix, intensity_matrix, material_matrix = build_distance_matrix_from_returns(
        distances,
        config,
        emitter_ids=emitter_ids,
        intensities=intensities,
        material_ids=material_ids,
    )
    return VL53L8CXFrame(
        timestamp=datetime.now().time().isoformat(timespec="microseconds"),
        distances_mm=matrix,
        intensities=intensity_matrix,
        material_ids=material_matrix,
    )


def empty_tof_frame(config: VL53L8CXConfig) -> VL53L8CXFrame:
    """Create an accepted no-return frame with all zones marked invalid."""

    return VL53L8CXFrame(
        timestamp=datetime.now().time().isoformat(timespec="microseconds"),
        distances_mm=_empty_matrix(config.rows, config.cols, config.invalid_mm),
        intensities=None,
        material_ids=None,
    )


def frame_from_sensor_frame(
    sensor_frame: Any,
    config: VL53L8CXConfig,
    *,
    allow_empty_no_return: bool = False,
) -> VL53L8CXFrame | None:
    """Convert an RTX frame, optionally accepting no-return frames as all-zero output."""

    if _has_distance_payload(sensor_frame):
        return frame_from_rtx_scan(sensor_frame, config)
    if allow_empty_no_return:
        return empty_tof_frame(config)
    return None


def _parse_vec(text: str, expected_len: int, name: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected_len} comma-separated values")
    return values


def _optional_path(text: str) -> Path | None:
    return None if text == "" else Path(text)


def _make_test_results_path() -> Path:
    name = datetime.now().strftime("readings_%Y-%m-%d_%H-%M-%S.csv")
    return DEFAULT_TEST_RESULTS_DIR / name


def _launch_visualizer(args: argparse.Namespace, flat_csv_path: Path) -> subprocess.Popen[Any]:
    command = [
        str(args.visualizer_python),
        str(args.visualizer_script),
        "--source",
        "csv-tail",
        "--input_csv",
        str(flat_csv_path),
    ]
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    try:
        return subprocess.Popen(command, creationflags=creationflags)
    except OSError as exc:
        raise RuntimeError(f"failed to launch visualizer: {exc}") from exc


def _terminate_visualizer(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _make_sensor_attributes(
    config: VL53L8CXConfig,
    aux_output_type: str = "EXTRA",
    include_compat_attrs: bool = True,
) -> dict[str, Any]:
    half_h = config.fov_h_deg / 2.0
    half_v = config.fov_v_deg / 2.0

    azimuths: list[float] = []
    elevations: list[float] = []
    channel_ids: list[int] = []
    fire_times: list[int] = []

    for row in range(config.rows):
        elevation = half_v - (config.fov_v_deg * row / max(config.rows - 1, 1))
        for col in range(config.cols):
            azimuth = -half_h + (config.fov_h_deg * col / max(config.cols - 1, 1))
            azimuths.append(float(azimuth))
            elevations.append(float(elevation))
            channel_ids.append(row * config.cols + col + 1)
            fire_times.append(0)

    attrs = {
        "omni:sensor:Core:scanRateBaseHz": config.frame_rate_hz,
        "omni:sensor:Core:reportRateBaseHz": config.frame_rate_hz,
        "omni:sensor:Core:numberOfEmitters": config.zones,
        "omni:sensor:Core:numberOfChannels": config.zones,
        "omni:sensor:Core:nearRangeM": config.min_range_m,
        "omni:sensor:Core:farRangeM": config.max_range_m,
        "omni:sensor:Core:minDistBetweenEchosM": config.min_range_m,
        "omni:sensor:Core:maxReturns": 1,
        "omni:sensor:Core:auxOutputType": aux_output_type,
        "omni:sensor:Core:emitterState:s001:azimuthDeg": azimuths,
        "omni:sensor:Core:emitterState:s001:elevationDeg": elevations,
        "omni:sensor:Core:emitterState:s001:channelId": channel_ids,
        "omni:sensor:Core:emitterState:s001:fireTimeNs": fire_times,
    }
    if include_compat_attrs:
        attrs.update(
            {
                "OmniSensorGenericLidarCoreEmitterStateAPI:s001:beamCountHoriz": config.cols,
                "OmniSensorGenericLidarCoreEmitterStateAPI:s001:beamCountVert": config.rows,
                "OmniSensorGenericLidarCoreEmitterStateAPI:s001:azimuthStartDeg": -half_h,
                "OmniSensorGenericLidarCoreEmitterStateAPI:s001:azimuthEndDeg": half_h,
                "OmniSensorGenericLidarCoreEmitterStateAPI:s001:elevationStartDeg": half_v,
                "OmniSensorGenericLidarCoreEmitterStateAPI:s001:elevationEndDeg": -half_v,
                "OmniSensorGenericLidarCoreEmitterStateAPI:s001:minRangeM": config.min_range_m,
                "OmniSensorGenericLidarCoreEmitterStateAPI:s001:maxRangeM": config.max_range_m,
            }
        )
    return attrs


def run_isaac_prototype(args: argparse.Namespace) -> list[VL53L8CXFrame]:
    """Run the Isaac Sim RTX sensor prototype.

    Isaac/Omniverse modules are imported inside this function only. Keep the
    ``SimulationApp`` import and construction before all other Isaac imports.
    """

    # Isaac Sim must be booted before importing omni/isaac extension modules.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
        }
    )

    try:
        import carb
        import numpy as np
        import omni
        import omni.kit.commands
        import omni.timeline
        import omni.usd
        from isaacsim.sensors.rtx import LidarRtx
        from isaacsim.util.debug_draw import _debug_draw
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        carb.settings.get_settings().set_bool("/app/sensors/nv/lidar/outputBufferOnGPU", True)
        carb.settings.get_settings().set_bool("/rtx-transient/stableIds/enabled", True)

        config = VL53L8CXConfig.from_json(args.profile)
        sensor_translation = np.array(_parse_vec(args.sensor_xyz, 3, "sensor_xyz"), dtype=float)
        sensor_orientation = np.array(_parse_vec(args.sensor_quat_wxyz, 4, "sensor_quat_wxyz"), dtype=float)

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        _create_sandbox_world(
            stage=stage,
            Gf=Gf,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
            UsdLux=UsdLux,
        )

        scene_updater = _create_isaac_scene(
            args=args,
            config=config,
            stage=stage,
            Gf=Gf,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
        )
        if args.scene != "table-cube":
            _add_sensor_visual_marker(
                stage=stage,
                sensor_translation=sensor_translation,
                Gf=Gf,
                Sdf=Sdf,
                UsdGeom=UsdGeom,
                UsdShade=UsdShade,
            )

        sensor = LidarRtx(
            prim_path="/VL53L8CX",
            translation=sensor_translation,
            orientation=sensor_orientation,
            config_file_name=None,
            **_make_sensor_attributes(config, include_compat_attrs=False),
        )
        sensor.initialize()
        try:
            sensor.attach_annotator(
                "IsaacCreateRTXLidarScanBuffer",
                outputDistance=True,
                outputIntensity=True,
                outputEmitterId=True,
                outputMaterialId=True,
            )
        except TypeError:
            sensor.attach_annotator("IsaacCreateRTXLidarScanBuffer")
        ray_draw = None
        if args.debug_draw:
            sensor.attach_writer("RtxLidarDebugDrawPointCloud")
            ray_draw = _debug_draw.acquire_debug_draw_interface()

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(30):
            simulation_app.update()

        frames: list[VL53L8CXFrame] = []
        csv_writer_context = VL53L8CXCsvWriter(args.output_csv, append=args.append_csv) if args.output_csv else None
        csv_writer = None
        flat_csv_path = None if args.disable_flat_csv else Path(args.flat_csv)
        flat_writer_contexts: list[VL53L8CXFlatCsvWriter] = []
        if flat_csv_path is not None:
            flat_writer_contexts.append(VL53L8CXFlatCsvWriter(flat_csv_path, config))
        if args.record_test_results:
            flat_writer_contexts.append(VL53L8CXFlatCsvWriter(_make_test_results_path(), config))
        flat_writers: list[VL53L8CXFlatCsvWriter] = []
        visualizer_process: subprocess.Popen[Any] | None = None

        try:
            if csv_writer_context is not None:
                csv_writer = csv_writer_context.__enter__()
            for flat_writer_context in flat_writer_contexts:
                flat_writers.append(flat_writer_context.__enter__())
            if args.launch_visualizer:
                if flat_csv_path is None:
                    raise RuntimeError("--launch_visualizer requires flat CSV output; remove --disable_flat_csv")
                visualizer_process = _launch_visualizer(args, flat_csv_path)

            valid_count = 0
            sim_tick = 0
            max_sim_ticks = args.max_sim_ticks or max(args.frames * 20, args.frames + 300)
            while valid_count < args.frames:
                if sim_tick >= max_sim_ticks:
                    raise RuntimeError(
                        f"captured {valid_count}/{args.frames} valid ToF frames after "
                        f"{sim_tick} simulation ticks; increase --max_sim_ticks if the "
                        "scene is intentionally slow to produce RTX frames"
                    )
                if scene_updater is not None:
                    scene_updater(sim_tick)
                simulation_app.update()
                sensor_frame = sensor.get_current_frame()
                if args.debug_draw and ray_draw is not None and _has_distance_payload(sensor_frame):
                    _draw_lidar_rays(sensor_frame, ray_draw, stage, "/VL53L8CX", Gf, Usd, UsdGeom)
                if args.print_payload_debug:
                    _print_frame_summary(sensor_frame)
                frame = frame_from_sensor_frame(
                    sensor_frame,
                    config,
                    allow_empty_no_return=args.scene == "no-target",
                )
                if frame is None:
                    sim_tick += 1
                    continue

                frame_index = valid_count
                frame.distances_mm = np.asarray(frame.distances_mm, dtype=np.int32)
                if frame.intensities is not None:
                    frame.intensities = np.asarray(frame.intensities, dtype=object)
                if frame.material_ids is not None:
                    frame.material_ids = np.asarray(frame.material_ids, dtype=object)
                frames.append(frame)
                if csv_writer is not None:
                    csv_writer.write_frame(frame)
                for flat_writer in flat_writers:
                    flat_writer.write_frame(frame_index, sim_tick, frame)
                if args.print_arrays:
                    _print_distance_matrix(frame_index, sim_tick, frame.distances_mm)
                valid_count += 1
                sim_tick += 1
        finally:
            if visualizer_process is not None and not args.keep_visualizer_open:
                _terminate_visualizer(visualizer_process)
            for flat_writer_context in reversed(flat_writer_contexts):
                flat_writer_context.__exit__(None, None, None)
            if csv_writer_context is not None:
                csv_writer_context.__exit__(None, None, None)
            timeline.stop()

        if args.output_npy:
            if not frames:
                raise RuntimeError("no valid RTX lidar distance frames were captured")
            np.save(args.output_npy, np.stack([frame.distances_mm for frame in frames], axis=0))

        return frames
    finally:
        simulation_app.close()


def _create_isaac_scene(
    *,
    args: argparse.Namespace,
    config: VL53L8CXConfig,
    stage: Any,
    Gf: Any,
    Sdf: Any,
    UsdGeom: Any,
    UsdShade: Any,
) -> Any | None:
    """Create simple target scenes visible to a +X-facing solid-state lidar."""

    materials = {
        "white": _create_material(stage, "/World/Looks/WhiteDiffuse", (0.9, 0.9, 0.86), "Default", Gf, Sdf, UsdShade),
        "rubber": _create_material(stage, "/World/Looks/Rubber", (0.02, 0.02, 0.02), "RubberStandard", Gf, Sdf, UsdShade),
        "glass": _create_material(stage, "/World/Looks/PlexiGlass", (0.5, 0.8, 1.0), "PlexiGlassStandard", Gf, Sdf, UsdShade),
        "aluminum": _create_material(stage, "/World/Looks/Aluminum", (0.8, 0.82, 0.84), "MetalAluminum", Gf, Sdf, UsdShade),
        "concrete": _create_material(stage, "/World/Looks/Concrete", (0.5, 0.5, 0.45), "ConcreteRough", Gf, Sdf, UsdShade),
        "blue": _create_material(stage, "/World/Looks/TableCubeBlue", (0.05, 0.55, 1.0), "Default", Gf, Sdf, UsdShade),
        "red": _create_material(stage, "/World/Looks/SensorHousingRed", (0.72, 0.04, 0.05), "Default", Gf, Sdf, UsdShade),
        "black": _create_material(stage, "/World/Looks/SensorFaceBlack", (0.005, 0.005, 0.006), "RubberStandard", Gf, Sdf, UsdShade),
        "wood": _create_material(stage, "/World/Looks/TableWood", (0.74, 0.58, 0.38), "Default", Gf, Sdf, UsdShade),
        "wall": _create_material(stage, "/World/Looks/FabricWallGray", (0.55, 0.58, 0.58), "ConcreteRough", Gf, Sdf, UsdShade),
        "cable_yellow": _create_material(stage, "/World/Looks/CableYellow", (0.85, 0.66, 0.04), "Default", Gf, Sdf, UsdShade),
        "cable_purple": _create_material(stage, "/World/Looks/CablePurple", (0.28, 0.08, 0.38), "Default", Gf, Sdf, UsdShade),
    }

    distance = _scene_target_distance_m(args)
    center_z = args.target_center_z
    if args.scene == "no-target":
        return None
    if args.scene == "table-cube":
        _create_table_cube_scene(
            args=args,
            target_gap_m=distance,
            materials=materials,
            stage=stage,
            Gf=Gf,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
        )
        return None
    if args.scene == "cube":
        size = TARGET_CUBE_SIZE_M
        _add_box(
            stage,
            "/World/Targets/calibration_cube",
            (_center_x_from_front_distance(distance, size[0]), 0.0, center_z),
            size,
            materials["white"],
            Gf,
            UsdGeom,
            UsdShade,
        )
        return None
    if args.scene == "white":
        size = (0.02, 0.5, 0.5)
        _add_box(
            stage,
            "/World/Targets/white_panel",
            (_center_x_from_front_distance(distance, size[0]), 0.0, center_z),
            size,
            materials["white"],
            Gf,
            UsdGeom,
            UsdShade,
        )
        return None
    if args.scene == "white-full":
        y_span = _fov_span_at_distance(distance, config.fov_h_deg, margin=1.25)
        z_span = _fov_span_at_distance(distance, config.fov_v_deg, margin=1.25)
        size = (0.02, y_span, z_span)
        _add_box(
            stage,
            "/World/Targets/white_full_panel",
            (_center_x_from_front_distance(distance, size[0]), 0.0, center_z),
            size,
            materials["white"],
            Gf,
            UsdGeom,
            UsdShade,
        )
        return None
    if args.scene == "oblique":
        size = (0.02, 0.55, 0.55)
        prim, _translate_op = _add_box(
            stage,
            "/World/Targets/oblique_panel",
            (_center_x_from_front_distance(distance, size[0]), 0.0, center_z),
            size,
            materials["concrete"],
            Gf,
            UsdGeom,
            UsdShade,
        )
        rotate_op = UsdGeom.Xformable(prim).AddRotateZOp()
        rotate_op.Set(30.0)
        return None
    if args.scene == "moving":
        size = (0.02, 0.45, 0.45)
        _prim, translate_op = _add_box(
            stage,
            "/World/Targets/moving_panel",
            (_center_x_from_front_distance(distance, size[0]), 0.0, center_z),
            size,
            materials["white"],
            Gf,
            UsdGeom,
            UsdShade,
        )

        def update(frame_index: int) -> None:
            phase = frame_index / max(args.frames - 1, 1)
            front_x = distance + args.motion_amplitude_m * math.sin(2.0 * math.pi * phase)
            x = _center_x_from_front_distance(front_x, size[0])
            translate_op.Set(Gf.Vec3d(x, 0.0, center_z))

        return update

    # Material grid scene: five vertical strips across the sensor FoV.
    strip_materials = [materials["white"], materials["rubber"], materials["glass"], materials["aluminum"], materials["concrete"]]
    strip_width = 0.18
    strip_size = (0.02, strip_width * 0.95, 0.55)
    start_y = -strip_width * (len(strip_materials) - 1) / 2.0
    for index, material in enumerate(strip_materials):
        y = start_y + index * strip_width
        _add_box(
            stage,
            f"/World/Targets/material_strip_{index}",
            (_center_x_from_front_distance(distance, strip_size[0]), y, center_z),
            strip_size,
            material,
            Gf,
            UsdGeom,
            UsdShade,
        )
    return None


def _scene_target_distance_m(args: argparse.Namespace) -> float:
    if args.target_distance_m is not None:
        return args.target_distance_m
    return TABLE_CUBE_GAP_M if args.scene == "table-cube" else 1.0


def _create_table_cube_scene(
    *,
    args: argparse.Namespace,
    target_gap_m: float,
    materials: dict[str, Any],
    stage: Any,
    Gf: Any,
    UsdGeom: Any,
    UsdShade: Any,
) -> None:
    sensor_x, sensor_y, sensor_z = _parse_vec(args.sensor_xyz, 3, "sensor_xyz")
    cube_size = TARGET_CUBE_SIZE_M
    table_top_z = sensor_z - cube_size[2] / 2.0
    table_thickness = 0.03

    _add_box(
        stage,
        "/World/Table/Tabletop",
        center=(sensor_x + 0.18, sensor_y, table_top_z - table_thickness / 2.0),
        size=(0.75, 0.55, table_thickness),
        material=materials["wood"],
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
    )
    _add_box(
        stage,
        "/World/Table/BackWall",
        center=(sensor_x + 0.34, sensor_y, table_top_z + 0.28),
        size=(0.02, 0.75, 0.56),
        material=materials["wall"],
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
    )

    cube_center_x = sensor_x + _center_x_from_front_distance(target_gap_m, cube_size[0])
    _add_box(
        stage,
        "/World/Targets/table_blue_cube",
        center=(cube_center_x, sensor_y, sensor_z),
        size=cube_size,
        material=materials["blue"],
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
    )

    housing_size = TABLE_SENSOR_HOUSING_SIZE_M
    housing_center_z = table_top_z + housing_size[2] / 2.0
    _add_box(
        stage,
        "/World/Sensors/table_sensor_housing",
        center=(sensor_x - housing_size[0] / 2.0 - 0.003, sensor_y, housing_center_z),
        size=housing_size,
        material=materials["red"],
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
    )

    face_size = TABLE_SENSOR_FACE_SIZE_M
    face_center_x = sensor_x - face_size[0] / 2.0 - 0.001
    _add_box(
        stage,
        "/World/Sensors/table_sensor_face",
        center=(face_center_x, sensor_y, sensor_z),
        size=face_size,
        material=materials["black"],
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
    )

    cable_z = table_top_z + 0.003
    for index, (offset_y, material_name) in enumerate(((-0.006, "cable_yellow"), (0.0, "black"), (0.006, "cable_purple"))):
        _add_box(
            stage,
            f"/World/Sensors/table_sensor_cable_{index}",
            center=(sensor_x - 0.105, sensor_y + offset_y, cable_z),
            size=(0.14, 0.003, 0.003),
            material=materials[material_name],
            Gf=Gf,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
        )


def _center_x_from_front_distance(front_distance_m: float, size_x_m: float) -> float:
    return front_distance_m + size_x_m / 2.0


def _fov_span_at_distance(distance_m: float, fov_deg: float, margin: float = 1.0) -> float:
    half_angle_rad = math.radians(fov_deg / 2.0)
    return max(0.1, 2.0 * abs(distance_m) * math.tan(half_angle_rad) * margin)


def _create_sandbox_world(
    *,
    stage: Any,
    Gf: Any,
    Sdf: Any,
    UsdGeom: Any,
    UsdShade: Any,
    UsdLux: Any,
) -> None:
    """Create a minimal gray sandbox world with sunlight."""

    gray_floor_material = _create_material(
        stage,
        "/World/Looks/SandboxGray",
        (0.45, 0.45, 0.45),
        "Default",
        Gf,
        Sdf,
        UsdShade,
    )
    _add_box(
        stage,
        "/World/Sandbox/Floor",
        center=(1.0, 0.0, -0.01),
        size=(5.0, 5.0, 0.02),
        material=gray_floor_material,
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
    )

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.88))
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, -35.0))

    sky = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
    sky.CreateIntensityAttr(120.0)
    sky.CreateColorAttr(Gf.Vec3f(0.55, 0.58, 0.62))


def _create_material(
    stage: Any,
    path: str,
    color: tuple[float, float, float],
    nonvisual_type: str,
    Gf: Any,
    Sdf: Any,
    UsdShade: Any,
) -> Any:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _apply_nonvisual_material(stage, path, nonvisual_type, Sdf)
    return material


def _apply_nonvisual_material(stage: Any, material_path: str, material_type: str, Sdf: Any) -> None:
    try:
        from isaacsim.sensors.rtx import set_non_visual_material_attributes

        set_non_visual_material_attributes(
            material_prim_path=material_path,
            attributes={"sensor:material:type": material_type},
        )
        return
    except Exception:
        pass

    material_prim = stage.GetPrimAtPath(material_path)
    if material_prim:
        for attr_name in (
            "omni:simready:nonvisual:sensor:material:type",
            "omni:simready:nonvisual:attributes:sensor:material:type",
        ):
            material_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.String).Set(material_type)


def _add_box(
    stage: Any,
    path: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material: Any,
    Gf: Any,
    UsdGeom: Any,
    UsdShade: Any,
) -> tuple[Any, Any]:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    translate_op = xform.AddTranslateOp()
    scale_op = xform.AddScaleOp()
    translate_op.Set(Gf.Vec3d(*center))
    scale_op.Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
    return cube.GetPrim(), translate_op


def _add_sensor_visual_marker(
    stage: Any,
    sensor_translation: Any,
    Gf: Any,
    Sdf: Any,
    UsdGeom: Any,
    UsdShade: Any,
) -> None:
    """Add a small flat square to visualize the ToF/Lidar sensor body."""

    sensor_material = _create_material(
        stage,
        "/World/Looks/SensorMarkerBlue",
        (0.05, 0.35, 1.0),
        "Default",
        Gf,
        Sdf,
        UsdShade,
    )

    x = float(sensor_translation[0])
    y = float(sensor_translation[1])
    z = float(sensor_translation[2])
    _add_box(
        stage,
        "/World/Sensors/VL53L8CX_VisualMarker",
        center=(x - 0.015, y, z),
        size=(0.01, 0.12, 0.12),
        material=sensor_material,
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a VL53L8CX-style RTX ToF prototype in Isaac Sim.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH, help="Path to the VL53L8CX JSON profile.")
    parser.add_argument("--output_csv", type=_optional_path, default=DEFAULT_OUTPUT_CSV, help="CSV output path; pass an empty string to disable.")
    parser.add_argument("--output_npy", type=Path, default=None, help="Optional .npy output path for stacked distance frames.")
    parser.add_argument("--append_csv", action="store_true", help="Append to CSV instead of replacing it.")
    parser.add_argument("--frames", type=int, default=120, help="Number of valid ToF frames to capture.")
    parser.add_argument(
        "--max_sim_ticks",
        type=int,
        default=0,
        help="Maximum simulation ticks to search for valid frames; 0 uses an automatic limit.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headlessly.")
    parser.add_argument("--renderer", default="RaytracedLighting", help="Isaac renderer name.")
    parser.set_defaults(debug_draw=True)
    parser.add_argument("--debug_draw", dest="debug_draw", action="store_true", help="Enable RTX lidar debug drawing.")
    parser.add_argument("--no_debug_draw", dest="debug_draw", action="store_false", help="Disable RTX lidar debug drawing.")
    parser.set_defaults(print_arrays=True)
    parser.add_argument("--print_arrays", dest="print_arrays", action="store_true", help="Print each valid 8x8 distance matrix.")
    parser.add_argument("--quiet_arrays", dest="print_arrays", action="store_false", help="Disable console matrix output.")
    parser.add_argument("--print_payload_debug", action="store_true", help="Print verbose RTX payload keys and array shapes.")
    parser.add_argument("--print_frames", dest="print_arrays", action="store_true", help="Deprecated alias for --print_arrays.")
    parser.add_argument("--flat_csv", type=Path, default=DEFAULT_FLAT_CSV, help="Flat live CSV path with zone_00 through zone_63 columns.")
    parser.add_argument("--disable_flat_csv", action="store_true", help="Disable flat live CSV output.")
    parser.add_argument(
        "--record_test_results",
        action="store_true",
        help="Also write sim/test_results/readings_YYYY-MM-DD_HH-MM-SS.csv.",
    )
    parser.add_argument("--launch_visualizer", action="store_true", help="Launch the live CSV visualizer alongside the simulator.")
    parser.add_argument("--visualizer_script", type=Path, default=DEFAULT_VISUALIZER_SCRIPT, help="Path to the visualizer script.")
    parser.add_argument("--visualizer_python", default="python", help="Python executable used to launch the visualizer.")
    parser.add_argument("--keep_visualizer_open", action="store_true", help="Do not terminate the visualizer when the simulator exits.")
    parser.add_argument(
        "--scene",
        choices=("cube", "table-cube", "materials", "white", "white-full", "oblique", "moving", "no-target"),
        default="cube",
        help="Prototype scene to render.",
    )
    parser.add_argument(
        "--target_distance_m",
        "--target_distance",
        dest="target_distance_m",
        type=float,
        default=None,
        help="Nominal target front-surface distance in meters; defaults to 50 mm for table-cube and 1 m otherwise.",
    )
    parser.add_argument("--target_center_z", type=float, default=0.35, help="Target center height in meters.")
    parser.add_argument("--motion_amplitude_m", type=float, default=0.25, help="Moving-scene sinusoid amplitude.")
    parser.add_argument("--sensor_xyz", default="0,0,0.35", help="Sensor translation as x,y,z meters.")
    parser.add_argument("--sensor_quat_wxyz", default="1,0,0,0", help="Sensor orientation quaternion as w,x,y,z.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.max_sim_ticks < 0:
        raise SystemExit("--max_sim_ticks must be non-negative")
    run_isaac_prototype(args)


if __name__ == "__main__":
    main()
