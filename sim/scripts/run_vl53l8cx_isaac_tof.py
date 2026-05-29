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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH = REPO_ROOT / "sim" / "config" / "vl53l8cx_8x8.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "sim" / "output" / "vl53l8cx_isaac_tof.csv"


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


def write_frames_csv(frames: Iterable[VL53L8CXFrame], path: str | Path, append: bool = False) -> None:
    with VL53L8CXCsvWriter(path, append=append) as writer:
        for frame in frames:
            writer.write_frame(frame)


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


def _parse_vec(text: str, expected_len: int, name: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected_len} comma-separated values")
    return values


def _optional_path(text: str) -> Path | None:
    return None if text == "" else Path(text)


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
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

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

        scene_updater = _create_isaac_scene(
            args=args,
            stage=stage,
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
        if csv_writer_context is None:
            csv_writer = None
        else:
            csv_writer = csv_writer_context.__enter__()

        try:
            for frame_index in range(args.frames):
                if scene_updater is not None:
                    scene_updater(frame_index)
                simulation_app.update()
                sensor_frame = sensor.get_current_frame()
                if args.debug_draw and ray_draw is not None and _has_distance_payload(sensor_frame):
                    _draw_lidar_rays(sensor_frame, ray_draw, stage, "/VL53L8CX", Gf, Usd, UsdGeom)
                if args.print_frames:
                    _print_frame_summary(sensor_frame)
                if not _has_distance_payload(sensor_frame):
                    continue

                frame = frame_from_rtx_scan(sensor_frame, config)
                frame.distances_mm = np.asarray(frame.distances_mm, dtype=np.int32)
                if frame.intensities is not None:
                    frame.intensities = np.asarray(frame.intensities, dtype=object)
                if frame.material_ids is not None:
                    frame.material_ids = np.asarray(frame.material_ids, dtype=object)
                frames.append(frame)
                if csv_writer is not None:
                    csv_writer.write_frame(frame)
                if args.print_frames:
                    print(format_matrix_for_csv(frame.distances_mm))
        finally:
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
    }

    distance = args.target_distance_m
    center_z = args.target_center_z
    if args.scene == "no-target":
        return None
    if args.scene == "white":
        _add_box(stage, "/World/Targets/white_panel", (distance, 0.0, center_z), (0.02, 0.5, 0.5), materials["white"], Gf, UsdGeom, UsdShade)
        return None
    if args.scene == "oblique":
        prim, _translate_op = _add_box(
            stage,
            "/World/Targets/oblique_panel",
            (distance, 0.0, center_z),
            (0.02, 0.55, 0.55),
            materials["concrete"],
            Gf,
            UsdGeom,
            UsdShade,
        )
        rotate_op = UsdGeom.Xformable(prim).AddRotateZOp()
        rotate_op.Set(30.0)
        return None
    if args.scene == "moving":
        _prim, translate_op = _add_box(
            stage,
            "/World/Targets/moving_panel",
            (distance, 0.0, center_z),
            (0.02, 0.45, 0.45),
            materials["white"],
            Gf,
            UsdGeom,
            UsdShade,
        )

        def update(frame_index: int) -> None:
            phase = frame_index / max(args.frames - 1, 1)
            x = distance + args.motion_amplitude_m * math.sin(2.0 * math.pi * phase)
            translate_op.Set(Gf.Vec3d(x, 0.0, center_z))

        return update

    # Material grid scene: five vertical strips across the sensor FoV.
    strip_materials = [materials["white"], materials["rubber"], materials["glass"], materials["aluminum"], materials["concrete"]]
    strip_width = 0.18
    start_y = -strip_width * (len(strip_materials) - 1) / 2.0
    for index, material in enumerate(strip_materials):
        y = start_y + index * strip_width
        _add_box(
            stage,
            f"/World/Targets/material_strip_{index}",
            (distance, y, center_z),
            (0.02, strip_width * 0.95, 0.55),
            material,
            Gf,
            UsdGeom,
            UsdShade,
        )
    return None


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a VL53L8CX-style RTX ToF prototype in Isaac Sim.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH, help="Path to the VL53L8CX JSON profile.")
    parser.add_argument("--output_csv", type=_optional_path, default=DEFAULT_OUTPUT_CSV, help="CSV output path; pass an empty string to disable.")
    parser.add_argument("--output_npy", type=Path, default=None, help="Optional .npy output path for stacked distance frames.")
    parser.add_argument("--append_csv", action="store_true", help="Append to CSV instead of replacing it.")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to run.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headlessly.")
    parser.add_argument("--renderer", default="RaytracedLighting", help="Isaac renderer name.")
    parser.add_argument("--debug_draw", action="store_true", help="Attach RTX lidar debug-draw point cloud writer.")
    parser.add_argument("--print_frames", action="store_true", help="Print each 8x8 distance matrix.")
    parser.add_argument(
        "--scene",
        choices=("materials", "white", "oblique", "moving", "no-target"),
        default="materials",
        help="Prototype scene to render.",
    )
    parser.add_argument("--target_distance_m", type=float, default=1.0, help="Nominal target distance in meters.")
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
    run_isaac_prototype(args)


if __name__ == "__main__":
    main()
