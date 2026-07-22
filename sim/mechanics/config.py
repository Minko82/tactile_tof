"""Strict, typed JSON loading and defaults for the mechanics runner."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .schema import MECHANICS_OUTPUT_SCHEMA_VERSION


SUPPORTED_CONFIG_SCHEMA_VERSION = 1
SUPPORTED_DAMPING_SEMANTICS = {"stiffness_relative_dimensionless"}

VIDEO_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "path": "mechanics.mp4",
    "fps": 30,
    "codec": "libx264",
    "quality": 7,
    "include_ui": False,
}

EQUILIBRATION_DEFAULTS: dict[str, Any] = {
    "minimum_duration_s": 0.5,
    "maximum_duration_s": 3.0,
    "velocity_tolerance_m_s": 1.0e-5,
    "stable_frames": 10,
    "timeout_behavior": "warn",
}


class ConfigError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not read JSON config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"JSON config {path} must contain an object")
    return value


def _require(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{context} is missing required fields: {', '.join(missing)}")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be an object")
    return value


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a nonempty string")
    return value


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{context} must be a boolean")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ConfigError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{context} must be at least {minimum}")
    return value


def _finite_scalar(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{context} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{context} must be finite")
    return result


def _finite_vector(
    value: Any, length: int, context: str, *, nonzero: bool = False
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ConfigError(f"{context} must contain exactly {length} values")
    result = [_finite_scalar(item, context) for item in value]
    if nonzero and math.sqrt(sum(item * item for item in result)) <= 0.0:
        raise ConfigError(f"{context} must be nonzero")
    return result


def _positive(value: Any, context: str, *, allow_zero: bool = False) -> float:
    result = _finite_scalar(value, context)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ConfigError(f"{context} must be {qualifier}")
    return result


def _schema_version(mapping: dict[str, Any], context: str) -> int:
    _require(mapping, ("schema_version",), context)
    version = _integer(
        mapping["schema_version"], f"{context} schema_version", minimum=1
    )
    if version != SUPPORTED_CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"{context} schema_version {version} is unsupported; "
            f"expected {SUPPORTED_CONFIG_SCHEMA_VERSION}"
        )
    return version


def _resolve_path(base: Path, value: Any, context: str) -> str:
    candidate = Path(_nonempty_string(value, context))
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve())


def load_run_config(path: str | Path) -> dict[str, Any]:
    experiment_path = Path(path).resolve()
    experiment = _read_json(experiment_path)
    _schema_version(experiment, "experiment config")
    _require(
        experiment,
        (
            "asset_config",
            "material_config",
            "fingertip_transform",
            "contact",
            "indenter",
            "trajectory",
            "solver",
            "contact_parameters",
            "monitoring",
            "output",
        ),
        "experiment config",
    )
    for field in (
        "fingertip_transform",
        "contact",
        "indenter",
        "trajectory",
        "solver",
        "contact_parameters",
        "monitoring",
        "output",
    ):
        _mapping(experiment[field], f"experiment {field}")
    raw_video = copy.deepcopy(_mapping(experiment.get("video", {}), "experiment video"))
    raw_equilibration = copy.deepcopy(
        _mapping(experiment.get("equilibration", {}), "experiment equilibration")
    )

    asset_path = Path(
        _resolve_path(
            experiment_path.parent,
            experiment["asset_config"],
            "experiment asset_config",
        )
    )
    material_path = Path(
        _resolve_path(
            experiment_path.parent,
            experiment["material_config"],
            "experiment material_config",
        )
    )
    asset = _read_json(asset_path)
    material = _read_json(material_path)
    _schema_version(asset, "asset config")
    _schema_version(material, "material config")
    _require(
        asset,
        ("surface_stl", "volume_msh", "regions_npz", "surface_mapping_npz"),
        "asset config",
    )
    _require(
        material,
        ("youngs_modulus_pa", "poisson_ratio", "density_kg_m3", "damping"),
        "material config",
    )

    asset = copy.deepcopy(asset)
    for key in ("surface_stl", "volume_msh", "regions_npz", "surface_mapping_npz"):
        asset[key] = _resolve_path(asset_path.parent, asset[key], f"asset {key}")

    trajectory = copy.deepcopy(experiment["trajectory"])
    legacy_post_recovery = raw_video.pop("post_recovery_s", None)
    if legacy_post_recovery is not None:
        legacy_post_recovery = _positive(
            legacy_post_recovery,
            "deprecated video post_recovery_s",
            allow_zero=True,
        )
    if "post_recovery_s" in trajectory and legacy_post_recovery is not None:
        configured_post_recovery = _finite_scalar(
            trajectory["post_recovery_s"], "trajectory post_recovery_s"
        )
        if not math.isclose(configured_post_recovery, legacy_post_recovery):
            raise ConfigError(
                "trajectory.post_recovery_s conflicts with deprecated "
                "video.post_recovery_s"
            )
    trajectory.setdefault(
        "post_recovery_s",
        0.0 if legacy_post_recovery is None else legacy_post_recovery,
    )

    resolved = copy.deepcopy(experiment)
    resolved["trajectory"] = trajectory
    resolved["video"] = {**VIDEO_DEFAULTS, **raw_video}
    resolved["equilibration"] = {
        **EQUILIBRATION_DEFAULTS,
        **raw_equilibration,
    }
    resolved["solver"] = {
        "deterministic": False,
        "newton_strict": False,
        **copy.deepcopy(experiment["solver"]),
    }
    resolved["output"] = {
        "chunk_size_frames": 300,
        **copy.deepcopy(experiment["output"]),
    }
    resolved["mechanics_output_schema_version"] = MECHANICS_OUTPUT_SCHEMA_VERSION
    resolved["asset_config"] = str(asset_path)
    resolved["material_config"] = str(material_path)
    resolved["asset"] = asset
    resolved["material"] = material
    resolved["_experiment_config_path"] = str(experiment_path)

    # Validate before indexing nested values for path resolution so malformed
    # user configs always raise ConfigError instead of raw KeyError/TypeError.
    validate_run_config(resolved)

    if resolved["indenter"]["type"] == "rigid_stl":
        resolved["indenter"]["stl"] = _resolve_path(
            experiment_path.parent,
            resolved["indenter"]["stl"],
            "rigid STL indenter path",
        )
    resolved["output"]["directory"] = _resolve_path(
        experiment_path.parent,
        resolved["output"]["directory"],
        "output directory",
    )
    if resolved["video"]["enabled"]:
        resolved["video"]["path"] = _resolve_path(
            experiment_path.parent,
            resolved["video"]["path"],
            "video path",
        )
    return resolved


def validate_run_config(config: dict[str, Any]) -> None:
    _schema_version(config, "experiment config")

    transform = _mapping(config.get("fingertip_transform"), "fingertip_transform")
    _require(
        transform, ("position_m", "quaternion_xyzw", "scale"), "fingertip_transform"
    )
    _finite_vector(transform["position_m"], 3, "fingertip position_m")
    _finite_vector(
        transform["quaternion_xyzw"],
        4,
        "fingertip quaternion_xyzw",
        nonzero=True,
    )
    _positive(transform["scale"], "fingertip transform scale")

    material = _mapping(config.get("material"), "material config")
    _schema_version(material, "material config")
    _require(
        material,
        ("youngs_modulus_pa", "poisson_ratio", "density_kg_m3", "damping"),
        "material config",
    )
    _positive(material["youngs_modulus_pa"], "material Young's modulus")
    poisson = _finite_scalar(material["poisson_ratio"], "material Poisson ratio")
    _positive(material["density_kg_m3"], "material density")
    if not (-1.0 < poisson < 0.5):
        raise ConfigError("material Poisson ratio must be in (-1, 0.5)")
    damping = _mapping(material["damping"], "material damping")
    _require(damping, ("value",), "material damping")
    _positive(damping["value"], "material damping value", allow_zero=True)

    contact = _mapping(config.get("contact"), "contact")
    _require(contact, ("location_m", "direction"), "contact")
    _finite_vector(contact["location_m"], 3, "contact location_m")
    _finite_vector(contact["direction"], 3, "contact direction", nonzero=True)

    indenter = _mapping(config.get("indenter"), "indenter")
    _require(indenter, ("type", "quaternion_xyzw"), "indenter")
    kind = _nonempty_string(indenter["type"], "indenter type")
    supported = {"sphere", "flat_plate", "cylinder", "rigid_stl"}
    if kind not in supported:
        raise ConfigError(f"indenter type must be one of {sorted(supported)}")
    _finite_vector(
        indenter["quaternion_xyzw"],
        4,
        "indenter quaternion_xyzw",
        nonzero=True,
    )
    if kind == "sphere":
        _require(indenter, ("radius_m",), "sphere indenter")
        _positive(indenter["radius_m"], "sphere radius_m")
    elif kind == "flat_plate":
        _require(indenter, ("width_m", "depth_m", "thickness_m"), "flat plate")
        for field in ("width_m", "depth_m", "thickness_m"):
            _positive(indenter[field], f"flat plate {field}")
    elif kind == "cylinder":
        _require(indenter, ("radius_m", "height_m"), "cylinder indenter")
        _positive(indenter["radius_m"], "cylinder radius_m")
        _positive(indenter["height_m"], "cylinder height_m")
    else:
        _require(
            indenter,
            ("stl", "scale_to_m", "contact_point_local_m"),
            "rigid STL indenter",
        )
        _nonempty_string(indenter["stl"], "rigid STL path")
        _positive(indenter["scale_to_m"], "rigid STL scale_to_m")
        _finite_vector(
            indenter["contact_point_local_m"],
            3,
            "rigid STL contact_point_local_m",
        )

    trajectory = _mapping(config.get("trajectory"), "trajectory")
    _require(
        trajectory,
        ("clearance_m", "indentation_m", "easing", "durations_s", "post_recovery_s"),
        "trajectory",
    )
    _positive(trajectory["clearance_m"], "trajectory clearance_m", allow_zero=True)
    indentation = _positive(trajectory["indentation_m"], "trajectory indentation_m")
    _positive(
        trajectory["post_recovery_s"],
        "trajectory post_recovery_s",
        allow_zero=True,
    )
    if trajectory["easing"] not in {"linear", "smoothstep"}:
        raise ConfigError("trajectory easing must be 'linear' or 'smoothstep'")
    durations = _mapping(trajectory["durations_s"], "trajectory durations_s")
    _require(
        durations, ("approach", "press", "hold", "release", "recovery"), "durations_s"
    )
    for phase in ("approach", "press", "release"):
        _positive(durations[phase], f"trajectory {phase} duration")
    for phase in ("hold", "recovery"):
        _positive(durations[phase], f"trajectory {phase} duration", allow_zero=True)
    slip = _mapping(trajectory.get("lateral_slip", {}), "trajectory lateral_slip")
    if slip:
        _require(slip, ("enabled",), "lateral slip")
        slip_enabled = _boolean(slip["enabled"], "lateral slip enabled")
        if slip_enabled:
            _require(slip, ("duration_s", "distance_m", "direction"), "lateral slip")
            _positive(slip["duration_s"], "lateral slip duration_s")
            _positive(slip["distance_m"], "lateral slip distance_m", allow_zero=True)
            _finite_vector(slip["direction"], 3, "lateral slip direction", nonzero=True)

    monitoring = _mapping(config.get("monitoring"), "monitoring")
    _require(
        monitoring,
        (
            "nominal_wall_thickness_m",
            "engineering_strain_limit",
            "minimum_rest_tet_volume_m3",
            "minimum_rest_tet_quality",
            "maximum_rest_tet_condition_number",
            "minimum_relative_tet_volume",
            "tet_check_interval_substeps",
        ),
        "monitoring",
    )
    wall = _positive(monitoring["nominal_wall_thickness_m"], "nominal wall thickness")
    strain_limit = _positive(
        monitoring["engineering_strain_limit"], "engineering strain limit"
    )
    if indentation > wall * strain_limit + 1.0e-15:
        raise ConfigError(
            "trajectory indentation exceeds nominal_wall_thickness_m * "
            "engineering_strain_limit"
        )
    for field in (
        "minimum_rest_tet_volume_m3",
        "minimum_rest_tet_quality",
        "maximum_rest_tet_condition_number",
        "minimum_relative_tet_volume",
    ):
        _positive(monitoring[field], f"monitoring {field}")
    _integer(
        monitoring["tet_check_interval_substeps"],
        "monitoring tet_check_interval_substeps",
        minimum=1,
    )

    solver = _mapping(config.get("solver"), "solver")
    _require(
        solver,
        (
            "simulation_fps",
            "substeps",
            "vbd_iterations",
            "gravity_m_s2",
            "particle_radius_edge_fraction",
            "deterministic",
            "newton_strict",
            "particle_enable_self_contact",
            "particle_enable_tile_solve",
            "particle_collision_detection_interval",
            "rigid_body_particle_contact_buffer_size",
        ),
        "solver",
    )
    simulation_fps = _integer(
        solver["simulation_fps"], "solver simulation_fps", minimum=1
    )
    _integer(solver["substeps"], "solver substeps", minimum=1)
    _integer(solver["vbd_iterations"], "solver vbd_iterations", minimum=1)
    _finite_scalar(solver["gravity_m_s2"], "solver gravity_m_s2")
    _positive(
        solver["particle_radius_edge_fraction"], "solver particle_radius_edge_fraction"
    )
    _boolean(solver["deterministic"], "solver deterministic")
    _boolean(solver["newton_strict"], "solver newton_strict")
    _boolean(
        solver["particle_enable_self_contact"], "solver particle_enable_self_contact"
    )
    _boolean(solver["particle_enable_tile_solve"], "solver particle_enable_tile_solve")
    _integer(
        solver["particle_collision_detection_interval"],
        "solver particle_collision_detection_interval",
        minimum=-1,
    )
    _integer(
        solver["rigid_body_particle_contact_buffer_size"],
        "solver rigid_body_particle_contact_buffer_size",
        minimum=1,
    )

    contact_parameters = _mapping(
        config.get("contact_parameters"), "contact_parameters"
    )
    _require(
        contact_parameters,
        (
            "normal_stiffness",
            "normal_damping",
            "damping_semantics",
            "static_friction",
            "dynamic_friction",
            "margin_m",
            "force_threshold_n",
            "friction_epsilon_m_s",
            "face_mask_radius_multiplier",
        ),
        "contact_parameters",
    )
    _positive(contact_parameters["normal_stiffness"], "contact normal_stiffness")
    _positive(
        contact_parameters["normal_damping"],
        "contact normal_damping",
        allow_zero=True,
    )
    damping_semantics = _nonempty_string(
        contact_parameters["damping_semantics"],
        "contact damping_semantics",
    )
    if damping_semantics not in SUPPORTED_DAMPING_SEMANTICS:
        raise ConfigError(
            "contact damping_semantics must be one of "
            f"{sorted(SUPPORTED_DAMPING_SEMANTICS)}"
        )
    for field in (
        "static_friction",
        "dynamic_friction",
        "margin_m",
        "force_threshold_n",
    ):
        _positive(contact_parameters[field], f"contact {field}", allow_zero=True)
    _positive(
        contact_parameters["friction_epsilon_m_s"],
        "contact friction_epsilon_m_s",
    )
    _positive(
        contact_parameters["face_mask_radius_multiplier"],
        "contact face_mask_radius_multiplier",
    )

    equilibration = _mapping(config.get("equilibration"), "equilibration")
    _require(
        equilibration,
        (
            "minimum_duration_s",
            "maximum_duration_s",
            "velocity_tolerance_m_s",
            "stable_frames",
            "timeout_behavior",
        ),
        "equilibration",
    )
    minimum = _positive(
        equilibration["minimum_duration_s"],
        "equilibration minimum_duration_s",
        allow_zero=True,
    )
    maximum = _positive(
        equilibration["maximum_duration_s"],
        "equilibration maximum_duration_s",
    )
    if maximum < minimum:
        raise ConfigError(
            "equilibration maximum_duration_s must be >= minimum_duration_s"
        )
    _positive(
        equilibration["velocity_tolerance_m_s"],
        "equilibration velocity_tolerance_m_s",
    )
    _integer(
        equilibration["stable_frames"],
        "equilibration stable_frames",
        minimum=1,
    )
    if equilibration["timeout_behavior"] not in {"warn", "fail"}:
        raise ConfigError("equilibration timeout_behavior must be 'warn' or 'fail'")

    output = _mapping(config.get("output"), "output")
    _require(output, ("directory", "rate_hz", "chunk_size_frames"), "output")
    _nonempty_string(output["directory"], "output directory")
    output_hz = _positive(output["rate_hz"], "output rate_hz")
    if output_hz > simulation_fps:
        raise ConfigError("output rate_hz cannot exceed simulation_fps")
    _integer(output["chunk_size_frames"], "output chunk_size_frames", minimum=1)

    video = _mapping(config.get("video"), "video")
    _require(
        video, ("enabled", "path", "fps", "codec", "quality", "include_ui"), "video"
    )
    _boolean(video["enabled"], "video enabled")
    _nonempty_string(video["path"], "video path")
    video_fps = _positive(video["fps"], "video fps")
    if video_fps > simulation_fps:
        raise ConfigError("video fps cannot exceed simulation_fps")
    _nonempty_string(video["codec"], "video codec")
    quality = _integer(video["quality"], "video quality", minimum=0)
    if quality > 10:
        raise ConfigError("video quality must be between 0 and 10")
    _boolean(video["include_ui"], "video include_ui")

    viewer = _mapping(config.get("viewer", {}), "viewer")
    if "render_contacts" in viewer:
        _boolean(viewer["render_contacts"], "viewer render_contacts")
    camera = viewer.get("camera")
    if camera is not None:
        camera = _mapping(camera, "viewer camera")
        _require(
            camera, ("position_m", "pitch_degrees", "yaw_degrees"), "viewer camera"
        )
        _finite_vector(camera["position_m"], 3, "viewer camera position_m")
        _finite_scalar(camera["pitch_degrees"], "viewer camera pitch_degrees")
        _finite_scalar(camera["yaw_degrees"], "viewer camera yaw_degrees")


def material_lame_parameters(material: dict[str, Any]) -> tuple[float, float]:
    youngs = float(material["youngs_modulus_pa"])
    poisson = float(material["poisson_ratio"])
    mu = youngs / (2.0 * (1.0 + poisson))
    lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam
