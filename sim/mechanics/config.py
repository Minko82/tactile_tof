"""Strict JSON loading and defaults for the mechanics runner."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .schema import MECHANICS_OUTPUT_SCHEMA_VERSION


VIDEO_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "path": "mechanics.mp4",
    "fps": 30,
    "codec": "libx264",
    "quality": 7,
    "include_ui": False,
    "post_recovery_s": 0.0,
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


def _resolve_path(base: Path, value: str) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve())


def _require(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{context} is missing required fields: {', '.join(missing)}")


def _finite_scalar(value: Any, context: str) -> float:
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


def load_run_config(path: str | Path) -> dict[str, Any]:
    experiment_path = Path(path).resolve()
    experiment = _read_json(experiment_path)
    _require(
        experiment,
        (
            "schema_version",
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
    asset_path = Path(_resolve_path(experiment_path.parent, experiment["asset_config"]))
    material_path = Path(
        _resolve_path(experiment_path.parent, experiment["material_config"])
    )
    asset = _read_json(asset_path)
    material = _read_json(material_path)
    _require(
        asset,
        (
            "schema_version",
            "surface_stl",
            "volume_msh",
            "regions_npz",
            "surface_mapping_npz",
        ),
        "asset config",
    )
    _require(
        material,
        ("youngs_modulus_pa", "poisson_ratio", "density_kg_m3", "damping"),
        "material config",
    )

    asset = copy.deepcopy(asset)
    for key in ("surface_stl", "volume_msh", "regions_npz", "surface_mapping_npz"):
        asset[key] = _resolve_path(asset_path.parent, asset[key])

    resolved = copy.deepcopy(experiment)
    resolved["video"] = {**VIDEO_DEFAULTS, **copy.deepcopy(experiment.get("video", {}))}
    resolved["equilibration"] = {
        **EQUILIBRATION_DEFAULTS,
        **copy.deepcopy(experiment.get("equilibration", {})),
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
    if resolved["indenter"]["type"] == "rigid_stl":
        resolved["indenter"]["stl"] = _resolve_path(
            experiment_path.parent, resolved["indenter"]["stl"]
        )
    resolved["output"]["directory"] = _resolve_path(
        experiment_path.parent, resolved["output"]["directory"]
    )
    if bool(resolved["video"]["enabled"]):
        resolved["video"]["path"] = _resolve_path(
            experiment_path.parent, resolved["video"]["path"]
        )
    resolved["_experiment_config_path"] = str(experiment_path)
    validate_run_config(resolved)
    return resolved


def validate_run_config(config: dict[str, Any]) -> None:
    transform = config["fingertip_transform"]
    _require(
        transform, ("position_m", "quaternion_xyzw", "scale"), "fingertip_transform"
    )
    _finite_vector(transform["position_m"], 3, "fingertip position_m")
    _finite_vector(
        transform["quaternion_xyzw"], 4, "fingertip quaternion_xyzw", nonzero=True
    )
    _positive(transform["scale"], "fingertip transform scale")

    material = config["material"]
    youngs = _positive(material["youngs_modulus_pa"], "material Young's modulus")
    poisson = _finite_scalar(material["poisson_ratio"], "material Poisson ratio")
    density = _positive(material["density_kg_m3"], "material density")
    if youngs <= 0.0 or density <= 0.0 or not (-1.0 < poisson < 0.5):
        raise ConfigError(
            "material E/density must be positive and Poisson ratio must be in (-1, 0.5)"
        )
    if not isinstance(material["damping"], dict) or "value" not in material["damping"]:
        raise ConfigError("material damping must contain a value")
    _positive(material["damping"]["value"], "material damping", allow_zero=True)

    contact = config["contact"]
    _require(contact, ("location_m", "direction"), "contact")
    _finite_vector(contact["location_m"], 3, "contact location_m")
    _finite_vector(contact["direction"], 3, "contact direction", nonzero=True)

    indenter = config["indenter"]
    supported = {"sphere", "flat_plate", "cylinder", "rigid_stl"}
    kind = indenter.get("type")
    if kind not in supported:
        raise ConfigError(f"indenter type must be one of {sorted(supported)}")
    _finite_vector(
        indenter.get("quaternion_xyzw"), 4, "indenter quaternion_xyzw", nonzero=True
    )
    if kind == "sphere":
        _positive(indenter.get("radius_m"), "sphere radius_m")
    elif kind == "flat_plate":
        for field in ("width_m", "depth_m", "thickness_m"):
            _positive(indenter.get(field), f"flat plate {field}")
    elif kind == "cylinder":
        _positive(indenter.get("radius_m"), "cylinder radius_m")
        _positive(indenter.get("height_m"), "cylinder height_m")
    else:
        if not str(indenter.get("stl", "")).strip():
            raise ConfigError("rigid STL indenter requires stl")
        _positive(indenter.get("scale_to_m"), "rigid STL scale_to_m")
        _finite_vector(
            indenter.get("contact_point_local_m"), 3, "rigid STL contact_point_local_m"
        )

    trajectory = config["trajectory"]
    _require(
        trajectory,
        ("clearance_m", "indentation_m", "easing", "durations_s"),
        "trajectory",
    )
    _positive(trajectory["clearance_m"], "trajectory clearance_m", allow_zero=True)
    indentation = _positive(trajectory["indentation_m"], "trajectory indentation_m")
    if trajectory["easing"] not in {"linear", "smoothstep"}:
        raise ConfigError("trajectory easing must be 'linear' or 'smoothstep'")
    durations = trajectory["durations_s"]
    _require(
        durations, ("approach", "press", "hold", "release", "recovery"), "durations_s"
    )
    for phase in ("approach", "press", "release"):
        _positive(durations[phase], f"trajectory {phase} duration")
    for phase in ("hold", "recovery"):
        _positive(durations[phase], f"trajectory {phase} duration", allow_zero=True)
    slip = trajectory.get("lateral_slip", {})
    if bool(slip.get("enabled", False)):
        _positive(slip.get("duration_s"), "lateral slip duration_s")
        _positive(slip.get("distance_m"), "lateral slip distance_m", allow_zero=True)
        _finite_vector(slip.get("direction"), 3, "lateral slip direction", nonzero=True)

    monitoring = config["monitoring"]
    wall = _positive(monitoring["nominal_wall_thickness_m"], "nominal wall thickness")
    strain_limit = _positive(
        monitoring["engineering_strain_limit"], "engineering strain limit"
    )
    if indentation > wall * strain_limit + 1.0e-15:
        raise ConfigError(
            "trajectory indentation exceeds nominal_wall_thickness_m * engineering_strain_limit"
        )
    for field in (
        "minimum_rest_tet_volume_m3",
        "minimum_rest_tet_quality",
        "maximum_rest_tet_condition_number",
        "minimum_relative_tet_volume",
    ):
        _positive(monitoring[field], f"monitoring {field}")
    if int(monitoring["tet_check_interval_substeps"]) <= 0:
        raise ConfigError("tet_check_interval_substeps must be positive")

    solver = config["solver"]
    _require(
        solver,
        (
            "simulation_fps",
            "substeps",
            "vbd_iterations",
            "gravity_m_s2",
            "particle_radius_edge_fraction",
            "deterministic",
        ),
        "solver",
    )
    if int(solver["simulation_fps"]) <= 0 or int(solver["substeps"]) <= 0:
        raise ConfigError("simulation_fps and substeps must be positive")
    if int(solver["vbd_iterations"]) <= 0:
        raise ConfigError("vbd_iterations must be positive")
    _finite_scalar(solver["gravity_m_s2"], "solver gravity_m_s2")
    _positive(
        solver["particle_radius_edge_fraction"], "solver particle_radius_edge_fraction"
    )

    equilibration = config["equilibration"]
    minimum = _positive(
        equilibration["minimum_duration_s"],
        "equilibration minimum_duration_s",
        allow_zero=True,
    )
    maximum = _positive(
        equilibration["maximum_duration_s"], "equilibration maximum_duration_s"
    )
    if maximum < minimum:
        raise ConfigError(
            "equilibration maximum_duration_s must be >= minimum_duration_s"
        )
    _positive(
        equilibration["velocity_tolerance_m_s"], "equilibration velocity_tolerance_m_s"
    )
    if int(equilibration["stable_frames"]) <= 0:
        raise ConfigError("equilibration stable_frames must be positive")
    if equilibration["timeout_behavior"] not in {"warn", "fail"}:
        raise ConfigError("equilibration timeout_behavior must be 'warn' or 'fail'")

    output = config["output"]
    output_hz = _positive(output["rate_hz"], "output rate_hz")
    if output_hz > float(solver["simulation_fps"]):
        raise ConfigError("output rate_hz cannot exceed simulation_fps")
    if int(output["chunk_size_frames"]) <= 0:
        raise ConfigError("output chunk_size_frames must be positive")

    video = config["video"]
    _require(
        video,
        ("enabled", "path", "fps", "codec", "quality", "include_ui", "post_recovery_s"),
        "video",
    )
    if not str(video["path"]).strip():
        raise ConfigError("video path cannot be empty")
    video_fps = _positive(video["fps"], "video fps")
    if video_fps > float(solver["simulation_fps"]):
        raise ConfigError("video fps cannot exceed simulation_fps")
    quality = int(video["quality"])
    if quality < 0 or quality > 10:
        raise ConfigError("video quality must be between 0 and 10")
    _positive(video["post_recovery_s"], "video post_recovery_s", allow_zero=True)

    camera = config.get("viewer", {}).get("camera")
    if camera is not None:
        _finite_vector(camera.get("position_m"), 3, "viewer camera position_m")
        _finite_scalar(camera.get("pitch_degrees"), "viewer camera pitch_degrees")
        _finite_scalar(camera.get("yaw_degrees"), "viewer camera yaw_degrees")


def material_lame_parameters(material: dict[str, Any]) -> tuple[float, float]:
    youngs = float(material["youngs_modulus_pa"])
    poisson = float(material["poisson_ratio"])
    mu = youngs / (2.0 * (1.0 + poisson))
    lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam
