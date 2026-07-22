"""Strict JSON loading for the mechanics runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


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
            "video",
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
    if len(transform["position_m"]) != 3 or len(transform["quaternion_xyzw"]) != 4:
        raise ConfigError(
            "fingertip transform requires a 3-vector position and xyzw quaternion"
        )
    scale = float(transform["scale"])
    if scale <= 0.0:
        raise ConfigError("fingertip transform scale must be positive")

    material = config["material"]
    youngs = float(material["youngs_modulus_pa"])
    poisson = float(material["poisson_ratio"])
    density = float(material["density_kg_m3"])
    if youngs <= 0.0 or density <= 0.0 or not (-1.0 < poisson < 0.5):
        raise ConfigError(
            "material E/density must be positive and Poisson ratio must be in (-1, 0.5)"
        )

    contact = config["contact"]
    _require(contact, ("location_m", "direction"), "contact")
    if len(contact["location_m"]) != 3 or len(contact["direction"]) != 3:
        raise ConfigError("contact location and direction must be 3-vectors")
    indenter = config["indenter"]
    supported = {"sphere", "flat_plate", "cylinder", "rigid_stl"}
    if indenter.get("type") not in supported:
        raise ConfigError(f"indenter type must be one of {sorted(supported)}")
    if len(indenter.get("quaternion_xyzw", [])) != 4:
        raise ConfigError("indenter quaternion_xyzw is required")

    trajectory = config["trajectory"]
    _require(
        trajectory,
        ("clearance_m", "indentation_m", "easing", "durations_s"),
        "trajectory",
    )
    wall = float(config["monitoring"]["nominal_wall_thickness_m"])
    strain_limit = float(config["monitoring"]["engineering_strain_limit"])
    if float(trajectory["indentation_m"]) > wall * strain_limit:
        raise ConfigError(
            "trajectory indentation exceeds nominal_wall_thickness_m * engineering_strain_limit"
        )
    solver = config["solver"]
    _require(
        solver,
        (
            "simulation_fps",
            "substeps",
            "vbd_iterations",
            "gravity_m_s2",
            "particle_radius_edge_fraction",
        ),
        "solver",
    )
    if int(solver["simulation_fps"]) <= 0 or int(solver["substeps"]) <= 0:
        raise ConfigError("simulation_fps and substeps must be positive")
    output_hz = float(config["output"]["rate_hz"])
    if output_hz <= 0.0 or output_hz > float(solver["simulation_fps"]):
        raise ConfigError(
            "output rate_hz must be positive and no greater than simulation_fps"
        )

    video = config["video"]
    _require(
        video,
        ("enabled", "path", "fps", "codec", "quality", "include_ui", "post_recovery_s"),
        "video",
    )
    video_fps = float(video["fps"])
    if video_fps <= 0.0 or video_fps > float(solver["simulation_fps"]):
        raise ConfigError(
            "video fps must be positive and no greater than simulation_fps"
        )
    quality = int(video["quality"])
    if quality < 0 or quality > 10:
        raise ConfigError("video quality must be between 0 and 10")
    if float(video["post_recovery_s"]) < 0.0:
        raise ConfigError("video post_recovery_s cannot be negative")


def material_lame_parameters(material: dict[str, Any]) -> tuple[float, float]:
    youngs = float(material["youngs_modulus_pa"])
    poisson = float(material["poisson_ratio"])
    mu = youngs / (2.0 * (1.0 + poisson))
    lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam
