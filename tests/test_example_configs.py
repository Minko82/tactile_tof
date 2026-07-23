import copy
import json
from pathlib import Path

import pytest

from sim.mechanics.config import VIDEO_DEFAULTS, ConfigError, load_run_config

EXPERIMENTS = Path("sim/config/mechanics/experiments")
BASE_EXPERIMENT = EXPERIMENTS / "sphere_regression.json"
ACTIVE_MATERIAL = "sorta_clear_37_provisional.json"
ACTIVE_MATERIAL_PATH = (
    Path("sim/config/mechanics/materials") / ACTIVE_MATERIAL
)


def _portable_source() -> dict:
    source = json.loads(BASE_EXPERIMENT.read_text(encoding="utf-8"))
    source["asset_config"] = str((EXPERIMENTS / source["asset_config"]).resolve())
    source["material_config"] = str((EXPERIMENTS / source["material_config"]).resolve())
    return source


def _write_config(tmp_path: Path, source: dict, name: str = "config.json") -> Path:
    config_path = tmp_path / name
    config_path.write_text(json.dumps(source), encoding="utf-8")
    return config_path


@pytest.mark.parametrize("config_path", sorted(EXPERIMENTS.glob("*.json")))
def test_every_committed_experiment_config_loads(config_path):
    resolved = load_run_config(config_path)
    assert resolved["mechanics_output_schema_version"] == 2
    assert Path(resolved["asset_config"]).is_absolute()
    assert Path(resolved["material_config"]).is_absolute()
    assert Path(resolved["output"]["directory"]).is_absolute()
    for field in ("surface_stl", "volume_msh", "regions_npz", "surface_mapping_npz"):
        assert Path(resolved["asset"][field]).is_absolute()
    assert Path(resolved["material_config"]).name == ACTIVE_MATERIAL
    assert resolved["material"]["product_family"] == "SORTA-Clear"
    assert resolved["material"]["grade"] == "37"
    assert resolved["material"]["calibration_status"] == "provisional"


def test_legacy_ecoflex_profile_remains_loadable_for_regression(tmp_path):
    source = _portable_source()
    source["material_config"] = str(
        Path("sim/config/mechanics/materials/ecoflex_00_30.json").resolve()
    )

    resolved = load_run_config(_write_config(tmp_path, source))

    assert resolved["material"]["material_id"] == "ecoflex_00_30_regression"
    assert resolved["material"]["product_family"] == "Ecoflex"
    assert resolved["material"]["calibration_status"] == "provisional"


def test_sorta_clear_37_profile_is_grade_specific_and_not_a_renamed_ecoflex():
    material = json.loads(ACTIVE_MATERIAL_PATH.read_text(encoding="utf-8"))

    assert material["material_id"] == "sorta_clear_37_provisional"
    assert material["grade"] == "37"
    assert material["density_kg_m3"] == 1080.0
    assert material["youngs_modulus_pa"] == 1_064_000.0
    assert material["youngs_modulus_pa"] != 50_000.0
    assert material["youngs_modulus_pa"] != pytest.approx(90.0 * 6894.757293)
    assert material["calibration_status"] == "provisional"
    assert material["calibration_report"] is None


def test_invalid_material_calibration_status_is_rejected(tmp_path):
    source = _portable_source()
    material_path = Path(source["material_config"])
    material = json.loads(material_path.read_text(encoding="utf-8"))
    material["calibration_status"] = "unknown"
    invalid_material = tmp_path / "invalid_material.json"
    invalid_material.write_text(json.dumps(material), encoding="utf-8")
    source["material_config"] = str(invalid_material)

    with pytest.raises(ConfigError, match="calibration_status"):
        load_run_config(_write_config(tmp_path, source))


def test_video_section_is_optional_and_does_not_own_physics_duration(tmp_path):
    source = _portable_source()
    expected_post_recovery = source["trajectory"]["post_recovery_s"]
    source.pop("video", None)

    resolved = load_run_config(_write_config(tmp_path, source))

    assert resolved["video"] == VIDEO_DEFAULTS
    assert resolved["video"]["path"] == "mechanics.mp4"
    assert resolved["trajectory"]["post_recovery_s"] == expected_post_recovery


def test_legacy_video_post_recovery_is_migrated_to_trajectory(tmp_path):
    source = _portable_source()
    source["trajectory"].pop("post_recovery_s")
    source["video"]["post_recovery_s"] = 0.75

    resolved = load_run_config(_write_config(tmp_path, source))

    assert resolved["trajectory"]["post_recovery_s"] == 0.75
    assert "post_recovery_s" not in resolved["video"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contact.direction", [0.0, 0.0, 0.0]),
        ("contact.direction", [float("nan"), 0.0, 1.0]),
        ("indenter.quaternion_xyzw", [0.0, 0.0, 0.0, 0.0]),
        ("solver.substeps", 1.5),
        ("solver.deterministic", 1),
        ("solver.particle_enable_self_contact", 0),
        ("solver.rigid_body_particle_contact_buffer_size", 0),
        ("contact_parameters.normal_stiffness", 0.0),
        ("contact_parameters.normal_damping", -1.0),
        ("contact_parameters.static_friction", -1.0),
        ("contact_parameters.dynamic_friction", -1.0),
        ("contact_parameters.margin_m", -1.0),
        ("contact_parameters.force_threshold_n", -1.0),
        ("contact_parameters.friction_epsilon_m_s", 0.0),
        ("contact_parameters.face_mask_radius_multiplier", 0.0),
        ("contact_parameters.damping_semantics", "unknown"),
    ],
)
def test_invalid_values_are_rejected_with_config_error(tmp_path, field, value):
    source = _portable_source()
    section, name = field.split(".")
    source[section][name] = value

    with pytest.raises(ConfigError):
        load_run_config(_write_config(tmp_path, source))


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("solver", "particle_enable_self_contact"),
        ("solver", "particle_enable_tile_solve"),
        ("solver", "particle_collision_detection_interval"),
        ("solver", "rigid_body_particle_contact_buffer_size"),
        ("contact_parameters", "normal_stiffness"),
        ("contact_parameters", "normal_damping"),
        ("contact_parameters", "damping_semantics"),
        ("contact_parameters", "static_friction"),
        ("contact_parameters", "dynamic_friction"),
        ("contact_parameters", "margin_m"),
        ("contact_parameters", "force_threshold_n"),
        ("contact_parameters", "friction_epsilon_m_s"),
        ("contact_parameters", "face_mask_radius_multiplier"),
        ("output", "directory"),
    ],
)
def test_runner_required_fields_fail_cleanly_before_indexing(tmp_path, section, field):
    source = _portable_source()
    del source[section][field]

    with pytest.raises(ConfigError, match=field):
        load_run_config(_write_config(tmp_path, source))


def test_unsupported_schema_version_is_rejected(tmp_path):
    source = _portable_source()
    source["schema_version"] = 2

    with pytest.raises(ConfigError, match="schema_version"):
        load_run_config(_write_config(tmp_path, source))


def test_video_toggle_leaves_trajectory_configuration_identical(tmp_path):
    disabled = _portable_source()
    enabled = copy.deepcopy(disabled)
    disabled["video"]["enabled"] = False
    enabled["video"]["enabled"] = True

    disabled_run = load_run_config(_write_config(tmp_path, disabled, "disabled.json"))
    enabled_run = load_run_config(_write_config(tmp_path, enabled, "enabled.json"))

    assert disabled_run["trajectory"] == enabled_run["trajectory"]
