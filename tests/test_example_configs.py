import json
from pathlib import Path

import pytest

from sim.mechanics.config import VIDEO_DEFAULTS, ConfigError, load_run_config


EXPERIMENTS = Path("sim/config/mechanics/experiments")


@pytest.mark.parametrize("config_path", sorted(EXPERIMENTS.glob("*.json")))
def test_every_committed_experiment_config_loads(config_path):
    resolved = load_run_config(config_path)
    assert resolved["mechanics_output_schema_version"] == 2
    assert Path(resolved["asset_config"]).is_absolute()
    assert Path(resolved["material_config"]).is_absolute()
    assert Path(resolved["output"]["directory"]).is_absolute()
    for field in ("surface_stl", "volume_msh", "regions_npz", "surface_mapping_npz"):
        assert Path(resolved["asset"][field]).is_absolute()


def test_video_section_is_optional_and_path_is_not_resolved_when_disabled(tmp_path):
    source = json.loads((EXPERIMENTS / "sphere_regression.json").read_text())
    source.pop("video", None)
    source["asset_config"] = str((EXPERIMENTS / source["asset_config"]).resolve())
    source["material_config"] = str((EXPERIMENTS / source["material_config"]).resolve())
    config_path = tmp_path / "without_video.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")

    resolved = load_run_config(config_path)

    assert resolved["video"] == VIDEO_DEFAULTS
    assert resolved["video"]["path"] == "mechanics.mp4"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contact.direction", [0.0, 0.0, 0.0]),
        ("contact.direction", [float("nan"), 0.0, 1.0]),
        ("indenter.quaternion_xyzw", [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_invalid_vectors_and_quaternions_are_rejected(tmp_path, field, value):
    source = json.loads((EXPERIMENTS / "sphere_regression.json").read_text())
    source["asset_config"] = str((EXPERIMENTS / source["asset_config"]).resolve())
    source["material_config"] = str((EXPERIMENTS / source["material_config"]).resolve())
    section, name = field.split(".")
    source[section][name] = value
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_run_config(config_path)
