from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sim.mechanics.exporter import load_frame_chunks

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "sim/config/mechanics/experiments/sphere_visible_demo.json"


def require_newton_runtime() -> None:
    if (
        importlib.util.find_spec("newton") is None
        or importlib.util.find_spec("warp") is None
    ):
        pytest.skip(
            "run with: uv --native-tls run --project sim/newton --extra examples "
            "python -m pytest -m newton_integration"
        )


def abbreviated_config(tmp_path: Path, name: str) -> Path:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base = BASE_CONFIG.parent
    config["asset_config"] = str((base / config["asset_config"]).resolve())
    config["material_config"] = str((base / config["material_config"]).resolve())
    config["experiment_id"] = name
    config["trajectory"]["durations_s"] = {
        "approach": 0.05,
        "press": 0.10,
        "hold": 0.10,
        "release": 0.10,
        "recovery": 0.15,
    }
    config["trajectory"]["post_recovery_s"] = 0.0
    config["equilibration"] = {
        "minimum_duration_s": 0.50,
        "maximum_duration_s": 2.00,
        "velocity_tolerance_m_s": 0.0001,
        "stable_frames": 5,
        "timeout_behavior": "fail",
    }
    config["solver"].update(
        {
            "deterministic": True,
            "newton_strict": True,
        }
    )
    config["output"] = {
        "directory": str((tmp_path / name).resolve()),
        "rate_hz": 30.0,
        "chunk_size_frames": 16,
    }
    config.pop("video", None)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def run_rollout_process(
    config_path: Path, output_dir: Path
) -> subprocess.CompletedProcess[str]:
    require_newton_runtime()
    command = [
        sys.executable,
        str(REPO_ROOT / "sim/scripts/run_touch_mechanics.py"),
        "--viewer",
        "null",
        "--headless",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--strict-newton-version",
    ]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def run_rollout(
    config_path: Path, output_dir: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    completed = run_rollout_process(config_path, output_dir)
    if completed.returncode:
        pytest.fail(
            "Newton rollout failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    frames = load_frame_chunks(output_dir)
    with (output_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    return frames, metrics
