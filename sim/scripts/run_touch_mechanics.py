#!/usr/bin/env python3
"""Run the Newton normal-indentation mechanics simulator.

This entry point is deliberately independent of the ToF simulator. It drives
a prescribed repeatable trajectory and exports schema-v2 deformation and
estimated penalty-contact metrics.
"""

# ruff: noqa: E402 -- the in-tree Newton checkout must precede package imports.

from __future__ import annotations

import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_NEWTON_ROOT = Path(__file__).resolve().parents[1] / "newton"
if (LOCAL_NEWTON_ROOT / "newton").is_dir():
    sys.path.insert(0, str(LOCAL_NEWTON_ROOT))

import newton
import newton.examples

from sim.mechanics.config import load_run_config
from sim.mechanics.newton_runner import TouchMechanicsControllerV2
from sim.mechanics.trajectory import PrescribedTrajectory


def main() -> None:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "config/mechanics/experiments/sphere_regression.json"
    )
    parser = newton.examples.create_parser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", default=None)
    parser.add_argument(
        "--strict-newton-version",
        dest="strict_newton",
        action="store_true",
    )
    preliminary, _ = parser.parse_known_args()
    config = load_run_config(preliminary.config)
    trajectory = PrescribedTrajectory(config["trajectory"])
    video_enabled = bool(
        config["video"]["enabled"] or preliminary.record_video or preliminary.video_path
    )
    if video_enabled and preliminary.viewer != "gl":
        parser.error("MP4 recording requires --viewer gl (optionally with --headless)")

    maximum_duration_s = (
        float(config["equilibration"]["maximum_duration_s"])
        + trajectory.total_duration_s
    )
    if video_enabled:
        maximum_duration_s += float(config["video"]["post_recovery_s"])
    simulation_fps = float(config["solver"]["simulation_fps"])
    parser.set_defaults(
        num_frames=int(math.ceil(maximum_duration_s * simulation_fps)) + 2
    )

    viewer, args = newton.examples.init(parser)
    controller = TouchMechanicsControllerV2(viewer, args)
    try:
        newton.examples.run(controller, args)
    finally:
        controller.finalize()


if __name__ == "__main__":
    main()
