#!/usr/bin/env python3
"""Measure wall-clock mechanics frame rates for the optimization profiles."""

# ruff: noqa: E402 -- the in-tree Newton checkout must precede package imports.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_NEWTON_ROOT = Path(__file__).resolve().parents[1] / "newton"
if (LOCAL_NEWTON_ROOT / "newton").is_dir():
    sys.path.insert(0, str(LOCAL_NEWTON_ROOT))

import warp as wp

from sim.mechanics.interactive_runner import InteractiveTouchController
from sim.mechanics.interactive_safety import ProbePose
from sim.mechanics.newton_runner import TouchMechanicsControllerV2


EXPERIMENTS = REPO_ROOT / "sim/config/mechanics/experiments"


def _args(config: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=str(config),
        output_dir=str(output_dir),
        record_video=False,
        video_path=None,
        strict_newton=False,
    )


def _timed_frames(controller, count: int, *, render: bool = False) -> dict[str, float]:
    wp.synchronize()
    started = time.perf_counter()
    for _ in range(count):
        controller.step()
        if render:
            controller.render()
    wp.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "frames": count,
        "seconds": elapsed,
        "wall_clock_fps": count / elapsed,
    }


def benchmark_interactive(
    output_root: Path, frames: int, config: Path
) -> dict[str, object]:
    controller = InteractiveTouchController(
        None, _args(config, output_root / "interactive")
    )
    try:
        while controller.equilibrated_particle_positions is None:
            controller.step()

        free_space = _timed_frames(controller, frames)

        start = controller.current_body_position.copy()
        controller.mouse_target_pose = ProbePose(
            start + np.asarray([0.0, 0.0, -0.0105]),
            controller.indenter_quaternion,
        )
        for _ in range(240):
            controller.step()
            if controller.last_contact_flag:
                break
        if not controller.last_contact_flag:
            raise RuntimeError("benchmark probe did not reach contact")
        active_contact = _timed_frames(controller, frames)
        return {
            "free_space_interaction": free_space,
            "active_fingertip_contact": active_contact,
        }
    finally:
        controller.finalize()


def _make_gl_viewer():
    try:
        from newton.viewer import ViewerGL
    except ImportError as exc:  # pragma: no cover - depends on optional GL runtime
        raise RuntimeError("the GL benchmark requires Newton viewer dependencies") from exc
    return ViewerGL()


def benchmark_scripted(
    output_root: Path, *, config: Path, gl: bool
) -> dict[str, float]:
    viewer = _make_gl_viewer() if gl else None
    controller = TouchMechanicsControllerV2(
        viewer, _args(config, output_root / ("sphere_gl" if gl else "sphere_headless"))
    )
    frame_count = 0
    try:
        wp.synchronize()
        started = time.perf_counter()
        while not controller.finalized:
            controller.step()
            if gl:
                controller.render()
            frame_count += 1
        wp.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "frames": frame_count,
            "seconds": elapsed,
            "wall_clock_fps": frame_count / elapsed,
        }
    finally:
        controller.finalize()
        if viewer is not None:
            viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("interactive", "sphere-headless", "sphere-gl", "all"),
        default="all",
    )
    parser.add_argument("--interactive-frames", type=int, default=30)
    parser.add_argument(
        "--interactive-config",
        type=Path,
        default=EXPERIMENTS / "interactive_manual.json",
    )
    parser.add_argument(
        "--sphere-config",
        type=Path,
        default=EXPERIMENTS / "sphere_regression.json",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mechanics_benchmark_") as temporary:
        output_root = Path(temporary)
        results: dict[str, object] = {}
        if args.profile in {"interactive", "all"}:
            results.update(
                benchmark_interactive(
                    output_root,
                    args.interactive_frames,
                    args.interactive_config.resolve(),
                )
            )
        if args.profile in {"sphere-headless", "all"}:
            results["scripted_sphere_headless"] = benchmark_scripted(
                output_root,
                config=args.sphere_config.resolve(),
                gl=False,
            )
        if args.profile in {"sphere-gl", "all"}:
            results["scripted_sphere_gl"] = benchmark_scripted(
                output_root,
                config=args.sphere_config.resolve(),
                gl=True,
            )

    payload = json.dumps(results, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
