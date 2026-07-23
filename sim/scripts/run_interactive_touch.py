#!/usr/bin/env python3

# ruff: noqa: E402 -- the in-tree Newton checkout must precede package imports.

"""Launch the manual Newton fingertip deformation explorer."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_NEWTON_ROOT = REPO_ROOT / "sim/newton"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOCAL_NEWTON_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_NEWTON_ROOT))

import newton.examples

from sim.mechanics.interactive_runner import InteractiveTouchController


def close_viewer_after_fatal_error(viewer) -> None:
    """Close GL deterministically before propagating a fatal mechanics error."""

    renderer = getattr(viewer, "renderer", None)
    try:
        if renderer is not None and hasattr(renderer, "app"):
            renderer.app.event_loop.exit()
    except Exception as exc:  # pragma: no cover - backend-specific cleanup
        print(f"[interactive] viewer exit request failed: {exc}", file=sys.stderr)
    try:
        viewer.close()
    except Exception as exc:  # pragma: no cover - backend-specific cleanup
        print(f"[interactive] viewer close failed: {exc}", file=sys.stderr)


def main() -> None:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "config/mechanics/experiments/interactive_manual.json"
    )
    parser = newton.examples.create_parser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--strict-newton-version",
        dest="strict_newton",
        action="store_true",
    )
    parser.set_defaults(viewer="gl", headless=False)
    viewer, args = newton.examples.init(parser)
    if args.viewer != "gl" or args.headless:
        parser.error(
            "interactive touch requires the visible GL viewer "
            "(--viewer gl --no-headless)"
        )

    controller = InteractiveTouchController(viewer, args)
    try:
        newton.examples.run(controller, args)
    except BaseException:
        controller.finalize()
        close_viewer_after_fatal_error(viewer)
        raise
    finally:
        controller.finalize()


if __name__ == "__main__":
    main()
