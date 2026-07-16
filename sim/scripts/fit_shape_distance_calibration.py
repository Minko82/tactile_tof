"""CLI for the cup_spoon_ascending_v1 shared distance calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from distance_calibration import SupportPreflightError, TrainingManifest, fit_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit the ascending-only shared VL53L5CX distance layer.")
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("strict", "diagnostic"), default="strict")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = TrainingManifest.from_json(args.training_manifest)
    try:
        result = fit_manifest(manifest, mode=args.mode, preflight_only=args.preflight_only)
    except SupportPreflightError as exc:
        print(json.dumps(exc.report, indent=2, sort_keys=True))
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.preflight_only:
        print(f"[distance-calibration] wrote support report: {manifest.coverage_report}")
    else:
        print(f"[distance-calibration] wrote artifact: {manifest.output_artifact}")


if __name__ == "__main__":
    main()
