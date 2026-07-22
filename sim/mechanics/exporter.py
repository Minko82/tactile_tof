"""Mechanical frame and scalar-metric export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METRIC_COLUMNS = (
    "timestamp_s",
    "trajectory_phase",
    "contact_flag",
    "contact_area_m2",
    "normal_force_n",
    "tangential_force_x_n",
    "tangential_force_y_n",
    "tangential_force_z_n",
    "slip_velocity_x_m_s",
    "slip_velocity_y_m_s",
    "slip_velocity_z_m_s",
    "maximum_displacement_m",
    "minimum_relative_tet_volume",
    "inverted_tet_count",
)


class MechanicalDataExporter:
    def __init__(self, output_dir: str | Path, resolved_config: dict[str, Any]):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames: dict[str, list[np.ndarray | float | int | bool | str]] = {}
        self.metrics: list[dict[str, Any]] = []
        self.finalized = False
        (self.output_dir / "run_config.json").write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def append(self, frame: dict[str, Any], metric: dict[str, Any]) -> None:
        if self.finalized:
            raise RuntimeError("cannot append after exporter finalization")
        for name, value in frame.items():
            self.frames.setdefault(name, []).append(value)
        self.metrics.append(metric)

    def finalize(self) -> None:
        if self.finalized:
            return
        arrays: dict[str, np.ndarray] = {}
        for name, values in self.frames.items():
            if name == "trajectory_phase":
                arrays[name] = np.asarray(values, dtype="U32")
            else:
                arrays[name] = np.asarray(values)
        np.savez_compressed(self.output_dir / "frames.npz", **arrays)
        with (self.output_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=METRIC_COLUMNS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(self.metrics)
        self.finalized = True


def export_failure_state(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
