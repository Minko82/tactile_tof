"""Chunked, schema-v2 mechanical frame and scalar-metric export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    CONTACT_FORCE_ESTIMATOR_VERSION,
    CONTACT_METRIC_MODEL,
    DEPRECATED_ALIASES,
    MECHANICS_OUTPUT_SCHEMA_VERSION,
    SHEAR_VALIDATED,
    SIMULATION_CAPABILITY,
    SLIP_VALIDATED,
)


METRIC_COLUMNS = (
    "timestamp_s",
    "trajectory_time_s",
    "trajectory_phase",
    "contact_flag",
    "approx_contact_area_m2",
    "estimated_axial_reaction_n",
    "estimated_transverse_reaction_x_n",
    "estimated_transverse_reaction_y_n",
    "estimated_transverse_reaction_z_n",
    "estimated_tangential_relative_velocity_x_m_s",
    "estimated_tangential_relative_velocity_y_m_s",
    "estimated_tangential_relative_velocity_z_m_s",
    "maximum_displacement_from_cad_rest_m",
    "maximum_displacement_from_equilibrated_baseline_m",
    "minimum_relative_tet_volume",
    "inverted_tet_count",
    "max_free_particle_speed_m_s",
    "equilibration_stable_frames",
    "contact_buffer_configured_capacity",
    "contact_buffer_observed_count",
    "contact_buffer_maximum_count_observed",
    "contact_buffer_saturation_flag",
    "contact_buffer_first_saturation_frame",
    "contact_buffer_first_saturation_substep",
    # Deprecated schema-v2 aliases.
    "contact_area_m2",
    "normal_force_n",
    "tangential_force_x_n",
    "tangential_force_y_n",
    "tangential_force_z_n",
    "slip_velocity_x_m_s",
    "slip_velocity_y_m_s",
    "slip_velocity_z_m_s",
    "maximum_displacement_m",
)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class MechanicalDataExporter:
    """Write bounded frame chunks and stream scalar metrics incrementally."""

    def __init__(self, output_dir: str | Path, resolved_config: dict[str, Any]):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size_frames = int(
            resolved_config.get("output", {}).get("chunk_size_frames", 300)
        )
        if self.chunk_size_frames <= 0:
            raise ValueError("chunk_size_frames must be positive")
        self.frames: dict[str, list[Any]] = {}
        self.chunk_frame_count = 0
        self.frame_count = 0
        self.chunk_index = 0
        self.chunks: list[dict[str, Any]] = []
        self.last_metric: dict[str, Any] | None = None
        self.minimum_relative_tet_volume_seen = float("inf")
        self.finalized = False

        _json_write(self.output_dir / "run_config.json", resolved_config)
        self.manifest = {
            "mechanics_output_schema_version": MECHANICS_OUTPUT_SCHEMA_VERSION,
            "contact_metric_model": CONTACT_METRIC_MODEL,
            "force_estimator_version": CONTACT_FORCE_ESTIMATOR_VERSION,
            "simulation_capability": SIMULATION_CAPABILITY,
            "shear_validated": SHEAR_VALIDATED,
            "slip_validated": SLIP_VALIDATED,
            "deprecated_aliases": DEPRECATED_ALIASES,
            "chunks": self.chunks,
            "total_frames": 0,
        }
        self.manifest.update(resolved_config.get("output_metadata", {}))
        self._write_manifest()

        self._metrics_stream = (self.output_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._metric_writer = csv.DictWriter(
            self._metrics_stream,
            fieldnames=METRIC_COLUMNS,
            extrasaction="ignore",
        )
        self._metric_writer.writeheader()
        self._metrics_stream.flush()

    def _write_manifest(self) -> None:
        self.manifest["total_frames"] = self.frame_count
        _json_write(self.output_dir / "frames_manifest.json", self.manifest)

    def append(self, frame: dict[str, Any], metric: dict[str, Any]) -> None:
        if self.finalized:
            raise RuntimeError("cannot append after exporter finalization")
        if self.frames and set(frame) != set(self.frames):
            missing = sorted(set(self.frames) - set(frame))
            added = sorted(set(frame) - set(self.frames))
            raise ValueError(
                f"frame fields changed within a run; missing={missing}, added={added}"
            )
        for name, value in frame.items():
            self.frames.setdefault(name, []).append(value)
        self.chunk_frame_count += 1
        self.frame_count += 1
        self.last_metric = dict(metric)
        if "minimum_relative_tet_volume" in metric:
            self.minimum_relative_tet_volume_seen = min(
                self.minimum_relative_tet_volume_seen,
                float(metric["minimum_relative_tet_volume"]),
            )
        self._metric_writer.writerow(metric)
        if self.chunk_frame_count >= self.chunk_size_frames:
            self.flush_chunk()

    def flush_chunk(self) -> None:
        if self.chunk_frame_count == 0:
            return
        arrays: dict[str, np.ndarray] = {}
        for name, values in self.frames.items():
            if name == "trajectory_phase":
                arrays[name] = np.asarray(values, dtype="U32")
            else:
                arrays[name] = np.asarray(values)
        filename = f"frames_{self.chunk_index:05d}.npz"
        np.savez_compressed(self.output_dir / filename, **arrays)
        start = self.frame_count - self.chunk_frame_count
        timestamps = arrays.get("timestamp_s")
        self.chunks.append(
            {
                "file": filename,
                "frame_start": start,
                "frame_count": self.chunk_frame_count,
                "timestamp_start_s": (
                    float(timestamps[0]) if timestamps is not None else None
                ),
                "timestamp_end_s": (
                    float(timestamps[-1]) if timestamps is not None else None
                ),
            }
        )
        self.chunk_index += 1
        self.frames.clear()
        self.chunk_frame_count = 0
        self._metrics_stream.flush()
        self._write_manifest()

    def finalize(self) -> None:
        if self.finalized:
            return
        self.flush_chunk()
        self._metrics_stream.flush()
        self._metrics_stream.close()
        self._write_manifest()
        self.finalized = True


def load_frame_chunks(output_dir: str | Path) -> dict[str, np.ndarray]:
    """Load and concatenate a completed chunked mechanics run."""

    root = Path(output_dir)
    manifest = json.loads((root / "frames_manifest.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[np.ndarray]] = {}
    for chunk in manifest["chunks"]:
        with np.load(root / chunk["file"]) as loaded:
            for name in loaded.files:
                grouped.setdefault(name, []).append(np.asarray(loaded[name]))
    return {name: np.concatenate(values, axis=0) for name, values in grouped.items()}


def export_failure_state(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
