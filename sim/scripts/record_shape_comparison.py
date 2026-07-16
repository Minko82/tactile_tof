"""Record a synchronized real-versus-simulation ToF comparison as MP4.

The real sensor has already been recorded, so "simultaneous" playback means that
each real frame and RTX frame with the same reference timestamp are rendered in
the same video frame.  The plots reuse the styling and 8x8 helpers from
``visualizer.py``.

Example:

    py sim/scripts/record_shape_comparison.py --direction descending
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import visualizer


GRID = visualizer.GRID
ZONES = visualizer.ZONES
ZONE_COLUMNS = visualizer.ZONE_COLUMNS
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PROFILE = REPO_ROOT / "sim" / "config" / "shape_experiments" / "cup.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "sim" / "output" / "shape_experiments"
DEFAULT_SIM_SCRIPT = SCRIPT_DIR / "run_vl53l8cx_isaac_tof.py"
ZONE_TRANSFORMS = (
    "identity",
    "mirror",
    "rot90",
    "mirror_rot90",
    "rot180",
    "mirror_rot180",
    "rot270",
    "mirror_rot270",
)


@dataclass(frozen=True)
class ZoneFrame:
    timestamp: str
    zones_mm: tuple[int, ...]
    frame_index: int | None = None
    elapsed_s: float | None = None
    tcp_z_m: float | None = None
    sensor_z_m: float | None = None


@dataclass(frozen=True)
class AlignedFrame:
    real: ZoneFrame
    sim: ZoneFrame


@dataclass(frozen=True)
class ComparisonInputs:
    experiment_name: str
    real_csv: Path
    sim_csv: Path
    zone_transform: str


def _optional_number(row: dict[str, str], column: str, number_type: type[int] | type[float]) -> int | float | None:
    value = str(row.get(column, "") or "").strip()
    if not value:
        return None
    return number_type(float(value))


def load_zone_frames(
    path: str | Path,
    *,
    sim_distance_mode: str = "raw",
    zone_transform: str = "identity",
) -> list[ZoneFrame]:
    """Load either a real ToF flat CSV or a shape-replay ``sim_flat.csv``."""

    path = Path(path)
    if sim_distance_mode not in visualizer.MODE_PREFIXES:
        raise ValueError(f"unsupported simulation distance mode {sim_distance_mode!r}")
    prefix = visualizer.MODE_PREFIXES[sim_distance_mode]
    zone_columns = [f"{prefix}_{index:02d}" for index in range(ZONES)]
    frames: list[ZoneFrame] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        timestamp_column = "reference_timestamp" if "reference_timestamp" in fieldnames else "timestamp"
        if timestamp_column not in fieldnames:
            raise ValueError(f"{path} is missing a timestamp or reference_timestamp column")
        missing = [column for column in zone_columns if column not in fieldnames]
        if missing:
            raise ValueError(
                f"{path} does not provide --sim-distance-mode {sim_distance_mode}; "
                f"first missing column: {missing[0]}"
            )

        for row_number, row in enumerate(reader, start=2):
            timestamp = str(row.get(timestamp_column, "") or "").strip()
            if not timestamp:
                raise ValueError(f"{path}:{row_number} has an empty timestamp")
            try:
                zones = tuple(int(float(str(row.get(column, "0") or "0"))) for column in zone_columns)
            except ValueError as exc:
                raise ValueError(f"{path}:{row_number} has a non-numeric zone value") from exc
            frames.append(
                ZoneFrame(
                    timestamp=timestamp,
                    zones_mm=transform_zones(zones, zone_transform),
                    frame_index=_optional_number(row, "frame_index", int),
                    elapsed_s=_optional_number(row, "elapsed_s", float),
                    tcp_z_m=_optional_number(row, "tcp_z_m", float),
                    sensor_z_m=_optional_number(row, "sensor_z_m", float),
                )
            )

    if not frames:
        raise ValueError(f"{path} contains no frames")
    return frames


def transform_zones(values: Sequence[int], transform: str) -> tuple[int, ...]:
    if transform not in ZONE_TRANSFORMS:
        raise ValueError(f"unsupported zone transform {transform!r}")
    if len(values) != ZONES:
        raise ValueError(f"expected {ZONES} zones, got {len(values)}")

    matrix = [list(values[start : start + GRID]) for start in range(0, ZONES, GRID)]
    if transform.startswith("mirror"):
        matrix = [list(reversed(row)) for row in matrix]
    rotations = {
        "identity": 0,
        "mirror": 0,
        "rot90": 1,
        "mirror_rot90": 1,
        "rot180": 2,
        "mirror_rot180": 2,
        "rot270": 3,
        "mirror_rot270": 3,
    }[transform]
    for _ in range(rotations):
        matrix = [list(row) for row in zip(*reversed(matrix))]
    return tuple(value for row in matrix for value in row)


def align_frames(
    real_frames: Sequence[ZoneFrame],
    sim_frames: Sequence[ZoneFrame],
    real_zone_transform: str = "identity",
) -> list[AlignedFrame]:
    if real_zone_transform != "identity":
        raise ValueError("real zone_transform must be applied exactly once during ingestion")
    real_by_timestamp: dict[str, ZoneFrame] = {}
    for frame in real_frames:
        if frame.timestamp in real_by_timestamp:
            raise ValueError(f"real CSV contains duplicate timestamp {frame.timestamp}")
        real_by_timestamp[frame.timestamp] = frame

    aligned: list[AlignedFrame] = []
    missing: list[str] = []
    for sim in sim_frames:
        real = real_by_timestamp.get(sim.timestamp)
        if real is None:
            missing.append(sim.timestamp)
            continue
        aligned.append(AlignedFrame(real=real, sim=sim))

    if missing:
        raise ValueError(
            f"{len(missing)} simulated timestamps have no exact real frame; first missing: {missing[0]}"
        )
    if not aligned:
        raise ValueError("the real and simulation CSVs contain no matching timestamps")
    return aligned


def frame_metrics(real: Sequence[int], sim: Sequence[int]) -> dict[str, float | int]:
    paired_errors: list[int] = []
    real_valid = 0
    sim_valid = 0
    both_invalid = 0
    no_return_union = 0
    sim_only_returns = 0
    real_only_returns = 0

    for real_value, sim_value in zip(real, sim):
        real_has_return = real_value > 0
        sim_has_return = sim_value > 0
        real_valid += int(real_has_return)
        sim_valid += int(sim_has_return)
        if real_has_return and sim_has_return:
            paired_errors.append(sim_value - real_value)
        elif sim_has_return:
            sim_only_returns += 1
        elif real_has_return:
            real_only_returns += 1
        if not real_has_return and not sim_has_return:
            both_invalid += 1
        if not real_has_return or not sim_has_return:
            no_return_union += 1

    mae = sum(abs(error) for error in paired_errors) / len(paired_errors) if paired_errors else 0.0
    bias = sum(paired_errors) / len(paired_errors) if paired_errors else 0.0
    no_return_iou = both_invalid / no_return_union if no_return_union else 1.0
    return {
        "real_valid": real_valid,
        "sim_valid": sim_valid,
        "paired_valid": len(paired_errors),
        "mae_mm": mae,
        "bias_mm": bias,
        "no_return_iou": no_return_iou,
        "sim_only_returns": sim_only_returns,
        "real_only_returns": real_only_returns,
    }


def load_comparison_inputs(
    profile_path: str | Path,
    direction: str,
    experiment_output_root: str | Path,
) -> ComparisonInputs:
    profile_path = Path(profile_path).resolve()
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    try:
        experiment_name = str(profile["name"])
        direction_config = profile["directions"][direction]
        real_csv = (profile_path.parent / str(direction_config["tof_csv"])).resolve()
        zone_transform = str(profile.get("zone_transform", "identity"))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{profile_path} does not define direction {direction!r}") from exc
    if zone_transform not in ZONE_TRANSFORMS:
        raise ValueError(f"{profile_path} has unsupported zone_transform {zone_transform!r}")
    sim_csv = Path(experiment_output_root).resolve() / experiment_name / direction / "sim_flat.csv"
    return ComparisonInputs(experiment_name, real_csv, sim_csv, zone_transform)


def run_shape_experiment(args: argparse.Namespace) -> None:
    isaac_python = Path(args.isaac_python)
    if not isaac_python.is_file():
        raise SystemExit(f"Isaac Python was not found: {isaac_python}")
    command = [
        str(isaac_python),
        str(args.sim_script.resolve()),
        "--headless",
        "--quiet_arrays",
        "--no_debug_draw",
        "--scene",
        "shape-replay",
        "--experiment-profile",
        str(args.experiment_profile.resolve()),
        "--experiment-direction",
        args.direction,
        "--experiment-output-dir",
        str(args.experiment_output_root.resolve()),
        "--distance-calibration-mode",
        args.distance_calibration_mode,
    ]
    print("Running descending/ascending Isaac replay before video rendering:")
    print(subprocess.list2cmdline(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _style_heat_axis(ax: Any, title: str) -> None:
    ax.set_facecolor(visualizer.BACKGROUND)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(GRID - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=visualizer.TEXT_COLOR, fontsize=11, fontweight="bold", pad=7)
    for line in range(GRID + 1):
        coordinate = line - 0.5
        ax.axvline(coordinate, color=visualizer.GRID_COLOR, linewidth=0.6)
        ax.axhline(coordinate, color=visualizer.GRID_COLOR, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(visualizer.GRID_COLOR)


def _difference_rgb(real: Any, sim: Any, error_max_mm: float, np: Any, plt: Any) -> Any:
    output = np.zeros((GRID, GRID, 3), dtype=float)
    empty_rgb = np.asarray(visualizer.EMPTY_COLOR[:3])
    output[:, :] = empty_rgb
    for zone, (real_value, sim_value) in enumerate(zip(real, sim)):
        row, col = divmod(zone, GRID)
        if real_value > 0 and sim_value > 0:
            normalized = min(abs(float(sim_value) - float(real_value)) / max(error_max_mm, 1.0), 1.0)
            output[row, col] = plt.cm.magma(normalized)[:3]
        elif sim_value > 0:
            output[row, col] = (1.0, 0.20, 0.75)  # Sim returned; real did not.
        elif real_value > 0:
            output[row, col] = (0.15, 0.65, 1.0)  # Real returned; sim did not.
    return output


def record_video(
    aligned: Sequence[AlignedFrame],
    output_path: str | Path,
    *,
    fps: float,
    max_mm: float,
    error_max_mm: float,
    dpi: int,
    title: str,
    ffmpeg_path: str | Path | None = None,
) -> None:
    visualizer.load_plotting_dependencies()
    animation = visualizer.animation
    plt = visualizer.plt
    np = visualizer.np
    GridSpec = visualizer.GridSpec
    if ffmpeg_path is not None:
        plt.rcParams["animation.ffmpeg_path"] = str(Path(ffmpeg_path))
    if not animation.writers.is_available("ffmpeg"):
        raise SystemExit("FFmpeg is required to save MP4. Install it or pass --ffmpeg-path to ffmpeg.exe.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), facecolor=visualizer.BACKGROUND)
    if hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title("Real vs Isaac ToF comparison")
    grid = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[3.0, 1.25],
        width_ratios=[1.0, 1.0, 0.82],
        hspace=0.12,
        wspace=0.08,
        left=0.035,
        right=0.985,
        top=0.89,
        bottom=0.055,
    )
    real_3d = fig.add_subplot(grid[0, 0], projection="3d", facecolor=visualizer.BACKGROUND)
    sim_3d = fig.add_subplot(grid[0, 1], projection="3d", facecolor=visualizer.BACKGROUND)
    error_ax = fig.add_subplot(grid[0, 2], facecolor=visualizer.BACKGROUND)
    real_heat_ax = fig.add_subplot(grid[1, 0], facecolor=visualizer.BACKGROUND)
    sim_heat_ax = fig.add_subplot(grid[1, 1], facecolor=visualizer.BACKGROUND)
    metrics_ax = fig.add_subplot(grid[1, 2], facecolor=visualizer.BACKGROUND)

    _style_heat_axis(real_heat_ax, "REAL 8x8")
    _style_heat_axis(sim_heat_ax, "SIMULATION 8x8")
    _style_heat_axis(error_ax, f"ABSOLUTE ERROR (0-{error_max_mm:g} mm)")
    metrics_ax.set_axis_off()
    metrics_text = metrics_ax.text(
        0.0,
        0.98,
        "",
        transform=metrics_ax.transAxes,
        ha="left",
        va="top",
        color="#d7d7f5",
        fontsize=9.5,
        family="monospace",
        linespacing=1.25,
    )
    metrics_ax.text(0.0, 0.12, "■ sim-only return", color="#ff33bf", fontsize=8.5, transform=metrics_ax.transAxes)
    metrics_ax.text(0.0, 0.01, "■ real-only return", color="#26a6ff", fontsize=8.5, transform=metrics_ax.transAxes)

    blank = np.zeros((GRID, GRID, 3), dtype=float)
    real_heat_image = real_heat_ax.imshow(blank, origin="upper", interpolation="nearest")
    sim_heat_image = sim_heat_ax.imshow(blank, origin="upper", interpolation="nearest")
    error_image = error_ax.imshow(blank, origin="upper", interpolation="nearest")
    fig.suptitle(title, color="#8f8fe8", fontsize=18, fontweight="bold", y=0.975)
    frame_text = fig.text(0.5, 0.925, "", color="#c5c5e8", fontsize=11, ha="center", va="center")

    cols = np.arange(GRID)
    rows = np.arange(GRID)
    xpos, ypos = np.meshgrid(cols, rows)
    xpos = xpos.flatten().astype(float)
    ypos = ypos.flatten().astype(float)
    zpos = np.zeros(ZONES)
    dx = dy = np.full(ZONES, 0.72)

    def draw_3d(ax: Any, values: Any, panel_title: str) -> None:
        ax.cla()
        heights = np.where(values > 0, np.minimum(values, max_mm), 0.0)
        ax.bar3d(
            xpos,
            ypos,
            zpos,
            dx,
            dy,
            np.maximum(heights, 1.0),
            color=visualizer.colors_for_values(values, max_mm),
            shade=True,
            zsort="average",
        )
        valid = values[values > 0]
        mean_mm = float(valid.mean()) if valid.size else 0.0
        visualizer.style_3d_axis(ax, max_mm, f"{panel_title} | valid={valid.size} | mean={mean_mm:.1f} mm")

    def update(frame_number: int) -> None:
        pair = aligned[frame_number]
        real = np.asarray(pair.real.zones_mm, dtype=float)
        sim = np.asarray(pair.sim.zones_mm, dtype=float)
        metrics = frame_metrics(pair.real.zones_mm, pair.sim.zones_mm)
        draw_3d(real_3d, real, "REAL VL53L5CX")
        draw_3d(sim_3d, sim, "ISAAC RTX SIM")
        real_heat_image.set_data(visualizer.heatmap_rgb(real, max_mm))
        sim_heat_image.set_data(visualizer.heatmap_rgb(sim, max_mm))
        error_image.set_data(_difference_rgb(real, sim, error_max_mm, np, plt))

        elapsed = pair.sim.elapsed_s if pair.sim.elapsed_s is not None else frame_number / fps
        tcp_text = f" | TCP Z={pair.sim.tcp_z_m * 1000.0:.1f} mm" if pair.sim.tcp_z_m is not None else ""
        frame_text.set_text(
            f"frame {frame_number + 1}/{len(aligned)} | t={elapsed:.3f} s{tcp_text} | {pair.sim.timestamp}"
        )
        metrics_text.set_text(
            "CURRENT FRAME\n"
            f"real valid      {metrics['real_valid']:>3}/64\n"
            f"sim valid       {metrics['sim_valid']:>3}/64\n"
            f"paired valid    {metrics['paired_valid']:>3}/64\n"
            f"distance MAE    {metrics['mae_mm']:>7.1f} mm\n"
            f"distance bias   {metrics['bias_mm']:>+7.1f} mm\n"
            f"no-return IoU   {metrics['no_return_iou']:>7.1%}\n"
            f"sim-only zones  {metrics['sim_only_returns']:>3}\n"
            f"real-only zones {metrics['real_only_returns']:>3}"
        )

    interval_ms = 1000.0 / fps
    video = animation.FuncAnimation(fig, update, frames=len(aligned), interval=interval_ms, blit=False)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": title, "artist": "tactile_tof comparison recorder"},
    )

    def progress(current_frame: int, total_frames: int) -> None:
        completed = current_frame + 1
        if completed == 1 or completed == total_frames or completed % 25 == 0:
            print(f"Rendering video frame {completed}/{total_frames}")

    try:
        video.save(output_path, writer=writer, dpi=dpi, progress_callback=progress)
    finally:
        plt.close(fig)
    print(f"Saved synchronized comparison video: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render synchronized real and simulated 8x8 ToF frames to an MP4 comparison video."
    )
    parser.add_argument("--experiment-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--direction", choices=("ascending", "descending"), default="descending")
    parser.add_argument("--experiment-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--real-csv", type=Path, default=None, help="Override the real ToF CSV from the profile.")
    parser.add_argument("--sim-csv", type=Path, default=None, help="Override the generated sim_flat.csv path.")
    parser.add_argument(
        "--sim-distance-mode",
        choices=("raw", "projected", "comparison"),
        default="raw",
        help="Simulation distance layer rendered in the comparison video.",
    )
    parser.add_argument(
        "--distance-calibration-mode",
        choices=("off", "strict", "diagnostic"),
        default="strict",
        help="Calibration mode passed to --rerun-simulation.",
    )
    parser.add_argument("--output-mp4", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-mm", type=float, default=500.0, help="Shared distance scale for real and simulation.")
    parser.add_argument("--error-max-mm", type=float, default=100.0, help="Error value mapped to the top error color.")
    parser.add_argument("--dpi", type=int, default=120, help="120 DPI with the 16x9 figure produces 1920x1080 video.")
    parser.add_argument("--frame-limit", type=int, default=0, help="Render only the first N aligned frames; 0 renders all.")
    parser.add_argument("--ffmpeg-path", type=Path, default=None, help="Optional explicit path to ffmpeg.exe.")
    parser.add_argument(
        "--rerun-simulation",
        action="store_true",
        help="Run the Isaac shape replay first, then render the newly generated CSV.",
    )
    parser.add_argument("--isaac-python", type=Path, default=Path(r"C:\isaacsim\python.bat"))
    parser.add_argument("--sim-script", type=Path, default=DEFAULT_SIM_SCRIPT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than zero")
    if args.max_mm <= 0 or args.error_max_mm <= 0:
        raise SystemExit("--max-mm and --error-max-mm must be greater than zero")
    if args.frame_limit < 0:
        raise SystemExit("--frame-limit cannot be negative")

    if args.rerun_simulation:
        run_shape_experiment(args)

    inputs = load_comparison_inputs(args.experiment_profile, args.direction, args.experiment_output_root)
    real_csv = args.real_csv.resolve() if args.real_csv is not None else inputs.real_csv
    sim_csv = args.sim_csv.resolve() if args.sim_csv is not None else inputs.sim_csv
    if not real_csv.is_file():
        raise SystemExit(f"Real ToF CSV was not found: {real_csv}")
    if not sim_csv.is_file():
        raise SystemExit(f"Simulation CSV was not found: {sim_csv}. Run the shape replay or use --rerun-simulation.")

    real_frames = load_zone_frames(real_csv, zone_transform=inputs.zone_transform)
    sim_frames = load_zone_frames(sim_csv, sim_distance_mode=args.sim_distance_mode)
    aligned = align_frames(real_frames, sim_frames)
    if args.frame_limit:
        aligned = aligned[: args.frame_limit]
    output_path = (
        args.output_mp4.resolve()
        if args.output_mp4 is not None
        else sim_csv.parent / "real_vs_sim.mp4"
    )
    print(f"Aligned {len(aligned)} real/simulation frames by exact reference timestamp.")
    record_video(
        aligned,
        output_path,
        fps=args.fps,
        max_mm=args.max_mm,
        error_max_mm=args.error_max_mm,
        dpi=args.dpi,
        title=(
            f"{inputs.experiment_name.upper()} {args.direction.upper()} — REAL vs SIMULATION "
            f"[{args.sim_distance_mode.upper()}]"
        ),
        ffmpeg_path=args.ffmpeg_path,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
