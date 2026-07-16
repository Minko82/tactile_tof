"""Live VL53L8CX simulator visualizer.

The visualizer is intentionally source-agnostic. For Isaac Sim, use:

    python sim/scripts/visualizer.py --source csv-tail --input_csv sim/output/live_readings.csv

It expects flat CSV rows with zone_00 through zone_63 columns.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


GRID = 8
ZONES = GRID * GRID
ZONE_COLUMNS = [f"zone_{index:02d}" for index in range(ZONES)]
MODE_PREFIXES = {"raw": "zone", "projected": "projected_zone", "comparison": "comparison_zone"}
BACKGROUND = "#07071a"
GRID_COLOR = "#1a1a3a"
TEXT_COLOR = "#6666cc"
EMPTY_COLOR = (0.05, 0.05, 0.16, 1.0)
animation: Any = None
plt: Any = None
np: Any = None
GridSpec: Any = None


def load_plotting_dependencies() -> None:
    global animation, plt, np, GridSpec
    try:
        import matplotlib.animation as animation_module
        import matplotlib.pyplot as plt_module
        import numpy as np_module
        from matplotlib.gridspec import GridSpec as grid_spec_class
        from mpl_toolkits.mplot3d import Axes3D as _axes3d  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing dependency: install matplotlib and numpy, then run the visualizer again.") from exc

    animation = animation_module
    plt = plt_module
    np = np_module
    GridSpec = grid_spec_class


class DemoSource:
    def __init__(self) -> None:
        self._t = 0.0

    def poll(self) -> np.ndarray:
        self._t += 0.04
        values = np.zeros(ZONES, dtype=float)
        for zone in range(ZONES):
            row, col = divmod(zone, GRID)
            radius = math.sqrt((col - 3.5) ** 2 + (row - 3.5) ** 2)
            wave = math.sin(self._t - radius * 0.75) * 0.5 + 0.5
            values[zone] = round(250 + wave * 1500)
        return values


class CsvTailSource:
    def __init__(self, path: str | Path, sim_distance_mode: str = "raw") -> None:
        self.path = Path(path)
        if sim_distance_mode not in MODE_PREFIXES:
            raise ValueError(f"unsupported simulation distance mode {sim_distance_mode!r}")
        self.sim_distance_mode = sim_distance_mode
        prefix = MODE_PREFIXES[sim_distance_mode]
        self.zone_columns = [f"{prefix}_{index:02d}" for index in range(ZONES)]
        self._handle: Any | None = None
        self._header: list[str] | None = None
        self._latest: np.ndarray | None = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._header = None

    def poll(self) -> np.ndarray | None:
        if not self.path.exists():
            return self._latest

        if self._handle is not None and self.path.stat().st_size < self._handle.tell():
            self.close()

        if self._handle is None:
            self._handle = self.path.open("r", encoding="utf-8", newline="")
            header_line = self._handle.readline()
            if not header_line:
                return self._latest
            self._header = next(csv.reader([header_line]))
            missing = [column for column in self.zone_columns if column not in self._header]
            if missing:
                raise ValueError(
                    f"{self.path} does not provide --sim-distance-mode {self.sim_distance_mode}; "
                    f"first missing column: {missing[0]}"
                )

        assert self._handle is not None
        assert self._header is not None

        latest_row: list[str] | None = None
        last_complete_position = self._handle.tell()
        while True:
            position = self._handle.tell()
            line = self._handle.readline()
            if not line:
                self._handle.seek(last_complete_position)
                break
            if not line.endswith("\n"):
                self._handle.seek(position)
                break
            rows = list(csv.reader([line]))
            if rows:
                latest_row = rows[0]
                last_complete_position = self._handle.tell()

        if latest_row is None:
            return self._latest

        row = dict(zip(self._header, latest_row))
        values = np.zeros(ZONES, dtype=float)
        for index, column in enumerate(self.zone_columns):
            try:
                values[index] = float(row.get(column, 0) or 0)
            except ValueError:
                values[index] = 0
        self._latest = values
        return self._latest


def create_source(args: argparse.Namespace) -> DemoSource | CsvTailSource:
    if args.source == "demo":
        return DemoSource()
    if args.source == "csv-tail":
        if args.input_csv is None:
            raise SystemExit("--input_csv is required with --source csv-tail")
        return CsvTailSource(args.input_csv, args.sim_distance_mode)
    raise SystemExit("--source serial is not implemented in this simulator visualizer")


def colors_for_values(values: np.ndarray, max_mm: float) -> list[tuple[float, float, float, float]]:
    cmap = plt.cm.viridis
    colors = []
    for value in values:
        if value <= 0:
            colors.append(tuple(EMPTY_COLOR))
        else:
            colors.append(tuple(cmap(min(float(value) / max(max_mm, 1.0), 1.0))))
    return colors


def heatmap_rgb(values: np.ndarray, max_mm: float) -> np.ndarray:
    cmap = plt.cm.viridis
    rgba = np.zeros((GRID, GRID, 4), dtype=float)
    for zone, value in enumerate(values):
        row, col = divmod(zone, GRID)
        if value <= 0:
            rgba[row, col] = EMPTY_COLOR
        else:
            rgba[row, col] = cmap(min(float(value) / max(max_mm, 1.0), 1.0))
    return rgba[:, :, :3]


def style_3d_axis(ax: Any, max_mm: float, title: str) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(-0.5, GRID - 0.5)
    ax.set_zlim(0, max_mm)
    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))
    ax.tick_params(colors="#333355", labelsize=7)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax.zaxis.pane.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4)
    ax.set_zlabel("distance (mm)", color="#333355", fontsize=8, labelpad=6)
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.view_init(elev=28, azim=-60)


def build_figure() -> tuple[Any, Any, Any, Any]:
    fig = plt.figure(figsize=(9, 8.5), facecolor=BACKGROUND)
    fig.canvas.manager.set_window_title("VL53L8CX Isaac ToF Visualizer")
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08, left=0.05, right=0.98, top=0.96, bottom=0.08)
    ax_3d = fig.add_subplot(gs[0], projection="3d", facecolor=BACKGROUND)
    ax_heat = fig.add_subplot(gs[1], facecolor=BACKGROUND)
    ax_heat.set_aspect("equal")
    heatmap_im = ax_heat.imshow(
        np.zeros((GRID, GRID, 3)),
        origin="upper",
        extent=[-0.5, 7.5, 7.5, -0.5],
        interpolation="nearest",
    )
    for x in np.arange(-0.5, GRID, 1):
        ax_heat.axvline(x, color=GRID_COLOR, linewidth=0.8)
    for y in np.arange(-0.5, GRID, 1):
        ax_heat.axhline(y, color=GRID_COLOR, linewidth=0.8)
    ax_heat.set_xlim(-0.5, 7.5)
    ax_heat.set_ylim(7.5, -0.5)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("8x8 zones", color="#444466", fontsize=9, labelpad=4)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    status_text = ax_heat.text(3.5, -1.1, "waiting for frames", ha="center", va="center", fontsize=10, color=TEXT_COLOR)
    return fig, ax_3d, heatmap_im, status_text


def run_visualizer(args: argparse.Namespace) -> None:
    load_plotting_dependencies()
    source = create_source(args)
    fig, ax_3d, heatmap_im, status_text = build_figure()

    cols = np.arange(GRID)
    rows = np.arange(GRID)
    xpos, ypos = np.meshgrid(cols, rows)
    xpos = xpos.flatten().astype(float)
    ypos = ypos.flatten().astype(float)
    zpos = np.zeros(ZONES)
    dx = dy = np.full(ZONES, 0.72)

    latest = np.zeros(ZONES, dtype=float)

    def update(_frame_num: int) -> None:
        nonlocal latest
        polled = source.poll()
        if polled is not None:
            latest = polled

        valid = latest[latest > 0]
        valid_zones = int(valid.size)
        mean_mm = float(valid.mean()) if valid_zones else 0.0
        min_mm = int(valid.min()) if valid_zones else 0
        max_seen = int(valid.max()) if valid_zones else 0
        title = (
            f"VL53L8CX Isaac ToF [{args.sim_distance_mode}] | valid={valid_zones} | "
            f"mean={mean_mm:.1f} mm | min={min_mm} | max={max_seen}"
        )

        ax_3d.cla()
        heights = np.where(latest > 0, np.minimum(latest, args.max_mm), 0.0)
        ax_3d.bar3d(
            xpos,
            ypos,
            zpos,
            dx,
            dy,
            np.maximum(heights, 1.0),
            color=colors_for_values(latest, args.max_mm),
            shade=True,
            zsort="average",
        )
        style_3d_axis(ax_3d, args.max_mm, title)
        heatmap_im.set_data(heatmap_rgb(latest, args.max_mm))
        status_text.set_text(f"{valid_zones} valid zones")

    ani = animation.FuncAnimation(fig, update, interval=args.interval_ms, cache_frame_data=False)
    try:
        plt.show()
    finally:
        _ = ani
        if hasattr(source, "close"):
            source.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live visualizer for VL53L8CX simulator flat CSV output.")
    parser.add_argument("--source", choices=("csv-tail", "demo", "serial"), default="csv-tail")
    parser.add_argument("--input_csv", type=Path, default=None, help="CSV file to tail when --source csv-tail is used.")
    parser.add_argument("--max_mm", type=float, default=4000.0, help="Distance value mapped to the top of the plot.")
    parser.add_argument("--interval_ms", type=int, default=80, help="Matplotlib refresh interval.")
    parser.add_argument(
        "--sim-distance-mode",
        choices=("raw", "projected", "comparison"),
        default="raw",
        help="Distance columns to visualize from a shape-replay v2 CSV.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_visualizer(args)


if __name__ == "__main__":
    main()
