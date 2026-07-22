"""Rendered-frame MP4 recording for the Newton OpenGL viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


class VideoRecorder:
    """Sample rendered GL frames at a fixed rate and stream them to MP4."""

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float,
        codec: str,
        quality: int,
        include_ui: bool,
        writer_factory: Callable[..., Any] | None = None,
    ):
        if fps <= 0.0:
            raise ValueError("video fps must be positive")
        if not 0 <= quality <= 10:
            raise ValueError("video quality must be between 0 and 10")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.codec = str(codec)
        self.quality = int(quality)
        self.include_ui = bool(include_ui)
        self.frame_period_s = 1.0 / self.fps
        self.next_frame_time_s = 0.0
        self.frame_count = 0
        self.closed = False
        self._writer = None
        self._writer_factory = writer_factory or self._default_writer_factory

    @staticmethod
    def _default_writer_factory(path: Path, **options):
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401 -- verifies the MP4 backend is installed.
        except ImportError as exc:
            raise RuntimeError(
                "MP4 recording requires imageio and imageio-ffmpeg. Run through uv with "
                "'--with imageio --with imageio-ffmpeg', or install the mechanics requirements."
            ) from exc
        return imageio.get_writer(path, format="FFMPEG", mode="I", **options)

    def capture(self, viewer, time_s: float) -> bool:
        if self.closed:
            return False
        if time_s + 1.0e-12 < self.next_frame_time_s:
            return False
        if not hasattr(viewer, "get_frame"):
            raise RuntimeError("video recording requires Newton's OpenGL viewer")
        rendered = viewer.get_frame(render_ui=self.include_ui)
        frame = np.asarray(rendered.numpy(), dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(
                f"viewer returned an invalid RGB frame shape: {frame.shape}"
            )
        if self._writer is None:
            self._writer = self._writer_factory(
                self.path,
                fps=self.fps,
                codec=self.codec,
                quality=self.quality,
                macro_block_size=2,
                pixelformat="yuv420p",
            )
        self._writer.append_data(np.ascontiguousarray(frame))
        self.frame_count += 1
        while self.next_frame_time_s <= time_s + 1.0e-12:
            self.next_frame_time_s += self.frame_period_s
        return True

    def close(self) -> None:
        if self.closed:
            return
        if self._writer is not None:
            self._writer.close()
        self.closed = True
