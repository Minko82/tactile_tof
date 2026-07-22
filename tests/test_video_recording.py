import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim.mechanics.video import VideoRecorder


class _Frame:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class _Viewer:
    def __init__(self):
        self.calls = []

    def get_frame(self, render_ui=False):
        self.calls.append(render_ui)
        return _Frame(np.zeros((8, 12, 3), dtype=np.uint8))


class _Writer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


class VideoRecordingTests(unittest.TestCase):
    def test_fixed_rate_sampling_and_close(self):
        writer = _Writer()
        options = {}

        def factory(path, **kwargs):
            options.update(path=path, **kwargs)
            return writer

        with tempfile.TemporaryDirectory() as temporary:
            recorder = VideoRecorder(
                Path(temporary) / "experiment.mp4",
                fps=30.0,
                codec="libx264",
                quality=8,
                include_ui=False,
                writer_factory=factory,
            )
            viewer = _Viewer()
            self.assertTrue(recorder.capture(viewer, 0.0))
            self.assertFalse(recorder.capture(viewer, 0.01))
            self.assertTrue(recorder.capture(viewer, 1.0 / 30.0))
            recorder.close()

        self.assertEqual(len(writer.frames), 2)
        self.assertEqual(viewer.calls, [False, False])
        self.assertTrue(writer.closed)
        self.assertEqual(options["fps"], 30.0)
        self.assertEqual(options["codec"], "libx264")


if __name__ == "__main__":
    unittest.main()
