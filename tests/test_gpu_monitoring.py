from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp

from sim.mechanics.gpu_monitoring import (
    GpuSafetyMonitor,
    SAFETY_REASON_CONTACT_BUFFER,
    SAFETY_REASON_MINIMUM_J,
    SAFETY_REASON_NONFINITE,
)


def _state(points: np.ndarray):
    return SimpleNamespace(
        particle_q=wp.array(points, dtype=wp.vec3, device="cpu"),
        particle_qd=wp.zeros(len(points), dtype=wp.vec3, device="cpu"),
    )


def _monitor(contact_counts, *, fatal_on_contact_saturation=False):
    return GpuSafetyMonitor(
        device=wp.get_device("cpu"),
        tet_indices=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        rest_volumes=np.asarray([1.0 / 6.0]),
        particle_start=0,
        free_particle_indices=np.arange(4, dtype=np.int32),
        warning_minimum_j=0.8,
        stop_minimum_j=0.5,
        fatal_minimum_j=0.15,
        contact_counts=contact_counts,
        contact_body_index=0,
        contact_capacity=2,
        fatal_on_contact_saturation=fatal_on_contact_saturation,
    )


def test_gpu_monitor_reports_safe_and_stopped_tets_without_full_readback():
    contact_counts = wp.array([0], dtype=wp.int32, device="cpu")
    monitor = _monitor(contact_counts)
    safe = _state(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    monitor.reset_frame()
    monitor.reset_candidate()
    monitor.evaluate_candidate(safe, include_free_speed=True)
    snapshot = monitor.readback()
    assert np.isclose(snapshot.minimum_relative_j, 1.0)
    assert not snapshot.candidate_rejected
    assert not snapshot.fatal

    stopped = _state(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.4],
            ],
            dtype=np.float32,
        )
    )
    monitor.reset_frame()
    monitor.reset_candidate()
    monitor.evaluate_candidate(stopped)
    snapshot = monitor.readback()
    assert np.isclose(snapshot.minimum_relative_j, 0.4)
    assert snapshot.stop_tet_count == 1
    assert snapshot.candidate_rejected
    assert snapshot.reason_code == SAFETY_REASON_MINIMUM_J
    assert not snapshot.fatal


def test_gpu_monitor_latches_nonfinite_and_contact_saturation():
    contact_counts = wp.array([2], dtype=wp.int32, device="cpu")
    monitor = _monitor(contact_counts, fatal_on_contact_saturation=True)
    invalid = _state(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [np.nan, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    monitor.reset_frame()
    monitor.reset_candidate()
    monitor.evaluate_candidate(invalid)
    snapshot = monitor.readback()
    assert snapshot.fatal
    assert snapshot.candidate_rejected
    assert snapshot.nonfinite_tet_count == 1
    assert snapshot.contact_buffer_saturated
    assert snapshot.run_maximum_contact_count == 2
    # Non-finite tet state has higher diagnostic priority than buffer capacity.
    assert snapshot.reason_code == SAFETY_REASON_NONFINITE
    assert SAFETY_REASON_CONTACT_BUFFER > snapshot.reason_code
