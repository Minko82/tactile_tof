"""GPU-resident safety reductions shared by both Newton mechanics runners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


SAFETY_REASON_NONE = 2_147_483_647
SAFETY_REASON_NONFINITE = 0
SAFETY_REASON_MINIMUM_J = 1
SAFETY_REASON_COMMANDED_INDENTATION = 2
SAFETY_REASON_CONTACT_BUFFER = 3
SAFETY_REASON_ESTIMATED_FORCE = 4

SAFETY_REASON_NAMES = {
    SAFETY_REASON_NONE: "",
    SAFETY_REASON_NONFINITE: "nonfinite_tet_state",
    SAFETY_REASON_MINIMUM_J: "minimum_relative_tet_volume",
    SAFETY_REASON_COMMANDED_INDENTATION: "commanded_indentation_limit",
    SAFETY_REASON_CONTACT_BUFFER: "contact_buffer_saturation",
    SAFETY_REASON_ESTIMATED_FORCE: "estimated_force_limit",
}

# Float metric slots.
_FRAME_MIN_J = 0
_CANDIDATE_MIN_J = 1
_MAX_FREE_SPEED = 2
_FLOAT_METRIC_COUNT = 3

# Integer metric slots.
_FRAME_FIRST_BAD = 0
_FRAME_NONFINITE_MAX = 1
_FRAME_INVERTED_MAX = 2
_FRAME_WARNING_MAX = 3
_FRAME_STOP_MAX = 4
_FRAME_FATAL = 5
_CONTACT_CURRENT = 6
_CONTACT_FRAME_MAX = 7
_CONTACT_RUN_MAX = 8
_CONTACT_SATURATED = 9
_CANDIDATE_FIRST_BAD = 10
_CANDIDATE_NONFINITE = 11
_CANDIDATE_INVERTED = 12
_CANDIDATE_WARNING = 13
_CANDIDATE_STOP = 14
_CANDIDATE_REJECT = 15
_CANDIDATE_REASON = 16
_INT_METRIC_COUNT = 17


@wp.kernel
def _reset_frame_metrics(
    tet_count: int,
    float_metrics: wp.array[float],
    int_metrics: wp.array[int],
):
    float_metrics[0] = 1.0e30
    float_metrics[1] = 1.0e30
    float_metrics[2] = 0.0
    int_metrics[0] = tet_count
    int_metrics[1] = 0
    int_metrics[2] = 0
    int_metrics[3] = 0
    int_metrics[4] = 0
    int_metrics[5] = 0
    int_metrics[6] = 0
    int_metrics[7] = 0


@wp.kernel
def _reset_candidate_metrics(
    tet_count: int,
    float_metrics: wp.array[float],
    int_metrics: wp.array[int],
):
    float_metrics[1] = 1.0e30
    int_metrics[10] = tet_count
    int_metrics[11] = 0
    int_metrics[12] = 0
    int_metrics[13] = 0
    int_metrics[14] = 0
    int_metrics[15] = 0
    int_metrics[16] = 2_147_483_647


@wp.kernel
def _tet_safety_reduction(
    particle_q: wp.array[wp.vec3],
    particle_start: int,
    tet_indices: wp.array[int],
    rest_volumes: wp.array[float],
    warning_minimum_j: float,
    stop_minimum_j: float,
    fatal_minimum_j: float,
    float_metrics: wp.array[float],
    int_metrics: wp.array[int],
):
    tid = wp.tid()
    base = tid * 4
    i0 = particle_start + tet_indices[base]
    i1 = particle_start + tet_indices[base + 1]
    i2 = particle_start + tet_indices[base + 2]
    i3 = particle_start + tet_indices[base + 3]
    p0 = particle_q[i0]
    volume = (
        wp.dot(
            particle_q[i1] - p0,
            wp.cross(particle_q[i2] - p0, particle_q[i3] - p0),
        )
        / 6.0
    )
    relative_j = volume / rest_volumes[tid]

    if not wp.isfinite(relative_j):
        wp.atomic_min(float_metrics, 0, -1.0e30)
        wp.atomic_min(float_metrics, 1, -1.0e30)
        wp.atomic_add(int_metrics, 11, 1)
        wp.atomic_min(int_metrics, 10, tid)
        wp.atomic_min(int_metrics, 16, SAFETY_REASON_NONFINITE)
        return

    wp.atomic_min(float_metrics, 0, relative_j)
    wp.atomic_min(float_metrics, 1, relative_j)
    if relative_j <= 0.0:
        wp.atomic_add(int_metrics, 12, 1)
    if relative_j < warning_minimum_j:
        wp.atomic_add(int_metrics, 13, 1)
        wp.atomic_min(int_metrics, 10, tid)
    if relative_j < stop_minimum_j:
        wp.atomic_add(int_metrics, 14, 1)
        wp.atomic_min(int_metrics, 10, tid)
        wp.atomic_min(int_metrics, 16, SAFETY_REASON_MINIMUM_J)
    if relative_j < fatal_minimum_j:
        int_metrics[5] = 1


@wp.kernel
def _finish_tet_reduction(int_metrics: wp.array[int]):
    nonfinite = int_metrics[11]
    inverted = int_metrics[12]
    warning = int_metrics[13]
    stopped = int_metrics[14]
    if nonfinite > int_metrics[1]:
        int_metrics[1] = nonfinite
    if inverted > int_metrics[2]:
        int_metrics[2] = inverted
    if warning > int_metrics[3]:
        int_metrics[3] = warning
    if stopped > int_metrics[4]:
        int_metrics[4] = stopped
    if int_metrics[10] < int_metrics[0]:
        int_metrics[0] = int_metrics[10]
    if nonfinite > 0:
        int_metrics[5] = 1
        int_metrics[15] = 1
    if stopped > 0:
        int_metrics[15] = 1


@wp.kernel
def _free_particle_speed_reduction(
    particle_qd: wp.array[wp.vec3],
    particle_start: int,
    free_particle_indices: wp.array[int],
    float_metrics: wp.array[float],
):
    tid = wp.tid()
    speed = wp.length(particle_qd[particle_start + free_particle_indices[tid]])
    if wp.isfinite(speed):
        wp.atomic_max(float_metrics, 2, speed)
    else:
        wp.atomic_max(float_metrics, 2, 1.0e30)


@wp.kernel
def _contact_buffer_reduction(
    contact_counts: wp.array[int],
    body_index: int,
    capacity: int,
    fatal_on_saturation: int,
    int_metrics: wp.array[int],
):
    observed = contact_counts[body_index]
    int_metrics[6] = observed
    if observed > int_metrics[7]:
        int_metrics[7] = observed
    if observed > int_metrics[8]:
        int_metrics[8] = observed
    if observed >= capacity:
        int_metrics[9] = 1
        int_metrics[15] = 1
        wp.atomic_min(int_metrics, 16, SAFETY_REASON_CONTACT_BUFFER)
        if fatal_on_saturation != 0:
            int_metrics[5] = 1


@dataclass(frozen=True)
class GpuSafetySnapshot:
    minimum_relative_j: float
    first_affected_tet_id: int
    nonfinite_tet_count: int
    inverted_tet_count: int
    warning_tet_count: int
    stop_tet_count: int
    fatal: bool
    candidate_rejected: bool
    reason_code: int
    contact_count: int
    frame_maximum_contact_count: int
    run_maximum_contact_count: int
    contact_buffer_saturated: bool
    maximum_free_particle_speed_m_s: float

    @property
    def reason(self) -> str:
        return SAFETY_REASON_NAMES.get(self.reason_code, "unknown_safety_reason")


class GpuSafetyMonitor:
    """Launch per-substep safety reductions without downloading simulation arrays."""

    def __init__(
        self,
        *,
        device,
        tet_indices: np.ndarray,
        rest_volumes: np.ndarray,
        particle_start: int,
        free_particle_indices: np.ndarray,
        warning_minimum_j: float,
        stop_minimum_j: float,
        fatal_minimum_j: float,
        contact_counts,
        contact_body_index: int,
        contact_capacity: int,
        fatal_on_contact_saturation: bool,
    ) -> None:
        tets = np.asarray(tet_indices, dtype=np.int32).reshape(-1, 4)
        rest = np.asarray(rest_volumes, dtype=np.float32).reshape(-1)
        if len(tets) != len(rest):
            raise ValueError("tet_indices and rest_volumes must have equal length")
        self.device = device
        self.tet_count = len(tets)
        self.particle_start = int(particle_start)
        self.warning_minimum_j = float(warning_minimum_j)
        self.stop_minimum_j = float(stop_minimum_j)
        self.fatal_minimum_j = float(fatal_minimum_j)
        self.contact_counts = contact_counts
        self.contact_body_index = int(contact_body_index)
        self.contact_capacity = int(contact_capacity)
        self.fatal_on_contact_saturation = int(fatal_on_contact_saturation)
        self.tet_indices_device = wp.array(
            tets.reshape(-1), dtype=wp.int32, device=device
        )
        self.rest_volumes_device = wp.array(rest, dtype=float, device=device)
        self.free_particle_indices_device = wp.array(
            np.asarray(free_particle_indices, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        self.float_metrics_device = wp.zeros(
            _FLOAT_METRIC_COUNT, dtype=float, device=device
        )
        self.int_metrics_device = wp.zeros(
            _INT_METRIC_COUNT, dtype=wp.int32, device=device
        )
        self.reset_frame()

    def reset_frame(self) -> None:
        wp.launch(
            _reset_frame_metrics,
            dim=1,
            inputs=[
                self.tet_count,
                self.float_metrics_device,
                self.int_metrics_device,
            ],
            device=self.device,
        )

    def reset_candidate(self) -> None:
        wp.launch(
            _reset_candidate_metrics,
            dim=1,
            inputs=[
                self.tet_count,
                self.float_metrics_device,
                self.int_metrics_device,
            ],
            device=self.device,
        )

    def evaluate_candidate(self, state, *, include_free_speed: bool = False) -> None:
        wp.launch(
            _tet_safety_reduction,
            dim=self.tet_count,
            inputs=[
                state.particle_q,
                self.particle_start,
                self.tet_indices_device,
                self.rest_volumes_device,
                self.warning_minimum_j,
                self.stop_minimum_j,
                self.fatal_minimum_j,
                self.float_metrics_device,
                self.int_metrics_device,
            ],
            device=self.device,
        )
        wp.launch(
            _finish_tet_reduction,
            dim=1,
            inputs=[self.int_metrics_device],
            device=self.device,
        )
        if include_free_speed:
            wp.launch(
                _free_particle_speed_reduction,
                dim=len(self.free_particle_indices_device),
                inputs=[
                    state.particle_qd,
                    self.particle_start,
                    self.free_particle_indices_device,
                    self.float_metrics_device,
                ],
                device=self.device,
            )
        wp.launch(
            _contact_buffer_reduction,
            dim=1,
            inputs=[
                self.contact_counts,
                self.contact_body_index,
                self.contact_capacity,
                self.fatal_on_contact_saturation,
                self.int_metrics_device,
            ],
            device=self.device,
        )

    def readback(self) -> GpuSafetySnapshot:
        floats = self.float_metrics_device.numpy()
        integers = self.int_metrics_device.numpy()
        first_bad = int(integers[_FRAME_FIRST_BAD])
        if first_bad >= self.tet_count:
            first_bad = -1
        reason_code = int(integers[_CANDIDATE_REASON])
        return GpuSafetySnapshot(
            minimum_relative_j=float(floats[_FRAME_MIN_J]),
            first_affected_tet_id=first_bad,
            nonfinite_tet_count=int(integers[_FRAME_NONFINITE_MAX]),
            inverted_tet_count=int(integers[_FRAME_INVERTED_MAX]),
            warning_tet_count=int(integers[_FRAME_WARNING_MAX]),
            stop_tet_count=int(integers[_FRAME_STOP_MAX]),
            fatal=bool(integers[_FRAME_FATAL]),
            candidate_rejected=bool(integers[_CANDIDATE_REJECT]),
            reason_code=reason_code,
            contact_count=int(integers[_CONTACT_CURRENT]),
            frame_maximum_contact_count=int(integers[_CONTACT_FRAME_MAX]),
            run_maximum_contact_count=int(integers[_CONTACT_RUN_MAX]),
            contact_buffer_saturated=bool(integers[_CONTACT_SATURATED]),
            maximum_free_particle_speed_m_s=float(floats[_MAX_FREE_SPEED]),
        )
