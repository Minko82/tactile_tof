"""Prescribed, repeatable approach/press/hold/release/recovery trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    phase: str
    normal_travel_m: float
    normal_velocity_m_s: float
    lateral_offset_m: np.ndarray
    lateral_velocity_m_s: np.ndarray


def _ramp(alpha: float, easing: str) -> tuple[float, float]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if easing == "linear":
        return alpha, 1.0
    if easing == "smoothstep":
        return alpha * alpha * (3.0 - 2.0 * alpha), 6.0 * alpha * (1.0 - alpha)
    raise ValueError(f"Unsupported trajectory easing {easing!r}")


class PrescribedTrajectory:
    """Piecewise prescribed trajectory with explicit phase durations."""

    def __init__(self, config: dict[str, Any]):
        self.clearance_m = float(config["clearance_m"])
        self.indentation_m = float(config["indentation_m"])
        self.easing = str(config["easing"])
        durations = config["durations_s"]
        self.durations = {
            "approach": float(durations["approach"]),
            "press": float(durations["press"]),
            "hold": float(durations["hold"]),
            "release": float(durations["release"]),
            "recovery": float(durations["recovery"]),
        }
        slip = config.get("lateral_slip", {})
        self.slip_enabled = bool(slip.get("enabled", False))
        if self.slip_enabled:
            self.durations["lateral_slip"] = float(slip["duration_s"])
            direction = np.asarray(slip["direction"], dtype=np.float64)
            direction /= np.linalg.norm(direction)
            self.slip_vector_m = direction * float(slip["distance_m"])
        else:
            self.slip_vector_m = np.zeros(3, dtype=np.float64)
        for phase, duration in self.durations.items():
            if duration < 0.0 or (
                phase in ("approach", "press", "release") and duration <= 0.0
            ):
                raise ValueError(
                    f"trajectory duration for {phase} is invalid: {duration}"
                )
        if self.clearance_m < 0.0 or self.indentation_m <= 0.0:
            raise ValueError(
                "trajectory clearance must be nonnegative and indentation positive"
            )

        order = ["approach", "press", "hold"]
        if self.slip_enabled:
            order.append("lateral_slip")
        order.extend(("release", "recovery"))
        self.phase_order = tuple(order)
        self.starts: dict[str, float] = {}
        cursor = 0.0
        for phase in self.phase_order:
            self.starts[phase] = cursor
            cursor += self.durations[phase]
        self.total_duration_s = cursor

    def _phase_at(self, time_s: float) -> str:
        clamped = float(np.clip(time_s, 0.0, self.total_duration_s))
        for phase in self.phase_order[:-1]:
            if clamped < self.starts[phase] + self.durations[phase]:
                return phase
        return self.phase_order[-1]

    def sample(self, time_s: float) -> TrajectorySample:
        time_s = float(np.clip(time_s, 0.0, self.total_duration_s))
        phase = self._phase_at(time_s)
        duration = self.durations[phase]
        local = 1.0 if duration == 0.0 else (time_s - self.starts[phase]) / duration
        value, derivative = _ramp(local, self.easing)
        full_travel = self.clearance_m + self.indentation_m
        lateral = np.zeros(3, dtype=np.float64)
        lateral_velocity = np.zeros(3, dtype=np.float64)

        if phase == "approach":
            travel = self.clearance_m * value
            velocity = self.clearance_m * derivative / duration
        elif phase == "press":
            travel = self.clearance_m + self.indentation_m * value
            velocity = self.indentation_m * derivative / duration
        elif phase == "hold":
            travel = full_travel
            velocity = 0.0
        elif phase == "lateral_slip":
            travel = full_travel
            velocity = 0.0
            lateral = self.slip_vector_m * value
            lateral_velocity = self.slip_vector_m * derivative / duration
        elif phase == "release":
            travel = full_travel * (1.0 - value)
            velocity = -full_travel * derivative / duration
            lateral = self.slip_vector_m
        else:
            travel = 0.0
            velocity = 0.0
            lateral = self.slip_vector_m
        return TrajectorySample(
            time_s=time_s,
            phase=phase,
            normal_travel_m=float(travel),
            normal_velocity_m_s=float(velocity),
            lateral_offset_m=lateral,
            lateral_velocity_m_s=lateral_velocity,
        )

    def nominal_indentation_m(self, sample: TrajectorySample) -> float:
        return max(0.0, sample.normal_travel_m - self.clearance_m)


# Backward-compatible import name for output-schema v2.
DeterministicTrajectory = PrescribedTrajectory
