"""
live_filter.py — a LIVE, causal, real-time smoothing / denoising filter.

Feed it one noisy measurement at a time; it returns a smoothed estimate using
ONLY past + current data (no lookahead), so it deploys straight onto a live
sensor stream.

Core = a constant-velocity Kalman filter (state = [position, velocity]):
  * tracks a MOVING signal with no steady-state lag (it predicts ahead with the
    estimated velocity), while
  * suppressing measurement noise (it blends prediction + measurement by their
    relative confidence).
It also handles the two things real streams do: dropped/invalid samples (it
coasts on the prediction) and gross outliers (an optional innovation gate).

    from live_filter import LiveFilter
    f = LiveFilter()
    for z, t in stream:                 # z = noisy sample (or None if dropped)
        estimate = f.update(z, t)       # smoothed, real-time

`KalmanCV` is the same thing with an explicit dt if you'd rather manage time.
This is the building block the 8x8 sensor (tof_sensor.py) runs per zone.
"""
from __future__ import annotations
import math


def _finite(z):
    return z is not None and not (isinstance(z, float) and math.isnan(z))


class KalmanCV:
    """Constant-velocity 1-D Kalman filter. Causal; one measurement per step.

    process_accel_psd : how much the target may accelerate (mm^2/s^3). Higher =
                        more agile (tracks maneuvers, passes more noise).
    measurement_var   : sensor noise variance (mm^2). Higher = smoother / more lag.
    reject_sigma      : gate — measurements more than this many sqrt(S) from the
                        prediction are treated as outliers (coasted). None = off.
    max_coast_s       : after this long with no valid measurement, reset.
    """

    def __init__(self, process_accel_psd: float = 50.0, measurement_var: float = 4.0,
                 reject_sigma: float | None = 5.0, max_consec_reject: int = 2,
                 max_coast_s: float = 0.5, init_vel_var: float = 1e6):
        if process_accel_psd <= 0 or measurement_var <= 0:
            raise ValueError("process_accel_psd and measurement_var must be > 0")
        self.q = float(process_accel_psd)
        self.r = float(measurement_var)
        self.reject_sigma = reject_sigma
        self.max_consec_reject = int(max_consec_reject)   # rejects before we call it a maneuver
        self.max_coast_s = float(max_coast_s)
        self.init_vel_var = float(init_vel_var)
        self.reset()

    def reset(self):
        self.x = 0.0            # position estimate
        self.v = 0.0            # velocity estimate
        self.P = [[0.0, 0.0], [0.0, 0.0]]   # covariance
        self.initialized = False
        self._coast = 0.0
        self._consec_reject = 0
        # last-step diagnostics
        self.innovation = float("nan")
        self.innovation_var = float("nan")
        self.gain = float("nan")
        self.rejected = False
        self.coasting = False

    @property
    def variance(self) -> float:
        """Position estimate variance (mm^2)."""
        return self.P[0][0]

    @property
    def velocity(self) -> float:
        """Velocity estimate (mm/s)."""
        return self.v

    def update(self, z, dt: float) -> float:
        """Advance dt seconds, fold in measurement z (None/nan = no measurement).
        Returns the position estimate."""
        if dt < 0:
            raise ValueError("dt must be >= 0")
        valid = _finite(z)

        if not self.initialized:
            if valid:
                self.x, self.v = float(z), 0.0
                self.P = [[self.r, 0.0], [0.0, self.init_vel_var]]
                self.initialized = True
                self._coast = 0.0
            return self.x

        # ---- predict ----
        q = self.q
        Q = [[q*dt**3/3.0, q*dt**2/2.0], [q*dt**2/2.0, q*dt]]
        # x' = x + v*dt ; v' = v
        x = self.x + self.v * dt
        v = self.v
        P = self.P
        # P' = F P F^T + Q, F = [[1,dt],[0,1]]
        p00 = P[0][0] + dt*(P[1][0] + P[0][1]) + dt*dt*P[1][1] + Q[0][0]
        p01 = P[0][1] + dt*P[1][1] + Q[0][1]
        p10 = P[1][0] + dt*P[1][1] + Q[1][0]
        p11 = P[1][1] + Q[1][1]

        self.rejected = False
        self.coasting = not valid
        if valid:
            y = float(z) - x                 # innovation
            S = p00 + self.r                 # innovation variance
            self.innovation, self.innovation_var = y, S
            gated = (self.reject_sigma is not None and S > 0
                     and abs(y) > self.reject_sigma * math.sqrt(S))
            if gated and self._consec_reject < self.max_consec_reject:
                self.rejected = True         # isolated spike -> coast this step
                valid = False
                self._consec_reject += 1
            elif gated:                      # sustained big innovation = real maneuver: re-lock
                self.x, self.v = float(z), 0.0
                self.P = [[self.r, 0.0], [0.0, self.init_vel_var]]
                self._consec_reject = 0
                self.gain = 1.0
                return self.x
            else:
                self._consec_reject = 0

        if valid:
            k0 = p00 / S                     # Kalman gain
            k1 = p10 / S
            self.gain = k0
            x = x + k0 * y
            v = v + k1 * y
            # Joseph-free covariance update: P = (I-KH)P
            np00 = (1 - k0) * p00
            np01 = (1 - k0) * p01
            np10 = p10 - k1 * p00
            np11 = p11 - k1 * p01
            self.P = [[np00, np01], [np10, np11]]
            self._coast = 0.0
        else:
            self.P = [[p00, p01], [p10, p11]]
            self._coast += dt
            if self._coast > self.max_coast_s:
                self.reset()
                if _finite(z):
                    return self.update(z, 0.0)
                return self.x

        self.x, self.v = x, v
        return self.x


class LiveFilter:
    """Timestamp-driven wrapper: call update(z, t) with absolute time stamps."""

    def __init__(self, **kw):
        self.kf = KalmanCV(**kw)
        self._t_prev = None

    def update(self, z, t: float) -> float:
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t
        return self.kf.update(z, dt)

    def reset(self):
        self.kf.reset(); self._t_prev = None

    @property
    def velocity(self): return self.kf.v

    @property
    def variance(self): return self.kf.variance


def stream(samples, filt: "LiveFilter | None" = None, realtime: bool = False):
    """Run a live stream of (t, z) through `filt`, yielding (t, z, estimate).

    samples : iterable of (t, z)  — t seconds, z measurement (None = dropped)
    realtime: if True, sleep to play the stream back at wall-clock speed.
    """
    import time
    filt = filt or LiveFilter()
    t0_wall = t0_data = None
    for t, z in samples:
        if realtime:
            if t0_wall is None:
                t0_wall, t0_data = time.perf_counter(), t
            target = t0_wall + (t - t0_data)
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
        yield t, z, filt.update(z, t)


if __name__ == "__main__":                   # tiny live demo
    import numpy as np
    rng = np.random.default_rng(0)
    t = np.arange(0, 5, 1/30)
    truth = 100 + 40 * t                     # a ramp
    noisy = truth + rng.normal(0, 5, len(t))
    est = np.array([e for _, _, e in stream(zip(t, noisy))])
    print(f"ramp: raw noise σ {np.std(noisy-truth):.2f} -> filtered σ {np.std(est-truth):.2f} mm")
