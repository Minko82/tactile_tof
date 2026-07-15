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
It also handles the things real streams do: dropped/invalid samples coast on
the prediction, isolated outliers are softly downweighted, and repeated
consistent innovations switch the filter briefly into an agile maneuver mode.

    from live_filter import LiveFilter
    f = LiveFilter()
    for z, t in stream:                 # z = noisy sample (or None if dropped)
        estimate = f.update(z, t)       # smoothed, real-time

`KalmanCV` is the same thing with an explicit dt if you'd rather manage time.
The 8x8 sensor pipeline in tof_sensor.py uses a separate IMM implementation.
See KALMAN_FILTER_THEORY.md for derivations and VL53L5CX integration guidance.
"""
from __future__ import annotations
import math
import operator


def _finite(z):
    """Whether z is a usable scalar measurement (conversion errors propagate)."""
    return z is not None and math.isfinite(float(z))


def _number(name, value, *, positive=False):
    """Convert and validate one finite numeric tuning parameter."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        relation = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be finite and {relation}")
    return value


class KalmanCV:
    """Constant-velocity 1-D Kalman filter. Causal; one measurement per step.

    process_accel_psd : how much the target may accelerate (mm^2/s^3). Higher =
                        more agile (tracks maneuvers, passes more noise).
    measurement_var   : sensor noise variance (mm^2). Higher = smoother / more lag.
    reject_sigma      : robust threshold in innovation standard deviations.
                        Large isolated measurements are softly downweighted.
    max_consec_reject : consistent large innovations needed after the first to
                        confirm a real maneuver and re-lock.
    max_coast_s       : after this long with no valid measurement, reset.
    maneuver_accel_psd: temporary high process PSD after a confirmed maneuver.
    adapt_measurement_var: estimate R online from ordinary innovations. Prefer
                        per-sample VL53L5CX range sigma when it is available.
    """

    def __init__(self, process_accel_psd: float = 500.0, measurement_var: float = 4.0,
                 reject_sigma: float | None = 5.0, max_consec_reject: int = 1,
                 max_coast_s: float = 0.5, init_vel_var: float = 1e6,
                 maneuver_accel_psd: float = 5000.0, maneuver_hold_s: float = 0.25,
                 maneuver_consistency_sigma: float = 3.0,
                 adapt_measurement_var: bool = False, measurement_adapt_rate: float = 0.02,
                 measurement_var_min: float = 0.25, measurement_var_max: float = 1e4):
        self.q = _number("process_accel_psd", process_accel_psd, positive=True)
        self.r = _number("measurement_var", measurement_var, positive=True)
        self.reject_sigma = (None if reject_sigma is None else
                             _number("reject_sigma", reject_sigma, positive=True))
        try:
            self.max_consec_reject = operator.index(max_consec_reject)
        except TypeError as exc:
            raise ValueError("max_consec_reject must be a nonnegative integer") from exc
        if self.max_consec_reject < 0:
            raise ValueError("max_consec_reject must be a nonnegative integer")
        self.max_coast_s = _number("max_coast_s", max_coast_s)
        self.init_vel_var = _number("init_vel_var", init_vel_var)
        self.q_maneuver = _number("maneuver_accel_psd", maneuver_accel_psd, positive=True)
        if self.q_maneuver < self.q:
            raise ValueError("maneuver_accel_psd must be >= process_accel_psd")
        self.maneuver_hold_s = _number("maneuver_hold_s", maneuver_hold_s)
        self.maneuver_consistency_sigma = _number(
            "maneuver_consistency_sigma", maneuver_consistency_sigma, positive=True)
        self.adapt_measurement_var = bool(adapt_measurement_var)
        self.measurement_adapt_rate = _number(
            "measurement_adapt_rate", measurement_adapt_rate, positive=True)
        if self.measurement_adapt_rate > 1.0:
            raise ValueError("measurement_adapt_rate must be <= 1")
        self.measurement_var_min = _number(
            "measurement_var_min", measurement_var_min, positive=True)
        self.measurement_var_max = _number(
            "measurement_var_max", measurement_var_max, positive=True)
        if self.measurement_var_max < self.measurement_var_min:
            raise ValueError("measurement_var_max must be >= measurement_var_min")
        self.reset()

    def reset(self):
        self.x = float("nan")   # position estimate; unavailable until first measurement
        self.v = 0.0            # velocity estimate
        self.P = [[0.0, 0.0], [0.0, 0.0]]   # covariance
        self.initialized = False
        self._coast = 0.0
        self._consec_reject = 0
        self._pending_z = None
        self._pending_sign = 0
        self._maneuver_remaining = 0.0
        # last-step diagnostics
        self.innovation = float("nan")
        self.innovation_var = float("nan")
        self.gain = float("nan")
        self.rejected = False
        self.coasting = False
        self.downweighted = False
        self.maneuver = False
        self.nis = float("nan")
        self.effective_measurement_var = self.r
        self.active_process_accel_psd = self.q

    @property
    def variance(self) -> float:
        """Position estimate variance (mm^2)."""
        return self.P[0][0]

    @property
    def velocity(self) -> float:
        """Velocity estimate (mm/s)."""
        return self.v

    def update(self, z, dt: float, measurement_var: float | None = None,
               maneuver_hint: bool = False) -> float:
        """Advance dt seconds and fold in z (None/non-finite = no measurement).

        Returns NaN until the first valid measurement initializes the filter.
        """
        try:
            dt = float(dt)
        except (TypeError, ValueError) as exc:
            raise ValueError("dt must be finite and >= 0") from exc
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and >= 0")
        valid = _finite(z)
        r = (self.r if measurement_var is None else
             _number("measurement_var", measurement_var, positive=True))
        self.effective_measurement_var = r

        if not self.initialized:
            self.coasting = not valid
            if valid:
                self.x, self.v = float(z), 0.0
                self.P = [[r, 0.0], [0.0, self.init_vel_var]]
                self.initialized = True
                self._coast = 0.0
            return self.x

        # ---- predict ----
        q = self.q_maneuver if self._maneuver_remaining > 0 else self.q
        self.active_process_accel_psd = q
        self._maneuver_remaining = max(0.0, self._maneuver_remaining - dt)
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
        self.downweighted = False
        self.maneuver = False
        self.innovation = float("nan")
        self.innovation_var = float("nan")
        self.nis = float("nan")
        if valid:
            y = float(z) - x                 # innovation
            S = p00 + r                      # nominal innovation variance
            self.innovation, self.innovation_var = y, S
            self.nis = y * y / S if S > 0 else float("inf")
            gated = (self.reject_sigma is not None and S > 0
                     and abs(y) > self.reject_sigma * math.sqrt(S))
            if gated:
                sign = 1 if y > 0 else -1
                tolerance = max(self.maneuver_consistency_sigma * math.sqrt(r),
                                0.05 * abs(y))
                consistent = (self._pending_z is not None and sign == self._pending_sign
                              and abs(float(z) - self._pending_z) <= tolerance)
                self._consec_reject = self._consec_reject + 1 if consistent else 1
                confirmed = bool(maneuver_hint) or (
                    consistent and self._consec_reject > self.max_consec_reject)
                if confirmed:
                    # Two or more agreeing large measurements are a maneuver, not a lone spike.
                    self.x = (float(z) if self._pending_z is None or maneuver_hint else
                              0.5 * (self._pending_z + float(z)))
                    self.v = v
                    relock_vel_var = max(self.q_maneuver * max(dt, 1.0 / 60.0), self.q)
                    self.P = [[r, 0.0], [0.0, relock_vel_var]]
                    self._consec_reject = 0
                    self._pending_z = None
                    self._pending_sign = 0
                    self._maneuver_remaining = self.maneuver_hold_s
                    self.active_process_accel_psd = self.q_maneuver
                    self.maneuver = True
                    self._coast = 0.0
                    self.gain = 1.0
                    return self.x

                self._pending_z = float(z)
                self._pending_sign = sign
                # Huber-style soft rejection: inflate R instead of discarding z.
                weight = self.reject_sigma * math.sqrt(S) / abs(y)
                r = r / max(weight * weight, 1e-12)
                S = p00 + r
                self.effective_measurement_var = r
                self.rejected = True
                self.downweighted = True
            else:
                self._pending_z = None
                self._pending_sign = 0
                self._consec_reject = 0
                if maneuver_hint:
                    self._maneuver_remaining = self.maneuver_hold_s
                    self.active_process_accel_psd = self.q_maneuver
                    self.maneuver = True

            if self.adapt_measurement_var and measurement_var is None and not gated:
                observed_r = min(max(y * y - p00, self.measurement_var_min),
                                 self.measurement_var_max)
                alpha = self.measurement_adapt_rate
                self.r = (1.0 - alpha) * self.r + alpha * observed_r
                self.r = min(max(self.r, self.measurement_var_min), self.measurement_var_max)

        if valid:
            k0 = p00 / S                     # Kalman gain
            k1 = p10 / S
            self.gain = k0
            x = x + k0 * y
            v = v + k1 * y
            # Joseph form: P = (I-KH)P(I-KH)^T + K R K^T.
            # It costs little for 2x2 state and preserves positive semidefiniteness.
            a = 1 - k0
            np00 = a*a*p00 + k0*k0*r
            np01 = a*(p01 - k1*p00) + k0*k1*r
            np10 = a*(p10 - k1*p00) + k1*k0*r
            np11 = p11 - k1*(p01 + p10) + k1*k1*(p00 + r)
            self.P = [[np00, np01], [np10, np11]]
            self._coast = 0.0
        else:
            self.P = [[p00, p01], [p10, p11]]
            self._coast += dt
            if self._coast > self.max_coast_s:
                self.reset()
                if _finite(z):
                    return self.update(z, 0.0, measurement_var, maneuver_hint)
                return self.x

        self.x, self.v = x, v
        return self.x


class LiveFilter:
    """Timestamp-driven wrapper: call update(z, t) with absolute time stamps."""

    def __init__(self, **kw):
        self.kf = KalmanCV(**kw)
        self._t_prev = None

    def update(self, z, t: float, measurement_var: float | None = None,
               maneuver_hint: bool = False) -> float:
        try:
            t = float(t)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be finite") from exc
        if not math.isfinite(t):
            raise ValueError("timestamp must be finite")
        if self._t_prev is not None and t < self._t_prev:
            raise ValueError("timestamps must be nondecreasing")
        dt = 0.0 if self._t_prev is None else t - self._t_prev
        estimate = self.kf.update(z, dt, measurement_var, maneuver_hint)
        self._t_prev = t
        return estimate

    def reset(self):
        self.kf.reset(); self._t_prev = None

    @property
    def velocity(self): return self.kf.v

    @property
    def variance(self): return self.kf.variance

    @property
    def nis(self): return self.kf.nis

    @property
    def downweighted(self): return self.kf.downweighted

    @property
    def maneuver(self): return self.kf.maneuver


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
