# Robust adaptive Kalman filtering for the VL53L5CX

This document explains the model in `live_filter.py`, the assumptions behind it,
and how to use it with STMicroelectronics' VL53L5CX multizone Time-of-Flight
(ToF) sensor. The implementation is causal: every output uses only the current
and previous frames, so it can run directly in the sensor acquisition loop.

## 1. What the VL53L5CX actually provides

The VL53L5CX is a direct-ToF sensor built around a 940 nm VCSEL and a SPAD
receiver array. It reports either a 4x4 or 8x8 grid, supports multiple targets
per zone, performs histogram processing, and has a nominal per-zone range of
2 cm to 4 m. Its field of view is approximately 65 degrees diagonally. See the
[VL53L5CX datasheet](https://www.st.com/resource/en/datasheet/vl53l5cx.pdf).

The resolution changes the available frame rate:

| Resolution | ST-supported frequency | This repository |
|---|---:|---:|
| 4x4 | 1-60 Hz | not the current configuration |
| 8x8 | 1-15 Hz | normally 10 Hz |

The limits and output definitions are documented in ST's
[VL53L5CX ULD user manual](https://www.st.com/resource/en/user_manual/dm00797144-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf).
The simulator therefore defaults to 10 Hz, not 30 or 60 Hz.

Each target result can include:

- `distance_mm`: measured distance;
- `range_sigma_mm`: ST's estimated standard deviation of that range;
- `target_status`: validity and failure information;
- `nb_target_detected`, signal rate, ambient rate, reflectance, and motion data.

ST describes status 5 as fully valid. Status 6 or 9 may be used at reduced
confidence. Other statuses should normally be treated as invalid rather than
fed into a Kalman correction. Keeping `range_sigma_mm`, target count, and
target status enabled gives the host much better information than distance
alone.

> Important close-range limitation: ST specifies ranging from 2 cm. A membrane
> or target at 10-15 mm is outside the guaranteed range. No Kalman filter can
> reconstruct accurate millimeters from consistently invalid or biased input.
> For quantitative tactile depth, design the rest gap and maximum indentation
> so valid zones preferably remain at or above 20 mm, and validate the complete
> membrane/cover/material system experimentally.

## 2. State and measurement model

For one VL53L5CX zone, the state is

```text
x_k = [position_mm, velocity_mm_per_s]^T
```

Over a short interval `dt`, the ordinary motion model assumes approximately
constant velocity:

```text
x_k^- = F x_(k-1)

F = [[1, dt],
     [0,  1]]
```

The distance sensor directly observes position:

```text
z_k = H x_k + measurement noise
H   = [1, 0]
```

This is a linear system, so an extended or unscented Kalman filter would not
improve it merely by being more complicated. The important questions are
whether the motion model is appropriate and whether the noise covariances are
credible.

## 3. Prediction and process covariance Q

Real targets accelerate, so velocity is not truly constant. The implementation
models acceleration as continuous white noise with spectral density `q`:

```text
Q = q * [[dt^3 / 3, dt^2 / 2],
         [dt^2 / 2, dt        ]]
```

Prediction is

```text
P_k^- = F P_(k-1) F^T + Q
```

`q` controls the central tradeoff:

- smaller `q`: smoother output, but more error during acceleration;
- larger `q`: faster tracking, but more measurement noise passes through.

The filter has two process-noise modes:

| Mode | Default PSD | Purpose |
|---|---:|---|
| ordinary | 500 mm^2/s^3 | balanced static and curved-motion tracking |
| maneuver | 5000 mm^2/s^3 | temporary agility after confirmed motion |

This is a lightweight multiple-model idea rather than a full interacting
multiple-model estimator. The ordinary mode handles static targets and modest
motion; the agile mode prevents the covariance from becoming overconfident
immediately after a large change. Constant-velocity and constant-acceleration
models, and combinations of motion models, are standard choices for tracking
maneuvering targets; see this
[survey of motion models](https://arxiv.org/abs/1905.06113).

## 4. Measurement covariance R

`R` is measurement-noise variance, not standard deviation:

```text
R = sigma_mm^2
```

The predicted measurement uncertainty and innovation are

```text
innovation:            nu = z - H x^-
innovation variance:   S  = H P^- H^T + R
normalized innovation: NIS = nu^2 / S
```

The scalar Kalman gain is derived from the position/velocity covariance:

```text
K = P^- H^T / S
x = x^- + K nu
```

When the firmware supplies `range_sigma_mm`, pass its square for that zone and
frame:

```python
estimate = filter.update(
    distance_mm,
    timestamp_s,
    measurement_var=max(float(range_sigma_mm) ** 2, 0.25),
)
```

This naturally trusts a strong return more than a weak, uncertain return. If
sigma is unavailable, characterize each zone at representative distances,
reflectances, integration times, and ambient-light levels. A single fixed `R`
is only a fallback because VL53L5CX detection volume and quality depend on
distance, reflectance, ambient light, resolution, sharpener, ranging mode, and
integration time.

The optional `adapt_measurement_var=True` mode estimates `R` slowly from normal
innovations. It is disabled by default because known per-frame sigma or a real
calibration is preferable. Adaptive innovation/residual covariance estimation
is supported in the literature; see
[Adaptive Adjustment of Noise Covariance in Kalman Filter](https://arxiv.org/abs/1702.00884).

## 5. Why a hard outlier gate was replaced

A conventional gate rejects a measurement when

```text
abs(nu) > reject_sigma * sqrt(S)
```

That protects a settled filter from an isolated spike. Unfortunately, a real
step produces exactly the same first observation: one unexpectedly large
innovation. Rejecting several frames caused the previous implementation to wait
about 100 ms at 30 Hz and made step RMSE very large.

The new filter uses robust variance inflation for a single large innovation:

```text
w = reject_sigma * sqrt(S) / abs(nu)       # 0 < w < 1 when gated
R_effective = R / w^2
```

The measurement is not discarded; its gain becomes very small. This is a
Huber-style soft update: an isolated VL53L5CX spike has bounded influence, but
the filter retains the evidence for maneuver detection. Innovation saturation
and robust Bayesian measurement updates are established approaches for
outlier-contaminated filtering:

- [Robust EKF with innovation saturation](https://arxiv.org/abs/1904.00335)
- [Outlier-robust Kalman filtering through Generalised Bayes](https://arxiv.org/abs/2405.05646)

The implementation uses the same principle in a small linear filter, without
the computational cost of an EKF or ensemble filter.

## 6. Maneuver confirmation

One extreme sample cannot be reliably classified as either a real jump or an
outlier. The filter therefore confirms a maneuver when consecutive large
innovations:

1. point in the same direction; and
2. report mutually consistent distances.

With the default `max_consec_reject=1`, the second agreeing large measurement
confirms the maneuver. The state position is relocked to the average of the two
measurements, velocity is retained, velocity uncertainty is reopened, and the
high-`q` mode runs for 0.25 s.

At the repository's 10 Hz 8x8 rate, confirmation necessarily costs one frame,
or about 100 ms. Reacting fully to the first sample would reduce step RMSE, but
would also make every isolated 60 mm sensor spike corrupt the estimate. That is
an information limit, not an algebra error. If another signal is available—ST's
motion indicator, robot command, contact switch, or agreement across adjacent
zones—pass `maneuver_hint=True`. It can confirm a maneuver and relock on the
first frame more safely.

## 7. Dropouts and target status

For `None`, NaN, infinity, or an invalid target status, the correction is
skipped and the filter returns its motion prediction. Covariance grows during
this coast, representing increasing uncertainty. After `max_coast_s`, the
filter resets and returns NaN until a valid measurement arrives.

A practical VL53L5CX mapping is:

```python
if status == 5 and target_count > 0:
    z = distance_mm
    r = max(range_sigma_mm**2, 0.25)
elif status in (6, 9) and target_count > 0:
    z = distance_mm
    r = max((2.0 * range_sigma_mm)**2, 1.0)  # project policy: reduced confidence
else:
    z = None
    r = None

estimate = filter.update(z, timestamp_s, measurement_var=r)
```

The factor of two for statuses 6/9 is a conservative project policy, not an ST
formula; validate or replace it using captured data.

## 8. Applying the model to the 8x8 field

Use one filter per zone after invalid-status rejection and per-zone distance
calibration:

```text
64 raw targets
  -> target-count/status validation
  -> per-zone distance calibration
  -> 64 robust adaptive filters using per-zone sigma
  -> spatial fusion / tactile deformation field
```

Temporal and spatial filtering solve different problems:

- the Kalman bank reduces frame-to-frame jitter and bridges short dropouts;
- robust spatial fusion uses agreement across zones and rejects geometrically
  inconsistent zones;
- baseline subtraction converts the filtered membrane distance into indentation;
- force calibration maps indentation features into physical force.

Do not estimate one global `R` from all 64 zones. Edge zones, reflectance,
ambient rate, cover geometry, and membrane angle can produce different noise.
The VL53L5CX range sigma is particularly valuable here.

## 9. Why this model fits VL53L5CX use cases

The model is a good fit when:

- frames arrive at a known 1-15 Hz rate in 8x8 mode;
- distance changes are locally smooth most of the time;
- occasional invalid targets, weak returns, and isolated spikes occur;
- output must be causal and cheap enough to run 64 times per frame;
- measurement confidence varies and can be represented by `range_sigma_mm`;
- fast contact or motion events need a controlled smoothness/latency tradeoff.

Each update is constant-time with a 2x2 covariance, so 64 filters are modest on
a host computer or microcontroller. The Joseph covariance update used in the
implementation preserves covariance symmetry and positive semidefiniteness more
reliably than the shorter covariance formula.

The model does **not** correct systematic range bias, cover-glass crosstalk,
zone geometry, target-selection mistakes, or operation below the specified
range. Calibration and correct VL53L5CX configuration must precede filtering.

## 10. Tuning and validation workflow

1. Configure the same resolution, rate, target order, sharpener, ranging mode,
   and integration time that production will use.
2. Enable target count, target status, and range sigma in the ULD output.
3. Record static datasets at several distances, zones, materials, and ambient
   conditions; estimate bias and variance separately.
4. Calibrate systematic distance bias before the Kalman update.
5. Start with `q=500`, `q_maneuver=5000`, `reject_sigma=5`, and real sigma-squared
   as `R`.
6. Validate on held-out static, ramp, curved, contact, dropout, and outlier data.
7. Report steady-state noise, total RMSE, transition RMSE, settling time,
   dropout RMSE, downweight count, and maneuver count separately.
8. Inspect NIS and innovation autocorrelation. Persistent biased or correlated
   innovations mean the motion/noise model is incomplete.

NIS/NEES-based automatic tuning can help, but matching only their average is
not sufficient. Distribution-aware consistency tests and multi-run optimization
are discussed in
[Kalman Filter Auto-tuning through Enforcing Chi-Squared Normalized Error Distributions](https://arxiv.org/abs/2306.07225).

Do not tune solely against the synthetic seed. Tune on multiple physical
captures and keep entire rounds held out, otherwise the reported RMSE will be
optimistic.

## 11. Reading RMSE correctly

RMSE combines bias, noise, lag, and outliers:

```text
RMSE = sqrt(mean((estimate - truth)^2))
```

Because errors are squared, one delayed 150 mm transition can outweigh hundreds
of accurate static samples. Always accompany total RMSE with:

- residual standard deviation during steady intervals;
- transition-only RMSE and settling time;
- maximum or 95th-percentile error;
- number of invalid, softly downweighted, and maneuver-confirming frames.

For offline analysis where future frames are available, a backward Kalman/RTS
smoother can reduce lag-related RMSE. It must not be used to claim live
performance because it is noncausal.

## 12. Minimal usage

```python
from live_filter import LiveFilter

zone_filter = LiveFilter(
    process_accel_psd=500.0,
    maneuver_accel_psd=5000.0,
    measurement_var=25.0,       # fallback only; prefer range_sigma_mm**2
    reject_sigma=5.0,
    max_consec_reject=1,
)

estimate_mm = zone_filter.update(
    distance_mm_or_none,
    timestamp_s,
    measurement_var=range_sigma_mm**2 if range_sigma_mm is not None else None,
    maneuver_hint=motion_or_spatial_consensus,
)

print(zone_filter.nis, zone_filter.downweighted, zone_filter.maneuver)
```

The simulator exposes the main tuning controls:

```bash
python3 simulate.py --hz 10 --process-accel-psd 500 \
  --maneuver-accel-psd 5000 --reject-sigma 5
```

Use `--adaptive-r` only when per-frame sigma is unavailable and compare it
against a fixed `R` estimated from real static captures.
