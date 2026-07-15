"""
A2 analysis -- quantify the noise reduction of a 1-D Kalman filter on the ToF
peak-distance for a STATIC sensor.

  Jitter        = sigma(peak-distance)
  Reduction     = jitter_raw / jitter_filtered

Input: a raw recording CSV with columns  time_s, z0..z63  (from a2_record.py).
Usage:  python3 a2_kalman.py A2/raw_1.csv

No robot, no sensor -- pure offline analysis. Safe to run anytime.
"""
import sys, csv, statistics

# How to reduce each 8x8 frame to one "peak distance" (mm)
#   center  : the central zone only (continuous, the boresight reading)   [default]
#   median  : median of all valid zones (robust; note: spatially pre-averages)
#   closest : the minimum valid distance
#   mode    : histogram peak in 10 mm bins (coarse -- quantizes, avoid for jitter)
PEAK_METHOD = "center"

# Kalman tuning
#   R is auto-estimated from the raw signal (sensor noise variance).
#   Q_OVER_R sets the smoothing: smaller -> more smoothing / more reduction / more lag.
Q_OVER_R = 1e-3


def peak_distance(frame):
    valid = [d for d in frame if d > 0]                 # drop -1 (no valid target)
    if not valid:
        return None
    if PEAK_METHOD == "center":
        d = frame[27]                                   # a central zone of the 8x8
        return float(d) if d > 0 else statistics.median(valid)
    if PEAK_METHOD == "closest":
        return float(min(valid))
    if PEAK_METHOD == "mode":
        from collections import Counter
        return float(Counter(round(d / 10) * 10 for d in valid).most_common(1)[0][0])
    return float(statistics.median(valid))              # "median"


class Kalman1D:
    """Constant-position (random-walk) 1-D Kalman filter.
    q = process-noise variance per step, r = measurement-noise variance."""
    def __init__(self, q, r, p0=1e6):
        self.q, self.r, self.p, self.x = q, r, p0, None

    def update(self, z):
        if self.x is None:                              # initialise on first sample
            self.x = z
            return z
        self.p += self.q                                # predict
        k = self.p / (self.p + self.r)                  # Kalman gain
        self.x += k * (z - self.x)                      # correct
        self.p *= (1 - k)
        return self.x


def load(path):
    ts, peaks = [], []
    with open(path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        zcols = [i for i, h in enumerate(header) if h.startswith("z")]
        for row in rdr:
            if not row:
                continue
            frame = [int(float(row[i])) for i in zcols]
            p = peak_distance(frame)
            if p is not None:
                ts.append(float(row[0]))
                peaks.append(p)
    return ts, peaks


def main(path):
    ts, raw = load(path)
    if len(raw) < 10:
        print("Not enough valid samples in", path)
        return

    R = statistics.pvariance(raw)                       # measurement noise variance (mm^2)
    R = max(R, 1e-6)
    Q = R * Q_OVER_R
    kf = Kalman1D(q=Q, r=R)
    filt = [kf.update(z) for z in raw]

    jitter_raw  = statistics.pstdev(raw)
    jitter_filt = statistics.pstdev(filt)
    reduction   = jitter_raw / jitter_filt if jitter_filt > 0 else float("inf")
    rate = len(ts) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else float("nan")

    print(f"file             : {path}")
    print(f"samples          : {len(raw)}  (~{rate:.1f} Hz over {ts[-1]-ts[0]:.1f} s)")
    print(f"peak method      : {PEAK_METHOD}")
    print(f"R (meas var)     : {R:.4f} mm^2      Q/R = {Q_OVER_R:g}")
    print(f"jitter raw       : {jitter_raw:.4f} mm  (sigma)")
    print(f"jitter filtered  : {jitter_filt:.4f} mm  (sigma)")
    print(f"noise reduction  : {reduction:.2f}x")

    out = path.rsplit(".", 1)[0] + "_filtered.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "peak_raw_mm", "peak_filtered_mm"])
        for t, zr, zf in zip(ts, raw, filt):
            w.writerow([f"{t:.4f}", f"{zr:.2f}", f"{zf:.3f}"])
    print(f"wrote            : {out}")

    try:                                                # optional plot if matplotlib present
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))
        plt.plot(ts, raw, ".", ms=2, alpha=.4, label=f"raw  (sigma={jitter_raw:.2f} mm)")
        plt.plot(ts, filt, "-", lw=1.6, label=f"Kalman  (sigma={jitter_filt:.2f} mm)")
        plt.xlabel("time (s)"); plt.ylabel("peak distance (mm)")
        plt.title(f"A2 Kalman noise reduction: {reduction:.1f}x"); plt.legend()
        plt.tight_layout()
        png = path.rsplit(".", 1)[0] + "_plot.png"
        plt.savefig(png, dpi=120)
        print(f"wrote            : {png}")
    except ImportError:
        print("(install matplotlib for a raw-vs-filtered plot)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A2/raw_1.csv")
