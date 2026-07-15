"""
visualize_live.py — LIVE raw-vs-filtered viewer for the 8x8 ToF sensor.

Plug in the ESP32-C6, run this, and watch three panels update in real time:
  * RAW field       — the noisy 8x8 as it arrives
  * FILTERED field  — the same after per-zone denoising (tof_sensor.ToFSensor)
  * distance trace  — raw central zone (noisy) vs fused proximity (smooth), scrolling

    ~/ur5-env/bin/python3 visualize_live.py                # real sensor
    ~/ur5-env/bin/python3 visualize_live.py --sim          # synthetic preview, no hardware
    ~/ur5-env/bin/python3 visualize_live.py --sim --snapshot figs/live_preview.png

A background thread reads the sensor and runs the filter at the true frame rate;
the plot just displays the latest state. No calibration is needed for raw-vs-
filtered; pass calibration to ToFSensor for accurate absolute mm.
"""
import argparse
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import matplotlib

from tof_sensor import ToFSensor, CENTRAL

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = 300                      # samples kept in the scrolling trace (~20 s at 15 Hz)
WINDOW_S = 20.0


class LiveState:
    """Thread-safe latest frame + rolling history."""
    def __init__(self):
        self.lock = threading.Lock()
        self.raw = np.full((8, 8), np.nan)
        self.filt = np.full((8, 8), np.nan)
        self.t = deque(maxlen=HIST)
        self.raw_c = deque(maxlen=HIST)
        self.filt_c = deque(maxlen=HIST)
        self.n = 0
        self.err = None

    def push(self, raw8, filt8, tsec, rawc, filtc):
        with self.lock:
            self.raw, self.filt = raw8, filt8
            self.t.append(tsec); self.raw_c.append(rawc); self.filt_c.append(filtc)
            self.n += 1

    def snapshot(self):
        with self.lock:
            return (self.raw.copy(), self.filt.copy(),
                    np.array(self.t), np.array(self.raw_c), np.array(self.filt_c),
                    self.n, self.err)


def _process(state, sensor, fr, t):
    frame = np.asarray(fr, float); frame[frame <= 0] = np.nan
    out = sensor.update(frame, t)
    filt8 = np.asarray(out["field_mm"], float)
    rawc = float(np.nanmedian([frame[z] for z in CENTRAL]))
    prox = out["proximity_mm"]
    filtc = float(prox) if (prox is not None and prox == prox) else \
        float(np.nanmedian([filt8.ravel()[z] for z in CENTRAL]))
    state.push(frame.reshape(8, 8), filt8, t, rawc, filtc)


def reader_serial(state, stop, dev):
    from capture_logger import read_frame
    sensor = ToFSensor()                 # coeffs=None -> identity cal, per-zone denoise
    t0 = time.perf_counter()
    try:
        while not stop.is_set():
            fr = read_frame(dev)
            if fr is None:
                continue
            _process(state, sensor, fr, time.perf_counter() - t0)
    except Exception as e:               # surface reader errors to the UI
        state.err = str(e)
    finally:
        try: dev.close()
        except Exception: pass


def reader_sim(state, stop):
    """Fake a noisy moving surface so the viewer can be previewed without hardware."""
    sensor = ToFSensor(); rng = np.random.default_rng(0); t0 = time.perf_counter()
    while not stop.is_set():
        t = time.perf_counter() - t0
        base = 200.0 + 150.0 * np.sin(2 * np.pi * 0.08 * t)      # surface sweeping in/out
        frame = base + rng.normal(0, 6.0, 64)                    # per-zone noise
        if rng.random() < 0.03:                                  # occasional outlier
            frame[rng.integers(0, 64)] += rng.choice([-1, 1]) * 70
        if rng.random() < 0.03:                                  # occasional dropout
            frame[rng.integers(0, 64)] = -1
        _process(state, sensor, frame, t)
        time.sleep(1 / 15.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", action="store_true", help="synthetic frames (no sensor)")
    ap.add_argument("--snapshot", metavar="PATH", help="render one frame to PATH and exit (headless)")
    args = ap.parse_args()

    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    state = LiveState(); stop = threading.Event()
    if args.sim:
        th = threading.Thread(target=reader_sim, args=(state, stop), daemon=True)
    else:
        try:
            from capture_logger import open_sensor
            dev = open_sensor()
        except Exception as e:
            print(f"Could not open the ToF sensor: {e}\n"
                  f"Plug in the ESP32-C6 and close any Serial Monitor, or preview with:\n"
                  f"    {sys.argv[0]} --sim")
            sys.exit(1)
        th = threading.Thread(target=reader_serial, args=(state, stop, dev), daemon=True)
    th.start()

    # Figure
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11})
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.2], wspace=0.42)
    axR, axF, axT = (fig.add_subplot(gs[0, i]) for i in range(3))
    imR = axR.imshow(np.zeros((8, 8)), cmap="viridis", interpolation="nearest")
    imF = axF.imshow(np.zeros((8, 8)), cmap="viridis", interpolation="nearest")
    for ax, ttl in [(axR, "RAW field"), (axF, "FILTERED field")]:
        ax.set_title(ttl, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(imF, ax=axF, fraction=0.046, pad=0.04, label="mm")
    lineR, = axT.plot([], [], color="#aeb6c0", lw=0.9, label="raw (central zone)")
    lineF, = axT.plot([], [], color="#d1354a", lw=1.8, label="filtered (proximity)")
    axT.set_title("raw vs filtered distance", fontweight="bold")
    axT.set_xlabel("time (s)"); axT.set_ylabel("mm"); axT.grid(alpha=.3)
    axT.legend(loc="upper right", framealpha=.95)
    base = "Live ToF — raw vs filtered" + ("  (SIM)" if args.sim else "")
    sup = fig.suptitle(base, fontsize=14, fontweight="bold")

    def update(_):
        raw, filt, ts, rc, fc, n, err = state.snapshot()
        if err:
            sup.set_text(f"{base}     ⚠ sensor error: {err}"); sup.set_color("#c0392b")
        elif n == 0:
            sup.set_text(f"{base}     …waiting for sensor")
        else:
            fps = 0.0
            if ts.size >= 2:
                m = min(ts.size, 30); span = ts[-1] - ts[-m]
                fps = (m - 1) / span if span > 0 else 0.0
            sup.set_text(f"{base}     ●  {fps:.0f} fps · {n} frames"); sup.set_color("#12263a")
        both = np.concatenate([raw[np.isfinite(raw)], filt[np.isfinite(filt)]])
        if both.size:
            lo, hi = np.nanpercentile(both, [4, 96])
            if hi - lo < 1:
                hi = lo + 1
            imR.set_clim(lo, hi); imF.set_clim(lo, hi)
        imR.set_data(np.nan_to_num(raw, nan=np.nanmin(raw) if np.isfinite(raw).any() else 0))
        imF.set_data(filt)
        if ts.size:
            lineR.set_data(ts, rc); lineF.set_data(ts, fc)
            axT.set_xlim(max(0, ts[-1] - WINDOW_S), ts[-1] + 0.5)
            vis = np.concatenate([rc[np.isfinite(rc)], fc[np.isfinite(fc)]])
            if vis.size:
                axT.set_ylim(vis.min() - 10, vis.max() + 10)
        return imR, imF, lineR, lineF

    if args.snapshot:                    # headless: wait for frames, render one
        deadline = time.time() + 5
        while state.n < 45 and time.time() < deadline and state.err is None:
            time.sleep(0.1)
        update(0)
        os.makedirs(os.path.dirname(os.path.join(HERE, args.snapshot)) or ".", exist_ok=True)
        out = args.snapshot if os.path.isabs(args.snapshot) else os.path.join(HERE, args.snapshot)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        stop.set()
        print(f"wrote {out}  ({state.n} frames)" + (f"  reader error: {state.err}" if state.err else ""))
        return

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
