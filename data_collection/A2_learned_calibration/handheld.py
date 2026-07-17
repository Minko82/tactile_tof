"""
handheld.py — point the (unmounted) sensor at things and watch live, no robot.

Left:  8x8 distance heatmap (mm, invalid zones blanked)
Right: 8x8 signal heatmap (kcps/SPAD — reflectivity map of whatever you aim at)
Below: scrolling trace of raw / corrected / filtered central distance

Calibration: generic model by default, or a fitted surface:
    python3 handheld.py
    python3 handheld.py wood
Close the window (or Ctrl-C) to stop.
"""
import math
import os
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "A3_proximity"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "A2_data_filter"))

import statistics
import numpy as np

import tof_logger
from robot import FT_CENTRAL
from calibration import load_for_surface, GenericCalibration
from live_filter import LiveFilter
from run_test import PROCESS_ACCEL_PSD, MANEUVER_ACCEL_PSD, REJECT_SIGMA

WINDOW_S = 15.0
HIST = 300


def main(dev=None, max_seconds=None):
    import matplotlib.pyplot as plt

    surf = sys.argv[1] if len(sys.argv) > 1 else "handheld_generic"
    cal = load_for_surface(surf)
    kind = ("generic model" if isinstance(cal, GenericCalibration)
            else f"'{surf}' surface fit")
    filt = LiveFilter(process_accel_psd=PROCESS_ACCEL_PSD,
                      maneuver_accel_psd=MANEUVER_ACCEL_PSD,
                      reject_sigma=REJECT_SIGMA)
    dev = dev or tof_logger.open_sensor()

    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.3], hspace=0.3, wspace=0.25)
    axD, axS = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[1, :])
    try:
        fig.canvas.manager.set_window_title("handheld ToF — live")
    except AttributeError:
        pass

    imD = axD.imshow(np.full((8, 8), np.nan), cmap="viridis", vmin=0, vmax=600)
    axD.set_title("distance (mm)")
    fig.colorbar(imD, ax=axD, fraction=0.046)
    imS = axS.imshow(np.full((8, 8), np.nan), cmap="magma", vmin=0, vmax=3000)
    axS.set_title("signal (kcps/SPAD)")
    fig.colorbar(imS, ax=axS, fraction=0.046)
    for ax in (axD, axS):
        ax.set_xticks([]); ax.set_yticks([])

    tq, rawq, corq, filq = (deque(maxlen=HIST) for _ in range(4))
    ln_raw, = axT.plot([], [], ".", ms=3, color="0.6", label="raw")
    ln_cor, = axT.plot([], [], "-", lw=1.2, color="tab:orange", label="corrected")
    ln_fil, = axT.plot([], [], "-", lw=1.6, color="tab:blue", label="filtered")
    axT.set_xlabel("time (s)"); axT.set_ylabel("central distance (mm)")
    axT.legend(loc="upper right", fontsize=8); axT.grid(alpha=0.3)
    fig.suptitle(f"handheld — {kind} — waiting for frames ...")

    t0 = time.perf_counter()
    # plt.pause() polling loop — FuncAnimation SIGTRAPs on the macosx backend
    plt.show(block=False)
    headless = plt.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template")
    try:
        while headless or plt.fignum_exists(fig.number):
            t = time.perf_counter() - t0
            if max_seconds and t > max_seconds:
                break
            # drain everything buffered so the display shows the NEWEST frame
            frame = None
            while True:
                full = tof_logger.read_frame_full(dev)
                if full is not None:
                    frame = full
                if dev.in_waiting < 100:
                    break
            if frame is None:
                plt.pause(0.02)
                continue
            dist, sig, sigma, amb = frame

            d8 = np.array([float(v) for v in dist], float).reshape(8, 8)
            d8[d8 <= 0] = np.nan
            imD.set_data(d8)
            if np.isfinite(d8).any():
                imD.set_clim(np.nanmin(d8), max(np.nanmax(d8), np.nanmin(d8) + 1))
            if sig:
                s8 = np.array([float(v) for v in sig], float).reshape(8, 8)
                s8[s8 <= 0] = np.nan
                imS.set_data(s8)
                if np.isfinite(s8).any():
                    imS.set_clim(0, max(np.nanmax(s8), 1))

            vals = [dist[i] for i in FT_CENTRAL if dist[i] > 0]
            svals = [sig[i] for i in FT_CENTRAL if sig[i] > 0] if sig else []
            med = statistics.median(vals) if vals else None
            sig_med = statistics.median(svals) if svals else None
            z = cal.apply(med, sig_med) if med is not None else None
            est = filt.update(z, t)
            tq.append(t)
            rawq.append(med if med is not None else math.nan)
            corq.append(z if z is not None else math.nan)
            filq.append(est)

            ln_raw.set_data(tq, rawq); ln_cor.set_data(tq, corq)
            ln_fil.set_data(tq, filq)
            axT.set_xlim(max(0.0, t - WINDOW_S), t + 0.5)
            vis = [v for v in list(corq) + list(filq) if math.isfinite(v)]
            if vis:
                axT.set_ylim(min(vis) - 15, max(vis) + 15)
            status = ("NO TARGET — coasting" if med is None else
                      f"corrected {z:6.1f} mm   vel {filt.velocity:+6.0f} mm/s   "
                      f"sig {sig_med or 0:5.0f}")
            fig.suptitle(f"handheld — {kind} — {status}")
            if not headless:
                plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()
    return fig


if __name__ == "__main__":
    main()
