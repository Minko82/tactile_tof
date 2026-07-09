#!/usr/bin/env python3
"""
A2 — Baseline signal recorder (static ToF capture)
--------------------------------------------------
Records the VL53L5CX stream for a fixed window (default 60 s) with the sensor in
a fully static state (no contact, fixed standoff, consistent ambient light), per
the A2 test spec. Output is a ``timestamp,zone_00..zone_63`` CSV — identical
format to visualizer.py — so it loads with tof_common and feeds analyze_a2_kalman.py.

This recorder is headless (no matplotlib) and only needs ``pyserial`` + ``numpy``:
    pip install pyserial numpy

Sample rate note
================
The spec calls for 60 Hz, but the VL53L5CX ceiling at 8x8 (64 zones) is 15 Hz;
60 Hz is only reachable at 4x4. Since we keep 64 zones, run the sensor as fast as
it goes at 8x8 (flash VL53L5CX_A2_Recorder at 15 Hz) and let the Kalman filter
process every sample using dt from timestamps. This script logs whatever rate the
firmware emits and reports the achieved effective rate at the end.

Run:
    python3 data_collection/a2_record.py                 # auto-detect port, 60 s
    python3 data_collection/a2_record.py --duration 120  # longer window
    python3 data_collection/a2_record.py --port /dev/cu.usbmodem2101
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("Missing dependency: run  pip install pyserial  then try again.")

N_ZONES = 64
BAUD_RATE = 115200
MAX_RANGE_MM = 800.0            # far-range gate (matches tof_common) for the live peak preview
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = HERE          # this script lives in the A2 folder; recordings save alongside it


def pick_port(explicit: str | None) -> str:
    if explicit:
        return explicit
    ports = serial.tools.list_ports.comports()
    for p in ports:
        dev = p.device.lower()
        if any(x in dev for x in ("usbmodem", "usbserial", "cu.usb", "cu.wchusbserial")):
            print(f"Auto-selected port: {p.device} ({p.description})")
            return p.device
    if not ports:
        sys.exit("No serial ports found. Is the ESP32-C6 plugged in?")
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  {i}  {p.device}  —  {p.description}")
    idx = int(input("Enter port number: "))
    return ports[idx].device


def parse_frame(line: str):
    """Return distances[64] (float, 0 = no reading) or None.

    Accepts two firmware formats, both 64 zones in mm:
      * ``FRAME:v0,...,v63``  — VL53L5CX_Visualizer / A2_Recorder. 0 = no reading,
        trailing '?' = low confidence.
      * ``D,8,v0,...,v63``    — the custom 8x8 firmware currently on the board.
        -1 (and any <=0) = no reading.
    Both are normalized to a 64-vector with 0 for no-reading.
    """
    if line.startswith("FRAME:"):
        parts = line[6:].split(",")
        if len(parts) != N_ZONES:
            return None
        d = np.zeros(N_ZONES, dtype=float)
        for i, p in enumerate(parts):
            p = p.strip()
            if p == "0" or p == "":
                continue
            if p.endswith("?"):
                p = p[:-1]
            try:
                d[i] = float(int(p))
            except ValueError:
                continue
        return d
    if line.startswith("D,"):
        parts = line.split(",")
        if len(parts) != N_ZONES + 2:       # "D", res, then 64 values
            return None
        d = np.zeros(N_ZONES, dtype=float)
        for i, p in enumerate(parts[2:]):
            try:
                v = int(p)
            except ValueError:
                continue
            if v > 0:
                d[i] = float(v)             # -1 / 0 stay 0 (no reading)
        return d
    return None


def peak_distance(frame: np.ndarray) -> float:
    """Per-frame peak-distance scalar: median of gated valid zones (NaN if none)."""
    v = frame[(frame > 0) & (frame <= MAX_RANGE_MM)]
    return float(np.median(v)) if v.size else np.nan


def main():
    ap = argparse.ArgumentParser(description="A2 static ToF recorder (60 s baseline capture).")
    ap.add_argument("--port", default=None, help="Serial port (auto-detected if omitted).")
    ap.add_argument("--baud", type=int, default=BAUD_RATE)
    ap.add_argument("--duration", type=float, default=60.0, help="Recording window in seconds.")
    ap.add_argument("--out", default=None, help="Output CSV path (auto-named under test_results/tof_mounted/a2 if omitted).")
    ap.add_argument("--label", default="static", help="Short tag folded into the auto filename.")
    args = ap.parse_args()

    port = pick_port(args.port)

    if args.out:
        out_path = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    else:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"vl53l5cx_a2_{args.label}_{int(args.duration)}s_{stamp}.csv"
        out_path = os.path.join(DEFAULT_OUT_DIR, fname)

    print(f"Opening {port} @ {args.baud} baud …")
    try:
        ser = serial.Serial(port, args.baud, timeout=2)
    except serial.SerialException as e:
        sys.exit(f"Could not open {port}: {e}\n(Close the Arduino Serial Monitor — the port allows one reader.)")

    csv_file = open(out_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp"] + [f"zone_{i:02d}" for i in range(N_ZONES)])

    print("Connected. Waiting for first frame … (keep the scene fully static — no contact, fixed standoff)")
    n = 0
    peaks = []
    ts_wall = []          # perf_counter timestamps, for effective-rate report
    n_dup = 0             # consecutive byte-identical frames (firmware over-sampling)
    prev_frame = None
    t_start = None
    last_print = 0.0
    try:
        while True:
            raw = ser.readline()
            line = raw.decode("utf-8", errors="ignore").strip()
            frame = parse_frame(line)
            if frame is None:
                continue

            now = time.perf_counter()
            if t_start is None:
                t_start = now
                print(f"Recording {args.duration:.0f} s → {out_path}")
            elapsed = now - t_start
            if elapsed > args.duration:
                break

            ts = datetime.now().isoformat(timespec="milliseconds")
            writer.writerow([ts] + [int(v) for v in frame])
            n += 1
            ts_wall.append(now)
            peaks.append(peak_distance(frame))
            if prev_frame is not None and np.array_equal(frame, prev_frame):
                n_dup += 1                  # identical to previous → not a fresh sensor read
            prev_frame = frame

            if now - last_print >= 1.0:
                pk = np.array(peaks, dtype=float)
                pk = pk[np.isfinite(pk)]
                rate = n / elapsed if elapsed > 0 else 0.0
                mean = pk.mean() if pk.size else float("nan")
                sd = pk.std(ddof=1) if pk.size > 1 else float("nan")
                print(f"  t={elapsed:5.1f}s  frames={n:5d}  ~{rate:4.1f} Hz  "
                      f"peak={mean:6.1f} mm  raw σ≈{sd:4.2f} mm", end="\r", flush=True)
                last_print = now
    except KeyboardInterrupt:
        print("\nInterrupted — finalizing what was recorded.")
    finally:
        csv_file.flush()
        csv_file.close()
        ser.close()

    print()  # clear the \r line
    if n < 2:
        print(f"Only {n} frame(s) captured — check the sensor/firmware. File: {out_path}")
        return

    tw = np.array(ts_wall)
    span = tw[-1] - tw[0]
    eff_hz = (n - 1) / span if span > 0 else float("nan")
    dts = np.diff(tw) * 1000.0  # ms
    pk = np.array(peaks, dtype=float)
    valid = np.isfinite(pk)
    pk_valid = pk[valid]

    print("── A2 recording complete ─────────────────────────────────────")
    print(f"  file            : {out_path}")
    print(f"  frames          : {n}   ({valid.sum()} with a valid peak, {n - valid.sum()} empty)")
    print(f"  duration        : {span:.2f} s")
    print(f"  effective rate  : {eff_hz:.2f} Hz   (frame dt: mean {dts.mean():.1f} ms, "
          f"min {dts.min():.1f}, max {dts.max():.1f})")
    dup_pct = 100.0 * n_dup / (n - 1) if n > 1 else 0.0
    print(f"  duplicate frames: {n_dup}/{n - 1} ({dup_pct:.1f}%)"
          + ("  [over-sampling: duplicate frames deflate jitter; flash the 15 Hz A2 sketch]"
             if dup_pct > 5.0 else "  [fresh reads]"))
    if pk_valid.size > 1:
        print(f"  raw peak-dist   : mean {pk_valid.mean():.2f} mm   "
              f"σ (raw jitter) {pk_valid.std(ddof=1):.3f} mm   "
              f"range {pk_valid.min():.1f}–{pk_valid.max():.1f} mm")
    print("  analyze: python3 data_collection/analyze_a2_kalman.py "
          f"\"{out_path}\" --q <q> --r <R>")
    if not (10.0 <= eff_hz <= 16.0):
        print(f"  NOTE: effective rate {eff_hz:.1f} Hz. At 8x8 the sensor tops out ~15 Hz; "
              "flash VL53L5CX_A2_Recorder (15 Hz) to run at the 64-zone ceiling.")


if __name__ == "__main__":
    main()
