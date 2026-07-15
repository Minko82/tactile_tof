"""
Record five 60-second runs of the STATIC ToF sensor into  <this folder>/raw_data/
  -> ToF/dataset/A2/raw_data/raw_1.csv .. raw_5.csv

Static test: sensor fixed, no target motion, fixed standoff, consistent ambient
light. No robot. Self-contained (no imports of tof_logger / a2_record), so it runs
regardless of how the folders are arranged.

Uses the same ESP32-C6 serial port as robot.py, so DON'T run this while a robot
`record`/`steps` batch is going -- the port has a single owner. Takes ~5 min
(5 x 60 s + short pauses). Re-running adds raw_6.csv, raw_7.csv, ... (never overwrites).

Usage:  python3 a2_record5.py
Then analyse each run with UR5/a2_kalman.py, e.g.:
    python3 <path-to>/a2_kalman.py raw_data/raw_1.csv
"""
import time, csv, os

N_RUNS         = 5
RECORD_SECONDS = 60.0
PAUSE_S        = 3.0            # settle pause between runs
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")

# Sensor: VL53L5CX 8x8 on ESP32-C6, "D,8,d0..d63" lines over USB serial
SERIAL_PORT = None             # None = auto-detect; or hard-set e.g. "/dev/cu.usbmodem101"
BAUD        = 115200
N_ZONES     = 64


def _find_port():
    from serial.tools import list_ports
    cands = []
    for p in list_ports.comports():
        blob = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
        if any(k in blob for k in ("usbmodem", "usbserial", "wchusbserial",
                                   "esp32", "cp210", "ch340", "espressif", "jtag")):
            cands.append(p.device)
    cands.sort(key=lambda d: ("usbmodem" not in d, "cu." in d))
    return cands[0] if cands else None


def open_sensor():
    import serial
    port = SERIAL_PORT or _find_port()
    if port is None:
        raise RuntimeError("No ESP32-C6 serial port found. Plug it in / close the "
                           "Serial Monitor & visualizer.  Find it: ls /dev/tty.usb*")
    dev = serial.Serial(port, BAUD, timeout=1.0)
    time.sleep(2.0)
    dev.reset_input_buffer()
    print(f"  ToF sensor on {port}")
    return dev


def read_frame(dev):
    raw = dev.readline().decode(errors="ignore").strip()
    if not raw or raw.startswith("#"):
        return None
    parts = raw.split(",")
    if len(parts) < 3 or parts[0] != "D":
        return None
    try:
        vals = [int(x) for x in parts[2:]]
    except ValueError:
        return None
    return vals if len(vals) == N_ZONES else None


def record_one(dev, path):
    t0 = time.perf_counter()
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + [f"z{i}" for i in range(N_ZONES)])
        while time.perf_counter() - t0 < RECORD_SECONDS:
            frame = read_frame(dev)
            if frame is None:
                continue
            t = time.perf_counter() - t0
            w.writerow([f"{t:.4f}"] + frame)
            rows += 1
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start = 1                                    # continue past any existing raw_*.csv
    while os.path.exists(os.path.join(OUT_DIR, f"raw_{start}.csv")):
        start += 1

    dev = open_sensor()
    print(f"Recording {N_RUNS} x {RECORD_SECONDS:.0f}s  ->  {OUT_DIR}")
    try:
        for k in range(N_RUNS):
            n = start + k
            path = os.path.join(OUT_DIR, f"raw_{n}.csv")
            print(f"  run {k + 1}/{N_RUNS}: raw_{n}.csv  -- keep sensor & target still")
            rows = record_one(dev, path)
            print(f"    {rows} frames (~{rows / RECORD_SECONDS:.1f} Hz)")
            if k < N_RUNS - 1:
                time.sleep(PAUSE_S)
    finally:
        dev.close()
    print(f"Done: {N_RUNS} runs in {OUT_DIR}")


if __name__ == "__main__":
    main()
