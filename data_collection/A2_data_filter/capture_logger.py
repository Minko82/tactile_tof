"""
Capture logger for tactile force calibration (see CAPTURE_PROTOCOL.md).

Records the ToF stream to tof_log.csv while you enter force labels that are
timestamped on the SAME clock into force_log.csv -- so tactile_force.py can line
them up exactly.  Same ESP32-C6 serial front-end as a2_record5.py.

Usage:
    ~/ur5-env/bin/python3 capture_logger.py normal_cal/round_1
    # ToF starts streaming.  At each load step, type the label and press Enter:
    #     2.0                 -> F_n = 2.0 N (normal cal)
    #     preload=2 0.5 90    -> F_s = 0.5 N at 90 deg, normal preload 2 N (shear cal)
    #     0                   -> release / zero
    #     q                   -> stop
Every typed line is stamped; values hold until the next line.  For distance_cal
type the plate gap in mm the same way (it lands in the F_n column -- fine, or
rename the header).

This needs the physical sensor; it is a ready-to-run skeleton, not exercised here.
"""
import csv, os, sys, threading, time

SERIAL_PORT = None            # None = auto-detect (see a2_record5.py)
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
        raise RuntimeError("No ESP32-C6 serial port found (ls /dev/tty.usb*).")
    dev = serial.Serial(port, BAUD, timeout=1.0)
    time.sleep(2.0); dev.reset_input_buffer()
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


def parse_label(line):
    """Normal/gap label: '2.0' -> F_n = 2.0.
    Shear label (preload keyword): 'preload=2 0.5 90' -> F_s = 0.5 N at 90 deg,
    normal preload 2 N."""
    toks = line.split(); preload = None
    for tk in list(toks):
        if tk.startswith("preload="):
            preload = float(tk.split("=", 1)[1]); toks.remove(tk)
    try:
        nums = [float(x) for x in toks]
    except ValueError:
        return None
    if preload is not None:                        # shear: F_s [dir_deg]
        if not nums:
            return None
        d = nums[1] if len(nums) > 1 else 0.0
        return dict(F_n_N=preload, F_s_N=nums[0], shear_dir_deg=d, F_n_preload_N=preload)
    if len(nums) == 1:                             # normal / gap: F_n
        return dict(F_n_N=nums[0], F_s_N=0.0, shear_dir_deg=0.0, F_n_preload_N=0.0)
    return None


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    dev = open_sensor()
    t0 = time.perf_counter()
    stop = threading.Event()

    def stream_tof():
        with open(os.path.join(outdir, "tof_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s"] + [f"z{i}" for i in range(N_ZONES)])
            n = 0
            while not stop.is_set():
                fr = read_frame(dev)
                if fr is None:
                    continue
                w.writerow([f"{time.perf_counter()-t0:.4f}"] + fr)
                n += 1
                if n % 150 == 0:
                    print(f"    ...{n} frames", flush=True)

    th = threading.Thread(target=stream_tof, daemon=True); th.start()
    print("ToF streaming. Type a load label + Enter at each step ('q' to stop):")
    with open(os.path.join(outdir, "force_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "F_n_N", "F_s_N", "shear_dir_deg", "F_n_preload_N", "note"])
        try:
            for line in sys.stdin:
                line = line.strip()
                if line.lower() in ("q", "quit", "exit"):
                    break
                lab = parse_label(line)
                if lab is None:
                    print("    (couldn't parse; e.g. '2.0' or 'preload=2 0.5 90')")
                    continue
                w.writerow([f"{time.perf_counter()-t0:.4f}", lab["F_n_N"], lab["F_s_N"],
                            lab["shear_dir_deg"], lab["F_n_preload_N"], ""])
                f.flush()
                print(f"    logged {lab}")
        finally:
            stop.set(); th.join(timeout=2.0); dev.close()
    print(f"Done -> {outdir}/tof_log.csv + force_log.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: capture_logger.py <output_dir>  (e.g. normal_cal/round_1)")
        sys.exit(1)
    main(sys.argv[1])
