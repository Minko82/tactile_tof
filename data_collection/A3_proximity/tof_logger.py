"""
8x8 ToF sensor reader (VL53L5CX on an ESP32-C6, over USB serial).

Firmware: firmware/vl53l5cx_esp32c6/ streams one line per frame:
    D,8,<d0>,<d1>,...,<d63>     distances in mm, -1 = no valid target
    lines starting with '#' are human-readable status messages

robot.py's `record` command uses open_sensor() + read_frame() from here.
Run standalone to sanity-check the sensor:  python3 tof_logger.py
"""
import time

# ===========================================================================
#  HARDWARE  — VL53L5CX (8x8) on an ESP32-C6 over USB serial
# ===========================================================================
SERIAL_PORT = None          # None = auto-detect the ESP32-C6; or hard-set e.g. "/dev/cu.usbmodem101"
BAUD        = 115200
GRID        = 8
N_ZONES     = GRID * GRID   # 64


def _find_port():
    """Auto-detect the ESP32-C6's serial port."""
    from serial.tools import list_ports
    cands = []
    for p in list_ports.comports():
        blob = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
        if any(k in blob for k in ("usbmodem", "usbserial", "wchusbserial",
                                   "esp32", "cp210", "ch340", "espressif", "jtag")):
            cands.append(p.device)
    cands.sort(key=lambda d: ("usbmodem" not in d, "cu." in d))  # prefer native USB CDC, tty over cu
    return cands[0] if cands else None


def open_sensor():
    """Open the serial connection to the ESP32-C6 streaming the ToF frames."""
    import serial                                 # pip install pyserial
    port = SERIAL_PORT or _find_port()
    if port is None:
        raise RuntimeError("No ESP32-C6 serial port found. Plug it in (and close "
                           "the Arduino Serial Monitor / browser visualizer), or "
                           "set SERIAL_PORT.  Find it with:  ls /dev/tty.usb*")
    dev = serial.Serial(port, BAUD, timeout=1.0)
    time.sleep(2.0)                               # let the board boot / USB CDC settle
    dev.reset_input_buffer()
    print(f"  ToF sensor on {port}")
    return dev


def read_frame(dev):
    """Parse one 'D,8,d0..d63' line -> list of 64 distances (mm, -1 = no target).
    Returns None for status ('#...') lines or malformed/partial lines."""
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
    if len(vals) != N_ZONES:
        return None
    return vals


if __name__ == "__main__":
    # Standalone sanity check: read and print a few real frames.
    dev = open_sensor()
    print("Reading 10 frames (center zone shown):")
    got = 0
    while got < 10:
        f = read_frame(dev)
        if f is None:
            continue
        got += 1
        valid = [v for v in f if v > 0]
        closest = min(valid) if valid else "n/a"
        print(f"  frame {got:2d}: center~{f[27]} mm   closest={closest} mm")
    dev.close()
