"""Compile + upload the vl53l5cx_stream sketch to the ESP32-C6 with arduino-cli.

    python3 flash.py [port]        port defaults to auto-detect (/dev/cu.usbmodem*)

Needs:  brew install arduino-cli
        arduino-cli core install esp32:esp32
The SparkFun VL53L5CX library is picked up from ~/Documents/Arduino/libraries.
No arduino-cli?  Open firmware/vl53l5cx_stream/vl53l5cx_stream.ino in the
Arduino IDE, select the ESP32C6 board + port, and press Upload.
"""
import glob
import os
import subprocess
import sys

BOARD = "esp32:esp32:esp32c6:CDCOnBoot=cdc"   # CDC on boot: Serial -> native USB port
HERE = os.path.dirname(os.path.abspath(__file__))
SKETCH = os.path.join(HERE, "vl53l5cx_stream")


def find_port():
    cands = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not cands:
        raise SystemExit("No /dev/cu.usbmodem* port found — plug in the ESP32-C6 "
                         "(and close the Serial Monitor), or pass the port explicitly.")
    return cands[0]


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_port()
    print(f"Compiling {SKETCH} ...")
    subprocess.run(["arduino-cli", "compile", "--fqbn", BOARD, SKETCH], check=True)
    print(f"Uploading to {port} ...")
    subprocess.run(["arduino-cli", "upload", "-p", port, "--fqbn", BOARD, SKETCH],
                   check=True)
    print("Done. Sanity-check the stream with:\n"
          "  python3 ../data_collection/A3_proximity/tof_logger.py")


if __name__ == "__main__":
    main()
