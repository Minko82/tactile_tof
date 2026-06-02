# Data Collection — VL53L5CX Tactile Sensor

This guide walks through flashing the sensor firmware and running the live visualizer to collect touch data.

---

## Prerequisites

**Python dependencies** (run once):
```bash
pip install pyserial matplotlib numpy
```

**Arduino setup** (run once):
1. Install the [SparkFun VL53L5CX library](https://github.com/sparkfun/SparkFun_VL53L5CX_Arduino_Library) via the Arduino Library Manager.
2. Inside the installed library, open `src/platform.h` and set:
   ```c
   #define NB_TARGET_PER_ZONE 1U
   ```
   This is required — without it the sensor outputs multiple targets per zone and the frame parser will break.
3. Install ESP32 board support if you haven't already:
   ```bash
   arduino-cli core install esp32:esp32
   ```

**Hardware:**
- ESP32-C6 connected via USB
- VL53L5CX sensor wired over Qwiic (SDA = GPIO 6, SCL = GPIO 7)

---

## Step 1 — Flash the Firmware

Open `data_collection/VL53L5CX_Visualizer/VL53L5CX_Visualizer.ino` in the Arduino IDE.

- **Board:** `ESP32C6 Dev Module` (under *esp32 by Espressif*)
- **Port:** whichever `usbmodem` / `usbserial` port appears when the board is plugged in
- **Baud rate:** 115200
- **USB CDC on Boot:** `Enabled` ← critical — without this the board won't appear as a serial port over USB

> To set USB CDC: with the ESP32-C6 selected as your board, go to **Tools → USB CDC On Boot → Enabled**.

Click **Upload**. Once the upload finishes, open the Serial Monitor briefly to confirm you see lines like:
```
VL53L5CX Visualizer starting...
READY ranging_hz=10
FRAME:342,18,200,...
```

> ⚠️ **Close the Arduino IDE before moving to Step 2.**
> The serial port can only be held by one process at a time — if Arduino still has it open, the Python visualizer will fail to connect.

---

## Step 2 — Run the Visualizer

From the **root of the repo**, run:
```bash
python3 visualizer.py
```

The script auto-detects the ESP32's serial port. If multiple ports are found, it will list them and ask you to pick one.

**What you'll see:**
- A **3D bar chart** (top) and a **2D grid** (bottom) updating at ~10 Hz.
- While waiting for the sensor to connect, an animated demo plays automatically.
- Color coding:

| Color  | Meaning |
|--------|---------|
| 🔵 Blue  | Zone is active — bar height = distance in mm (taller = farther) |
| 🔴 Red   | Touch detected — reading dropped below the calibrated baseline |
| 🟡 Yellow | Low-confidence reading from the sensor (status 4 or 10) |
| ⚫ Dark  | No return signal |

---

## Step 3 — Calibrate

Calibration sets a per-zone resting baseline so the visualizer knows when each zone is being pressed.

1. Rest the dome on a flat surface **without touching it**.
2. Click the **Calibrate** button in the visualizer window.
3. The button will show `Recording… (n/30)` while it averages 30 frames (~3 seconds).
4. When done it shows `Calibrate ✓` and saves thresholds to `calibration.json` in the repo root.

Calibration is loaded automatically every time `visualizer.py` starts. Re-calibrate whenever the sensor is repositioned or the dome is swapped.

---

## Recorded Data

Every run of `visualizer.py` automatically writes a timestamped CSV to `test_results/` in the repo root. **After each test session, move the CSV into a clearly named subfolder** describing the test condition so results stay organized. For example:

```
test_results/
├── Baseline/
│   └── readings_2025-06-02_14-32-10.csv
├── EcoFlex_Silicone/
│   └── readings_2025-06-02_15-10-44.csv
└── ...
```

Each row is one frame:
```
timestamp, zone_00, zone_01, ..., zone_63
```
Values are raw distances in mm; `0` means no valid return for that zone.

The file is finalized and closed cleanly when you close the visualizer window or press `Ctrl+C`.
