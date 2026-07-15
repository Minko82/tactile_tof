# UR5e + 8×8 ToF — control & data capture

Drive a **UR5e** from the Mac over Ethernet (pure Python sockets, no external robot
library) and record the arm's end-effector pose together with a **VL53L5CX 8×8
time-of-flight** sensor, frame-locked into CSV files.

## Files

| File | What it is |
|------|------------|
| `robot.py` | Everything: connect to the robot, move it, run the recording experiments. All commands below live here. |
| `tof_logger.py` | The ToF sensor reader (opens the ESP32-C6 serial port, parses frames). Used by `robot.py`; can also be run standalone to sanity-check the sensor. |
| `A3/` | Output. Each recording run lands in `A3/smooth/<n>/` or `A3/steps/<n>/`. |

---

## One-time setup

1. **Network (static IPs, already configured on the hardware):**
   - Robot: `192.168.1.10` — Mac Ethernet dongle: `192.168.1.20` (mask `255.255.255.0`).
   - Test it: `ping 192.168.1.10` should give steady replies.
   - *Never* let it fall back to a `169.254.x.x` link-local address — it drops constantly.
2. **Python env:**
   ```bash
   source ~/ur5-env/bin/activate      # every new terminal
   pip install pyserial               # once (for the sensor)
   ```
3. **Sensor:** plug in the ESP32-C6 (flashed with `firmware/vl53l5cx_esp32c6/`).
   **Close** the Arduino Serial Monitor and any browser visualizer first — only one
   program can hold the serial port.
4. **Robot must be in Remote Control mode** (pendant, top-right toggle) for any motion.
5. **Keep the Mac awake** during long runs: `caffeinate -i` in a spare terminal tab.

---

## Commands

```
python3 robot.py <command> [args]
```

### Data-capture experiments (the main event)

| Command | What it does |
|---------|--------------|
| `python3 robot.py record [N]` | Run **N smooth rounds**. Each round: go to the top, hold 3 s, glide **down** to 12.5 mm above the table, hold 3 s, glide back **up**, cut. Saved to `A3/smooth/<n>/`. Default N = 1. |
| `python3 robot.py steps [N]` | Same experiment but the down/up motion happens in **4 equal ~151 mm steps**, holding 3 s at each step. Saved to `A3/steps/<n>/`. Default N = 1. |

Examples:
```bash
python3 robot.py record 100     # 100 smooth rounds   (~1 hr)
python3 robot.py steps 100      # 100 stepped rounds  (~1.7 hr)
python3 robot.py record 1       # a single round, to eyeball it first
```

Each round writes two **frame-locked** CSVs (one robot row per sensor frame, same
timestamp): `robot_log.csv` and `tof_log.csv`. Run numbers auto-increment, so you can
run in batches without overwriting.

### Setup / one-shot helpers

| Command | What it does |
|---------|--------------|
| `python3 robot.py` (no args) | Go to the perpendicular init spot, then descend to 12.5 mm above the table — **once**, no recording. Prints start & final pose. |
| `python3 robot.py up` | Rise to the perpendicular, sensor-straight init spot and stop there. |
| `python3 robot.py pose` | Print the current end-effector pose (base frame). |
| `python3 robot.py status` | Print robot mode + safety status (e.g. `RUNNING` / `NORMAL`). |
| `python3 robot.py table` | Print `TABLE_Z_MM` — freedrive the sensor down to just touch the table first, then run this and paste the number into the settings. |
| `python3 robot.py wrist 45` | Spin wrist 3 by N degrees (used once to straighten the sensor; now baked in). |

Standalone sensor check (no robot):
```bash
python3 tof_logger.py           # prints 10 live frames, confirms the sensor works
```

---

## What one recording round does

```
position at init (NOT recorded)
  └─ recording starts ──────────────────────────────────────────────┐
     hold 3 s at top                                                 │
     DOWN to 12.5 mm above table   (smooth glide, or 4 stepped hops) │  saved to
     hold 3 s at bottom                                              │  robot_log.csv
     UP back to top                (smooth glide, or 4 stepped hops) │  + tof_log.csv
  └─ cut ────────────────────────────────────────────────────────────┘
```

- **`record`**: down and up are one continuous slow glide (50 mm/s).
- **`steps`**: down and up move in 4 equal ~151 mm hops, holding 3 s at each
  (stops at ≈ 323.6 → 172.1 → 20.6 → −130.8 mm, then reversed).

### Output CSV format

`A3/<smooth|steps>/<n>/`
- **`robot_log.csv`** — `time_s, x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad`
  (position in mm, orientation as an axis-angle rotation vector in radians, base frame)
- **`tof_log.csv`** — `time_s, z0 … z63` (64 zone distances in mm; `-1` = no valid target)

The two files share the same `time_s` values row-for-row, so they line up directly.

---

## Tunable settings (top of `robot.py`)

| Setting | Current | Meaning |
|---------|---------|---------|
| `CLEARANCE_MM` | `12.5` | How far above the table the arm stops (bottom of the sweep). |
| `HOME_Z_MM` | `475.0` | The init/top height each round starts from. |
| `HOLD_SECONDS` | `3.0` | Hold at the top and at the bottom. |
| `DOWN_ACC, DOWN_VEL` | `0.25, 0.05` | Sweep acceleration / speed (m/s² , m/s). |
| `N_STEPS` | `4` | `steps` mode: number of equal steps each way. |
| `STEP_DWELL_S` | `3.0` | `steps` mode: hold at each step. |
| `TABLE_Z_MM` | `-143.30` | Measured table height (from `robot.py table`). |
| `WRIST_OFFSET_DEG` | `45.0` | Wrist rotation that straightens the sensor (baked into the init orientation). |
| `STREAM_HZ` | `60` | Internal pose stream rate; the logged rate is frame-locked to the sensor (~29 Hz). |
| `BATCH_PAUSE_S` | `3.0` | Pause between rounds in a batch. |

---

## Safety

- Robot in **Remote Control**; keep a hand near the **E-stop**, especially on the first
  round after any change — the arm bottoms out only 12.5 mm above the table.
- The measured `TABLE_Z_MM` is what keeps that clearance honest. If the table or the
  robot base moves, re-run `python3 robot.py table`.
- Don't run two robot commands at once — a batch (`record`/`steps`) owns the robot and
  the callback port until it finishes.
