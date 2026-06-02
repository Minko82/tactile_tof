"""
VL53L5CX 3D Touch Visualizer
------------------------------
Reads FRAME: lines from the ESP32-C6 and renders a live 3D bar chart.

Install dependencies first:
    pip install pyserial matplotlib numpy

Then run:
    python3 visualizer.py

Calibration:
    Press the "Calibrate" button while the dome is resting (not pressed).
    The visualizer averages 30 frames and saves per-zone baselines to
    calibration.json. Thresholds are loaded automatically on next launch.
"""

import csv
import json
import os
import sys
import threading
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as mwidgets
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("Missing dependency: run  pip install pyserial  then try again.")

# ── Config ────────────────────────────────────────────────────────────────────
BAUD_RATE        = 115200
TOUCH_MAX_MM     = 15     # fallback per-zone threshold (mm) when no calibration.json
MAX_RANGE        = 800    # mm — anything beyond this displays at full depth
INTERVAL_MS      = 80     # animation refresh (ms)
N_CAL_FRAMES     = 30     # frames to average during calibration (~2.4 s)
DOME_HEIGHT_MM   = 9.0    # physical dome height — fallback threshold for zones that
                          # return 0 during calibration (IR-transparent silicone)
TOUCH_CEILING_MM = 25.0   # hard cap on any per-zone calibrated threshold. Prevents
                          # transparent zones (which read a far background during
                          # calibration) from setting an inflated touch threshold.
DOME_FLEX_MM     = 20.0  # hollow dome only: when a zone is next to a confirmed-touch
                          # zone and its reading went up by at most this many mm above
                          # its baseline, it's dome-wall flex — promote it to touch.
                          # Raise if flexing zones still show blue; lower if proximity
                          # bleeds into touch.

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "calibration.json")

# ── CSV logging ───────────────────────────────────────────────────────────────
_csv_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
os.makedirs(_csv_dir, exist_ok=True)
_csv_name = datetime.now().strftime("readings_%Y-%m-%d_%H-%M-%S.csv")
_csv_path = os.path.join(_csv_dir, _csv_name)
_csv_file = open(_csv_path, "w", newline="")
_csv_writer = csv.writer(_csv_file)
_csv_writer.writerow(["timestamp"] + [f"zone_{i:02d}" for i in range(64)])
print(f"Logging sensor data to {_csv_path}")

# ── Per-zone calibration ──────────────────────────────────────────────────────
def load_calibration() -> np.ndarray:
    """Return per-zone touch thresholds (mm). Falls back to TOUCH_MAX_MM."""
    arr = np.full(64, TOUCH_MAX_MM, dtype=float)
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE) as f:
                data = json.load(f)
            baseline = np.array(data["baseline_mm"], dtype=float)
            if baseline.shape == (64,):
                arr = baseline
                print(f"Loaded calibration from {CALIBRATION_FILE}")
            else:
                print("Calibration file has wrong shape; using default.")
        except Exception as e:
            print(f"Could not load calibration ({e}); using default.")
    return arr

touch_max_mm = load_calibration()   # shape (64,) — per-zone threshold array

# ── Serial port auto-detection ────────────────────────────────────────────────
def pick_port():
    ports = serial.tools.list_ports.comports()
    # prefer USB modem / serial ports (ESP32-C6 shows as usbmodem on macOS)
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

# ── Frame parsing ─────────────────────────────────────────────────────────────
def parse_frame(line: str):
    """Return (distances[64], poor_mask[64]) or None.

    poor_mask[i] is True when the zone has a low-confidence reading (Arduino
    appended '?' to the value — status 4 or 10 rather than 5/6/9).
    """
    if not line.startswith("FRAME:"):
        return None
    parts = line[6:].split(",")
    if len(parts) != 64:
        return None
    distances = np.zeros(64, dtype=float)
    poor_mask = np.zeros(64, dtype=bool)
    for i, p in enumerate(parts):
        p = p.strip()
        if p == "0":
            continue
        if p.endswith("?"):
            try:
                distances[i] = int(p[:-1])
                poor_mask[i] = True
            except ValueError:
                continue
        else:
            try:
                distances[i] = int(p)
            except ValueError:
                continue
    return distances, poor_mask

# ── Shared state ──────────────────────────────────────────────────────────────
_lock      = threading.Lock()
_latest    = None   # np.ndarray shape (64,) or None
_connected = False

# ── Calibration state ─────────────────────────────────────────────────────────
_cal_lock        = threading.Lock()
_cal_accumulator = None   # np.ndarray (N_CAL_FRAMES, 64) while recording
_cal_count       = 0
_cal_active      = False
_btn_reset_counter = [0]  # counts down frames after success flash

def serial_thread(port: str):
    global _latest, _connected, _cal_count
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        _connected = True
        print("Connected. Waiting for frames…")
        while True:
            raw  = ser.readline()
            line = raw.decode("utf-8", errors="ignore").strip()
            result = parse_frame(line)
            if result is not None:
                distances, poor_mask = result
                with _lock:
                    _latest = result          # stores (distances, poor_mask)
                with _cal_lock:
                    if _cal_active and _cal_count < N_CAL_FRAMES:
                        _cal_accumulator[_cal_count] = distances
                        _cal_count += 1
                # Write raw distances to CSV (integers, 0 = no reading)
                ts = datetime.now().isoformat(timespec="milliseconds")
                _csv_writer.writerow([ts] + [int(v) for v in distances])
                _csv_file.flush()
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        pass

# ── Color mapping ─────────────────────────────────────────────────────────────
def bar_colors(distances, touch_mask, poor_mask, thresholds):
    colors = []
    for i, (dist, is_touch, is_poor) in enumerate(zip(distances, touch_mask, poor_mask)):
        if dist == 0:
            colors.append((0.05, 0.05, 0.16, 0.5))
        elif is_poor:
            # Yellow — sensor returned a low-confidence status (4 or 10)
            colors.append((1.0, 0.88, 0.0, 0.85))
        elif is_touch:
            # Red — brighter the further below the calibrated baseline (harder press)
            t = max(0.0, 1.0 - (dist / max(thresholds[i], 1.0)))
            colors.append((1.0, 0.08 * (1 - t), 0.0, 1.0))
        else:
            # Blue — proximity; brighter when closer
            t = min(1.0, dist / MAX_RANGE)
            colors.append((0.0, 0.22 * (1 - t), 0.5 + 0.3 * (1 - t), 0.75))
    return colors

# ── Matplotlib setup ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 8.5), facecolor="#07071a")
fig.canvas.manager.set_window_title("VL53L5CX 3D Touch Visualizer")
fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.11)

# 3D bar chart on top, 2D touch heatmap on bottom
from matplotlib.gridspec import GridSpec
gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08,
               left=0.05, right=0.98, top=0.97, bottom=0.13)
ax  = fig.add_subplot(gs[0], projection="3d", facecolor="#07071a")
ax2 = fig.add_subplot(gs[1], facecolor="#07071a")
ax2.set_aspect("equal")
heatmap_im = ax2.imshow(
    np.zeros((8, 8, 3)), origin="upper",
    extent=[-0.5, 7.5, 7.5, -0.5], interpolation="nearest"
)
# Always-visible grid lines over the heatmap cells
for x in np.arange(-0.5, 8, 1):
    ax2.axvline(x, color="#1a1a3a", linewidth=0.8)
for y in np.arange(-0.5, 8, 1):
    ax2.axhline(y, color="#1a1a3a", linewidth=0.8)
ax2.set_xlim(-0.5, 7.5)
ax2.set_ylim(7.5, -0.5)   # origin="upper" so y is flipped
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel("touch", color="#444466", fontsize=9, labelpad=4)
for spine in ax2.spines.values():
    spine.set_edgecolor("#1a1a3a")
touch_label = ax2.text(
    3.5, -1.1, "", ha="center", va="center",
    fontsize=11, fontweight="bold", color="#ff2200", alpha=0
)

# ── Calibration button ────────────────────────────────────────────────────────
ax_btn = fig.add_axes([0.38, 0.02, 0.24, 0.06], facecolor="#1a1a3a")
btn_calibrate = mwidgets.Button(ax_btn, "Calibrate",
                                color="#1a1a3a", hovercolor="#2a2a5a")
btn_calibrate.label.set_color("#8888cc")
btn_calibrate.label.set_fontsize(10)

GRID = 8
cols_idx = np.arange(GRID)
rows_idx = np.arange(GRID)
xpos, ypos = np.meshgrid(cols_idx, rows_idx)
xpos = xpos.flatten().astype(float)
ypos = ypos.flatten().astype(float)
zpos = np.zeros(64)
dx = dy = np.full(64, 0.72)

_demo_t = [0.0]   # mutable for use inside closure

def get_demo_data() -> np.ndarray:
    _demo_t[0] += 0.03
    t = _demo_t[0]
    distances = np.zeros(64)
    for i in range(64):
        r, c = i // 8, i % 8
        d = np.sqrt((c - 3.5) ** 2 + (r - 3.5) ** 2)
        wave = np.sin(t - d * 0.75) * 0.5 + 0.5
        distances[i] = round(15 + wave * 720)
    return distances

def style_axes(any_touch: bool):
    if any_touch:
        bg   = "#1a0404"
        edge = "#2a0808"
        title = "◆  TOUCH DETECTED  ◆"
        col   = "#ff2200"
    else:
        bg   = "#07071a"
        edge = "#111133"
        title = "VL53L5CX · 8×8 Touch Visualizer"
        col   = "#6666cc"

    ax.set_facecolor(bg)
    ax2.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(-0.5, GRID - 0.5)
    ax.set_zlim(0, MAX_RANGE)
    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))
    ax.tick_params(colors="#333355", labelsize=7)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(edge)
    ax.yaxis.pane.set_edgecolor(edge)
    ax.zaxis.pane.set_edgecolor(edge)
    ax.grid(True, color=edge, linewidth=0.4)
    ax.set_zlabel("distance (mm)  taller = farther", color="#333355", fontsize=8, labelpad=6)
    ax.set_title(title, color=col, fontsize=12, fontweight="bold", pad=10)
    # Lock orientation so left=col0, top=row0 — matches the 2D touch grid
    ax.invert_yaxis()
    ax.view_init(elev=28, azim=-60)

def update_heatmap(distances, touch_mask, poor_mask, thresholds):
    """Update the 2D grid — red for touch, yellow for poor-quality, dark otherwise."""
    rgb = np.zeros((8, 8, 3))
    for i in range(64):
        row, col = i // 8, i % 8
        dist     = distances[i]
        is_touch = touch_mask[i]
        is_poor  = poor_mask[i]
        if is_poor and dist > 0:
            rgb[row, col] = [1.0, 0.88, 0.0]   # yellow — low-confidence reading
        elif is_touch and dist > 0:
            t = max(0.0, 1.0 - (dist / max(thresholds[i], 1.0)))
            rgb[row, col] = [1.0, 0.08 * (1 - t), 0.0]
        else:
            rgb[row, col] = [0.09, 0.09, 0.20]

    heatmap_im.set_data(rgb)
    any_touch   = bool(np.any(touch_mask & (distances > 0)))
    touch_count = int(np.sum(touch_mask  & (distances > 0)))

    if any_touch:
        touch_label.set_text(
            f"▼  {touch_count} zone{'s' if touch_count != 1 else ''} touching  ▼")
        touch_label.set_color("#ff2200")
        touch_label.set_alpha(1.0)
    else:
        touch_label.set_alpha(0.0)

    ax2.set_facecolor("#1a0404" if any_touch else "#07071a")

# ── Calibration button callback ───────────────────────────────────────────────
def _start_calibration(event):
    global _cal_accumulator, _cal_count, _cal_active
    if not _connected:
        btn_calibrate.label.set_text("No sensor!")
        btn_calibrate.label.set_color("#ff4444")
        _btn_reset_counter[0] = 15
        return
    with _cal_lock:
        if _cal_active:
            return
        _cal_accumulator = np.zeros((N_CAL_FRAMES, 64), dtype=float)
        _cal_count       = 0
        _cal_active      = True

btn_calibrate.on_clicked(_start_calibration)

def update(_frame_num):
    global touch_max_mm, _cal_active, _cal_count, _cal_accumulator

    # ── Calibration progress / completion ─────────────────────────────────────
    with _cal_lock:
        cal_active = _cal_active
        cal_count  = _cal_count

    if cal_active:
        if cal_count < N_CAL_FRAMES:
            btn_calibrate.label.set_text(f"Recording… ({cal_count}/{N_CAL_FRAMES})")
            btn_calibrate.label.set_color("#ffaa00")
        else:
            with _cal_lock:
                snapshot    = _cal_accumulator.copy()
                _cal_active = False

            new_thresholds = np.empty(64)
            for zone in range(64):
                col   = snapshot[:, zone]
                valid = col[col > 0]
                if len(valid) > 0:
                    # Use the zone's own resting distance as its touch threshold,
                    # capped at TOUCH_CEILING_MM. The cap prevents transparent
                    # zones that read a 300-400 mm background from treating a
                    # hovering hand as a touch.
                    new_thresholds[zone] = min(float(valid.mean()), TOUCH_CEILING_MM)
                else:
                    # Zone returned 0 throughout calibration (IR-transparent).
                    # Fall back to the physical dome height.
                    new_thresholds[zone] = DOME_HEIGHT_MM

            touch_max_mm = new_thresholds

            payload = {
                "baseline_mm": touch_max_mm.tolist(),
                "captured_at": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                with open(CALIBRATION_FILE, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"Calibration saved → {CALIBRATION_FILE}")
            except OSError as e:
                print(f"Could not save calibration: {e}")

            btn_calibrate.label.set_text("Calibrate ✓")
            btn_calibrate.label.set_color("#44cc44")
            _btn_reset_counter[0] = 12

    if _btn_reset_counter[0] > 0:
        _btn_reset_counter[0] -= 1
        if _btn_reset_counter[0] == 0:
            btn_calibrate.label.set_text("Calibrate")
            btn_calibrate.label.set_color("#8888cc")

    ax.cla()

    with _lock:
        data = _latest

    if data is None:
        distances = get_demo_data()
        poor_mask = np.zeros(64, dtype=bool)
    else:
        distances, poor_mask = data

    # Primary touch: reading dropped below the calibrated resting baseline.
    # Poor-quality readings (yellow) are excluded from touch.
    good_mask     = ~poor_mask
    primary_touch = good_mask & (distances > 0) & (distances <= touch_max_mm)

    # Hollow-dome flex: when the dome top is pressed, the walls flex outward —
    # zones adjacent to a confirmed-touch zone may read *longer* than their
    # baseline (they now see more air inside the cavity). Promote those zones
    # to touch if their upward deviation is within DOME_FLEX_MM.
    primary_grid  = primary_touch.reshape(8, 8)
    padded        = np.pad(primary_grid, 1, mode="constant", constant_values=False)
    has_touch_neighbour = (
        padded[:-2, 1:-1] | padded[2:, 1:-1] |   # above / below
        padded[1:-1, :-2] | padded[1:-1, 2:]       # left / right
    ).flatten()
    flex_touch = (
        good_mask &
        (distances > touch_max_mm) &
        (distances <= touch_max_mm + DOME_FLEX_MM) &
        has_touch_neighbour
    )

    touch_mask = primary_touch | flex_touch

    # For touch zones, cap the bar height at the calibrated threshold so they
    # always appear as SHORT bars in the 3D view — consistent with the 2D heatmap.
    # Flex-touch zones have readings above their threshold (dome wall flexed
    # outward) so without this cap they'd render as tall "proximity-looking" bars.
    raw_heights = np.where(distances > 0, np.minimum(distances, MAX_RANGE), 0.0)
    heights     = np.where(touch_mask,
                           np.minimum(raw_heights, touch_max_mm),
                           raw_heights)
    colors    = bar_colors(distances, touch_mask, poor_mask, touch_max_mm)
    any_touch = bool(np.any(touch_mask))

    ax.bar3d(xpos, ypos, zpos, dx, dy, np.maximum(heights, 1.0),
             color=colors, shade=True, zsort="average")
    style_axes(any_touch)
    update_heatmap(distances, touch_mask, poor_mask, touch_max_mm)

ani = animation.FuncAnimation(fig, update, interval=INTERVAL_MS, cache_frame_data=False)

# ── Start serial thread then show plot ────────────────────────────────────────
if __name__ == "__main__":
    port = pick_port()
    t = threading.Thread(target=serial_thread, args=(port,), daemon=True)
    t.start()
    print("Close the window or press Ctrl+C to quit.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        _csv_file.close()
        print(f"CSV saved → {_csv_path}")
