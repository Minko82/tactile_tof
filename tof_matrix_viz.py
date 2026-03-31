# Source - https://stackoverflow.com/a/62411517
# Posted by Zephyr, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-30, License - CC BY-SA 4.0


import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re
import argparse
import os
import pandas as pd
from datetime import datetime
parser = argparse.ArgumentParser(description="visualize_record_touchiq")

parser.add_argument("--port", type=str, default=None, help="port name; ex: /dev/ttyACM0 or like /dev/cu.usbmodem101")
parser.add_argument("--baud", type=int, default=115200, help="baud rate, defaults to 115200")
parser.add_argument("--max_distance", type=int, default=2000, help="max distance in mm for color scale")
parser.add_argument("--record_path", type=str, default=None, help="path to record csv data to. todo: docuemnt csv formatting")


args_cli = parser.parse_args()
SERIAL_PORT = args_cli.port
BAUD_RATE = args_cli.baud
MAX_DISTANCE_MM = args_cli.max_distance
RECORD_PATH = args_cli.record_path
if not SERIAL_PORT:
    raise AssertionError("you must provide a port val.; for example, --port /dev/ttyACM0")
else:
    assert(os.path.exists(str(SERIAL_PORT))), "port isn't showing up on file system"
if not RECORD_PATH:
   pass 
else:
    if not os.path.exists(RECORD_PATH):
        f = open(RECORD_PATH, "w")

new_df = pd.DataFrame(columns=['time_stamp', 'data'])

# ---------------- CONFIGURATION ---------------- #
# CHANGE THIS to match your Arduino's port
#SERIAL_PORT = '/dev/cu.usbmodem101'
#BAUD_RATE = 115200
# Set the max distance (in mm) for the color scale
#MAX_DISTANCE_MM = 2000 
# ----------------------------------------------- #

# Initialize Serial Connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error connecting to serial port: {e}")
    exit()

fig, ax = plt.subplots()
data_matrix = np.zeros((8, 8))
im = ax.imshow(data_matrix, cmap='viridis', vmin=0, vmax=MAX_DISTANCE_MM)
plt.colorbar(im, label='Distance (mm)')
ax.set_title("VL53L5CX 8x8 Matrix")
text_annotations = [[ax.text(j, i, '', ha="center", va="center", color="w", fontsize=6) 
                     for j in range(8)] for i in range(8)]
def update(frame):
    global data_matrix, new_df
    raw_lines = []
    # We need to read enough lines to form a frame. 
    # Your Arduino code outputs 8 rows + 1 empty line per frame.
    # We read until the buffer is clear or we have enough data.
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            # Only process lines that contain data (look for digits)
            print(f"Raw data: {line}")
            if len(line) > 0 and line[0].isdigit():
                # Split by tab or space
                parts = re.split(r'\s+', line)
                # Filter out empty strings and convert to int
                nums = [int(p) for p in parts if p.isdigit()]
                
                # If we parsed a row correctly (should be 8 numbers)
                if len(nums) == 8:
                    raw_lines.append(nums)
            # If we have collected 8 rows, we have a full frame
            if len(raw_lines) == 8:
                # Update the matrix
                data_matrix = np.array(raw_lines)
                if RECORD_PATH:
                    new_df = pd.concat([new_df, pd.DataFrame([[datetime.now(), data_matrix]], columns=['time_stamp', 'data'])], ignore_index=True)
                    new_df.to_csv(RECORD_PATH, index=False)
                # Update the image data
                im.set_data(data_matrix)
                # Update the text numbers inside the squares
                for i in range(8):
                    for j in range(8):
                        val = data_matrix[i, j]
                        text_annotations[i][j].set_text(str(val))
                        # Change text color based on brightness for readability
                        text_annotations[i][j].set_color('black' if val > MAX_DISTANCE_MM/2 else 'white')
                # Clear buffer for the next frame
                raw_lines = []
                # Flush input to avoid lag
                ser.reset_input_buffer()
                return [im] + [t for row in text_annotations for t in row]

        except ValueError:
            pass # Ignore malformed lines
        except Exception as e:
            print(f"Error: {e}")

    return [im]

# Run the animation
ani = FuncAnimation(fig, update, interval=50, blit=False) # 50ms refresh rate
plt.show()

ser.close()
