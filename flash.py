import subprocess
import time
import os

# --- CONFIGURATION ---
arduino_cli_path = "arduino-cli"

# Get absolute path to the current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Point to the SKETCH folder
sketch_dir = os.path.join(base_dir, "firmware", "basic_sensor")
# Point to the SparkFun VL53L5CX library folder
sparkfun_lib_path = os.path.join(base_dir, "SparkFun_VL53L5CX_Arduino_Library")

arduino_port = "/dev/cu.usbmodem1101" 
arduino_board = "esp32:esp32:esp32c6"

# --- COMMANDS ---
compile_cmd = [
    arduino_cli_path, 
    "compile", 
    "--fqbn", arduino_board, 
    "--libraries", sparkfun_lib_path,
    sketch_dir
]

upload_cmd = [
    arduino_cli_path, 
    "upload", 
    "-p", arduino_port, 
    "--fqbn", arduino_board, 
    sketch_dir
]

print(f"Compiling with custom library from {sparkfun_lib_path}...")
subprocess.run(compile_cmd, check=True)

print(f"Uploading to {arduino_board}...")
subprocess.run(upload_cmd, check=True)

print("Upload complete. Launching Python Viz...")
time.sleep(3) 
subprocess.run(["python3", os.path.join(base_dir, "vis", "tof_matrix_viz.py")], check=True)