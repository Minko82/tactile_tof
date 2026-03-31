# tof_sensor_setup

1. Clone the VL53L5CX library:
   ```bash
   git clone https://github.com/sparkfun/SparkFun_VL53L5CX_Arduino_Library.git
   ```

2. Install Arduino CLI:
   ```bash
   sudo snap install arduino-cli
   ```

   or for mac:
   ```bash
   brew install arduino-cli
   ```

4. Install ESP32 board support:
   ```bash
   # For ESP32:
   arduino-cli core update-index
   arduino-cli core install esp32:esp32
   ```

5. Find your board and port:
   ```bash
   arduino-cli board list
   ```
   Example usage for visualization/recording script(omit --record_path if you do not wish to record data):
   ```bash
   python tof_matrix_viz.py --port /dev/ttyACM0 --baud 115200 --record_path /home/mateo/example.csv 
   ```
