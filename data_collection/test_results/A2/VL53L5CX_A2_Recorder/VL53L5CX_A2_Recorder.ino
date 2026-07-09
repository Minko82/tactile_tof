/*
 * VL53L5CX A2 Recorder Firmware — ESP32-C6
 * ----------------------------------------
 * Identical FRAME: output format to VL53L5CX_Visualizer, but ranges at the 8x8
 * (64-zone) ceiling of 15 Hz for the A2 baseline / Kalman characterization test.
 *
 * The A2 spec asks for 60 Hz, but 60 Hz is only reachable at 4x4 (16 zones).
 * Keeping 64 zones caps the rate at 15 Hz — this sketch runs that ceiling, and
 * analyze_a2_kalman.py filters every sample using dt taken from timestamps, so
 * the noise-reduction factor is valid at whatever rate is actually achieved.
 *
 * Each frame is one line:
 *   FRAME:18,342,0,200,...   (64 comma-separated raw distance values in mm)
 *     "18" = 18 mm valid reading · "0" = no valid return · trailing "?" = low confidence
 *
 * Library: SparkFun VL53L5CX (platform.h: NB_TARGET_PER_ZONE = 1U)
 * Wiring:  Qwiic cable, SDA=GPIO6, SCL=GPIO7
 * Board:   ESP32C6 Dev Module · USB CDC On Boot: Enabled · 115200 baud
 */

#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

#define RANGING_HZ     15    // 8x8 ceiling (60 Hz would require dropping to 4x4)
#define I2C_SPEED      400000

SparkFun_VL53L5CX sensor;
VL53L5CX_ResultsData results;

void setup() {
  Serial.begin(115200);
  unsigned long t = millis();
  while (!Serial && millis() - t < 5000) delay(10);

  Serial.println("VL53L5CX A2 Recorder starting...");

  Wire.begin(6, 7);
  Wire.setClock(I2C_SPEED);

  Serial.print("Waiting for sensor");
  delay(500);
  bool ready = false;
  for (int i = 0; i < 10; i++) {
    Serial.print(".");
    if (sensor.begin()) { ready = true; break; }
    delay(500);
  }
  Serial.println();
  if (!ready) { Serial.println("Sensor not found. Halting."); while (true); }

  sensor.setResolution(8 * 8);          // keep 64 zones
  sensor.setTargetOrder(SF_VL53L5CX_TARGET_ORDER::CLOSEST);
  sensor.setRangingFrequency(RANGING_HZ);
  sensor.startRanging();

  Serial.printf("READY ranging_hz=%d\n", RANGING_HZ);
}

void loop() {
  if (!sensor.isDataReady()) return;
  if (!sensor.getRangingData(&results)) return;
  outputFrame();
}

void outputFrame() {
  Serial.print("FRAME:");
  for (int i = 0; i < 64; i++) {
    int status = results.target_status[i];
    bool good = (results.nb_target_detected[i] > 0) &&
                (status == 5 || status == 6 || status == 9);
    bool poor = !good && (results.nb_target_detected[i] > 0) &&
                (status == 4 || status == 10);

    if (good) {
      Serial.print(results.distance_mm[i]);
    } else if (poor) {
      Serial.print(results.distance_mm[i]);
      Serial.print("?");   // low-confidence reading
    } else {
      Serial.print("0");
    }

    if (i < 63) Serial.print(",");
  }
  Serial.println();
}
