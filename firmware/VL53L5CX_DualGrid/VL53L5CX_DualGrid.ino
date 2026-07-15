/*
 * VL53L5CX Touch + Proximity — ESP32-C6
 * Reads one distance per zone (8x8).
 * Zones within TOUCH_MAX_MM are classified as TOUCH.
 * Zones beyond that are classified as PROXIMITY.
 *
 * Set TOUCH_MAX_MM to the resting height of your silicone dome in mm.
 *
 * Library: SparkFun VL53L5CX (platform.h: NB_TARGET_PER_ZONE = 1U)
 * Wiring:  Qwiic cable, SDA=GPIO6, SCL=GPIO7
 */

#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

// ── Tune this to the resting height of your silicone dome (mm) ───────────────
#define TOUCH_MAX_MM    10

#define RANGING_HZ      1
#define I2C_SPEED       400000

SparkFun_VL53L5CX sensor;
VL53L5CX_ResultsData results;

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  unsigned long t = millis();
  while (!Serial && millis() - t < 5000) delay(10);

  Serial.println("\n=== VL53L5CX Touch + Proximity ===");
  Serial.printf("Touch threshold: <= %d mm\n\n", TOUCH_MAX_MM);

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

  sensor.setResolution(8 * 8);
  sensor.setTargetOrder(SF_VL53L5CX_TARGET_ORDER::CLOSEST);
  sensor.setRangingFrequency(RANGING_HZ);
  sensor.startRanging();

  Serial.println("Ready.\n");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  if (!sensor.isDataReady()) return;
  if (!sensor.getRangingData(&results)) return;
  printGrid();
}

// ── Grid ──────────────────────────────────────────────────────────────────────
void printGrid() {
  bool anyTouch = false;

  Serial.println("  Distance (mm)    T=touch  .=proximity  0=no reading");

  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      int zone = row * 8 + col;
      int status = results.target_status[zone];
      bool valid = (results.nb_target_detected[zone] > 0) &&
                   (status == 5 || status == 6 || status == 9);

      if (!valid) {
        Serial.print("   0 ");
        continue;
      }

      int dist = results.distance_mm[zone];
      bool isTouch = (dist <= TOUCH_MAX_MM);
      if (isTouch) anyTouch = true;

      Serial.printf("%3d%c ", dist, isTouch ? 'T' : '.');
    }
    Serial.println();
  }

  Serial.printf("\n%s\n\n", anyTouch ? ">>> TOUCH <<<" : "    proximity");
}
