/*
  vl53l5cx_stream.ino — stream VL53L5CX 8x8 frames over USB serial, including
  the per-zone histogram summary statistics (not just distance).

  Output per frame, CSV, one line each, in this ORDER:
      A,8,<a0>,...,<a63>    ambient_per_spad (kcps/SPAD; always valid — measures
                            background IR even in zones with no target)
      S,8,<s0>,...,<s63>    signal_per_spad  (kcps/SPAD; 0 = no valid target)
      Q,8,<q0>,...,<q63>    range_sigma_mm   (per-zone noise estimate; 0 = invalid)
      D,8,<d0>,...,<d63>    distance_mm      (-1 = no valid target)
  '#'-prefixed lines are human-readable status.

  S and Q are sent BEFORE D on purpose: readers key on the D line (the format
  the old firmware sent alone), so a reader that understands S/Q has already
  buffered them when D arrives — zero added latency — while old readers that
  only parse D lines keep working unchanged.

  Board: ESP32-C6 (native USB CDC — the baud setting is cosmetic; USB runs at
  bus speed, so the 3x line traffic costs nothing).
  Library: SparkFun VL53L5CX (ST ULD wrapper).
*/
#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

SparkFun_VL53L5CX sensor;
VL53L5CX_ResultsData results;

const uint8_t  RESOLUTION = 64;   // 8x8
const uint8_t  FREQ_HZ    = 15;   // max for 8x8 mode

// target_status 5 = 100% valid, 9 = 50% confidence (usable). Everything else
// is reported as invalid, matching the old firmware's -1 convention.
static inline bool zoneValid(uint8_t status) { return status == 5 || status == 9; }

static char lineBuf[1024];

// Emit one CSV line: <tag>,8,<v0>,...,<v63> using the per-zone value picker.
template <typename F>
void emitLine(const char *tag, F value) {
  int n = snprintf(lineBuf, sizeof(lineBuf), "%s,8", tag);
  for (int i = 0; i < RESOLUTION; i++) {
    n += snprintf(lineBuf + n, sizeof(lineBuf) - n, ",%ld", (long)value(i));
    if (n >= (int)sizeof(lineBuf) - 16) break;    // never overflow
  }
  Serial.println(lineBuf);
}

void setup() {
  Serial.begin(921600);
  delay(100);
  Serial.println("# vl53l5cx_stream starting");

  Wire.begin(6, 7);                // this board: SDA=6, SCL=7 (same as basic_sensor)
  Wire.setClock(400000);

  bool ready = false;
  for (int i = 0; i < 10 && !ready; i++) {   // begin() uploads the sensor's own
    ready = sensor.begin();                  // firmware blob — allow retries
    if (!ready) { Serial.println("# waiting for sensor ..."); delay(500); }
  }
  if (!ready) {
    Serial.println("# ERROR: VL53L5CX not found — check wiring, then reset");
    while (true) delay(1000);
  }
  sensor.setResolution(RESOLUTION);
  sensor.setRangingFrequency(FREQ_HZ);
  sensor.startRanging();
  Serial.println("# ready: streaming S,Q,D lines per frame");
}

void loop() {
  if (!sensor.isDataReady()) { delay(1); return; }
  if (!sensor.getRangingData(&results)) return;

  emitLine("A", [](int i) -> long {          // ambient: not gated on validity
    return (long)results.ambient_per_spad[i];
  });
  emitLine("S", [](int i) -> long {
    return zoneValid(results.target_status[i]) ? (long)results.signal_per_spad[i] : 0;
  });
  emitLine("Q", [](int i) -> long {
    return zoneValid(results.target_status[i]) ? (long)results.range_sigma_mm[i] : 0;
  });
  emitLine("D", [](int i) -> long {
    return zoneValid(results.target_status[i]) ? (long)results.distance_mm[i] : -1;
  });
}
