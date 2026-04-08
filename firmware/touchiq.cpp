#include <TouchIQ.hpp>

sensor_reading::sensor_reading() {};

touchiq_sensor::touchiq_sensor(uint32_t baud) {
  SparkFun_VL53L5CX imager;
  VL53L5CX_ResultsData md;
  Serial.begin(115200);
  Wire.begin(6, 7);
  Wire.setClock(400000);
  if (!imager.begin(0x29, Wire)) {
    while (1);
  }
  imager.setResolution(8 * 8);
  imager.setRangingFrequency(10);
}

int touchiq_sensor::set_state(bool sensor_on) {
  bool res;
  if (!sensor_on) {
    res = this->imager.stopRanging();
  }
  else {
    res = this->imager.startRanging();
  }
  if (!res) {
    return -1;
  }
  return 0;
}

int touchiq_sensor::read_VL53L5CX() {
  if (!this->imager.isDataReady()) {
    return -1;
  }
  else if (!this->imager.getRangingData(&this->md)) {
    return -1;
  }
  else {
    std::vector<uint32_t> data = md.distance_mm;
    if (!data) {
      return -1;
    }
    if (data.size() < 64) {
      return -1;
    }
    sensor_reading msg = sensor_reading()
    return 0;
  }
}
