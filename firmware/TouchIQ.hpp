#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

typedef struct sensor_reading {
  uint32_t elapsed_time_since_last;
  std::vector<int> data;
  sensor_reading();
};

typedef struct touchiq_sensor {
  touchiq_sensor(uint32_t baud);
  SparkFun_VL53L5CX imager;
  VL53L5CX_ResultsData md;
  int read_sensor();
  int set_state(bool sensor_on);
};
