#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>
namespace touchiq_sensor {
typedef struct sensor_reading {
  absolute_time_t elapsed_time_since_last;
  std::vector<int> data;
  sensor_reading();
}

typedef struct touchiq_sensor {
  touchiq_sensor(uint baud);
  int read_sensor();
  int set_state(bool sensor_on);
}
}

