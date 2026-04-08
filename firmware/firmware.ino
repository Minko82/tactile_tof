#include <cstdio>
#include <TouchIQ.hpp>
#include <memory>
extern touchiq_sensor sensor(115200);
void setup() {
  std::printf("hi1");
  sensor.set_state(false);
}

void loop() {
  std::printf("hi");
  sensor.read_VL53L5CX();
}

