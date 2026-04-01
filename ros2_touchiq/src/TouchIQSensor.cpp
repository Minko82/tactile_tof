#include "ros2_touchiq/TouchIQSensor.hpp"
#include <cstdio>
namespace touch_iq_sensor {
		
  TouchIQSensor::TouchIQSensor(std::string port, unsigned int baud) {
    std::printf("test");

    

  }
  TouchIQSensor::configSP(std::string port, unsigned int baud) {
    asio::io_service io();
    asio::serial_port ser(io);
    try {
      ser.open(port);
      ser.set_option(asio::serial_port_base::baud_rate(baud));
    } catch (const exception& e) {
      std::printf(e.what());
    }
  }
	}







}



