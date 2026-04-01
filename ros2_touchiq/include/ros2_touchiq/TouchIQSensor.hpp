#include <boost/asio.hpp>
#include <iostream>
#include <cstio>
namespace touch_iq_sensor {
	class TouchIQSensor {
		public:
			TouchIQSensor(std::string port, unsigned int baud);
		private:
			asio::serial_port* sp;
      asio::serial_port* configSP();
	}
}
