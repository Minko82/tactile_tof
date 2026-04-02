#include <boost/asio.hpp>
#include <iostream>
#include <cstdio>
namespace ros2_touchiq {

	class TouchIQSensor {
		public:
			TouchIQSensor(std::string port, unsigned int baud);
		private:
      void configSP();
	};
	typedef TouchIQSensor tof_sensor;
}
