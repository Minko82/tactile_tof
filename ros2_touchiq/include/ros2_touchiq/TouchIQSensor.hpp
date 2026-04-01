#include <boost/asio.hpp>
#include <iostream>
#include <cstdio>
namespace touch_iq_sensor {

	class TouchIQSensor {
		public:
			TouchIQSensor(std::string port, unsigned int baud);
		private:
		  //bool configured;
      //boost::asio::serial_port sp_;
      //boost::asio::serial_port configSP();
	};
}
