#include "rclcpp/rclcpp.hpp"
#include "rclcpp/node.hpp"
#include <cstdio>
#include "ros2_touchiq/TouchIQSensor.hpp"
#include "std_msgs/msg/string.hpp"

namespace ros2_touchiq {
  class TouchIQNode : public rclcpp::Node {
    public:
      explicit TouchIQNode(); 
    private:
      rclcpp::TimerBase::SharedPtr timer_;
      rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
      tof_sensor touch_iq_sensor_;

      void timer_callback();
  };


}
