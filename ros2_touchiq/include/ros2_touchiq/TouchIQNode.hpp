#include "rclcpp/rclcpp.hpp"
#include "rclcpp/Node.hpp"
#include <cstdio>
#include "ros2_touchiq/TouchIQSensor.hpp"
namespace touch_iq_node {
  class TouchIQNode : public rclcpp::Node {
    public:
      TouchIQNode(const rclcpp::NodeOptions &options);
    private:
      rclcpp::TimerBase::SharedPtr timer_;
      rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
      ros2_touchiq::TouchIQSensor touch_iq_sensor_;

      void timer_callback();
  }


}
