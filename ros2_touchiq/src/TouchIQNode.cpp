#include "ros2_touchiq/TouchIQNode.hpp"

namespace touch_iq_node {
namespace touch_iq_sensor {

  TouchIQNode::TouchIQNode(const rclcpp::NodeOptions &options) : Node("test_node"){
    TouchIQSensor(port, baud) touch_iq_sensor_;
    pub_ = this->create_publisher<std_msgs::msg::String>("/test_topic", 10);
    timer_ = this->create_wall_tiemer(500ms, std::bind(&TouchIQNode::timer_callback, this));

  }
  TouchIQNode::timer_callback() {
    auto msg = std_msgs::msg::String();
    msg.data = "hii";
    RCLCPP_INFO(this->get_logger(), "publishing hi!");
    pub_.publish(msg);
  }
}



}
int main(int argc, char* argv[]){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TouchIQNode>());
  rclcpp::shutdown();
}
