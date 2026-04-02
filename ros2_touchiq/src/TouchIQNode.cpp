#include "ros2_touchiq/TouchIQNode.hpp"
#include "std_msgs/msg/string.hpp"

namespace ros2_touchiq {

  TouchIQNode::TouchIQNode() : Node("test_node") {
    std::printf("hi");
    //TouchIQSensor(port, baud) touch_iq_sensor_;
    //pub_ = this->create_publisher<std_msgs::msg::String>("/test_topic", 10);
    //timer_ = this->create_wall_timer(500ms, std::bind(&TouchIQNode::timer_callback, this));

  }
  void TouchIQNode::timer_callback() {
    //auto msg = std_msgs::msg::String();
    //msg.data = "hii";
    //RCLCPP_INFO(this->get_logger(), "publishing hi!");
    //pub_.publish(msg);
  }
}



int main(int argc, char* argv[]){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TouchIQNode>());
  rclcpp::shutdown();
}
