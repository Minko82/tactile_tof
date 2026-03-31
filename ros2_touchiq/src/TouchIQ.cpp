#include "std_msgs/String.hpp"
#include "rclcpp/Node.hpp"
#include "rclcpp/Publisher.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/Executor.hpp"

class ROS2TouchIQNode : public rclcpp::Node {
	public:
		rclcpp::Publisher<std_msgs::msg::String>::shared_ptr pub_;
		ROS2TouchIQNode() : Node("ROS2TouchIQNode") {
			pub = this->create_publisher<std_msgs::msg::String>("test", 10);
		}

}





int main(int argc, char* argv[]) {
	

}
