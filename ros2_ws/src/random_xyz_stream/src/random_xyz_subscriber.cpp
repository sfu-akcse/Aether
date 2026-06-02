#include <memory>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"

class RandomXyzSubscriber : public rclcpp::Node {
public:
  RandomXyzSubscriber()
  : Node("random_xyz_subscriber") {
    subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
      "hand_xyz",
      10,
      std::bind(&RandomXyzSubscriber::handle_point, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Listening for XYZ points on /hand_xyz");
  }

private:
  void handle_point(const geometry_msgs::msg::Point::SharedPtr message) const {
    RCLCPP_INFO(
      this->get_logger(),
      "Received x=%.2f y=%.2f z=%.2f",
      message->x,
      message->y,
      message->z);
  }

  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr subscription_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RandomXyzSubscriber>());
  rclcpp::shutdown();
  return 0;
}
