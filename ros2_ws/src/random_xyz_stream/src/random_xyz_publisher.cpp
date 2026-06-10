#include <chrono>
#include <memory>
#include <random>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class RandomXyzPublisher : public rclcpp::Node {
public:
  RandomXyzPublisher()
  : Node("random_xyz_publisher"),
    generator_(std::random_device{}()),
    distribution_(-100.0, 100.0) {
    publisher_ = this->create_publisher<geometry_msgs::msg::Point>("hand_xyz", 10);
    timer_ = this->create_wall_timer(500ms, std::bind(&RandomXyzPublisher::publish_point, this));
    RCLCPP_INFO(this->get_logger(), "Publishing random XYZ points on /hand_xyz");
  }

private:
  void publish_point() {
    geometry_msgs::msg::Point point;
    point.x = distribution_(generator_);
    point.y = distribution_(generator_);
    point.z = distribution_(generator_);

    publisher_->publish(point);
    RCLCPP_INFO(
      this->get_logger(),
      "Published x=%.2f y=%.2f z=%.2f",
      point.x,
      point.y,
      point.z);
  }

  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::mt19937 generator_;
  std::uniform_real_distribution<double> distribution_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RandomXyzPublisher>());
  rclcpp::shutdown();
  return 0;
}
