#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"

class TcpXyzBridge : public rclcpp::Node {
public:
  TcpXyzBridge()
  : Node("tcp_xyz_bridge"),
    socket_fd_(-1),
    host_(get_env_or_default("TCP_XYZ_HOST", "host.docker.internal")),
    port_(get_env_or_default("TCP_XYZ_PORT", "5000")) {
    subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
      "hand_xyz",
      10,
      std::bind(&TcpXyzBridge::handle_point, this, std::placeholders::_1));

    RCLCPP_INFO(
      this->get_logger(),
      "TCP bridge ready. Target=%s:%s, topic=/hand_xyz",
      host_.c_str(),
      port_.c_str());
  }

  ~TcpXyzBridge() override {
    close_socket();
  }

private:
  static std::string get_env_or_default(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? std::string(value) : std::string(fallback);
  }

  void close_socket() {
    if (socket_fd_ >= 0) {
      close(socket_fd_);
      socket_fd_ = -1;
    }
  }

  bool ensure_connected() {
    if (socket_fd_ >= 0) {
      return true;
    }

    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo * result = nullptr;
    const int status = getaddrinfo(host_.c_str(), port_.c_str(), &hints, &result);
    if (status != 0) {
      RCLCPP_WARN(this->get_logger(), "getaddrinfo failed: %s", gai_strerror(status));
      return false;
    }

    for (struct addrinfo * rp = result; rp != nullptr; rp = rp->ai_next) {
      const int candidate_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if (candidate_fd == -1) {
        continue;
      }

      if (connect(candidate_fd, rp->ai_addr, rp->ai_addrlen) == 0) {
        socket_fd_ = candidate_fd;
        freeaddrinfo(result);
        RCLCPP_INFO(this->get_logger(), "Connected to TCP receiver at %s:%s", host_.c_str(), port_.c_str());
        return true;
      }

      close(candidate_fd);
    }

    freeaddrinfo(result);
    RCLCPP_WARN(this->get_logger(), "Could not connect to %s:%s", host_.c_str(), port_.c_str());
    return false;
  }

  void handle_point(const geometry_msgs::msg::Point::SharedPtr message) {
    if (!ensure_connected()) {
      return;
    }

    std::ostringstream payload;
    payload << "{\"x\":" << message->x
            << ",\"y\":" << message->y
            << ",\"z\":" << message->z
            << "}\n";
    const std::string serialized = payload.str();

    const ssize_t bytes_sent = send(socket_fd_, serialized.c_str(), serialized.size(), MSG_NOSIGNAL);
    if (bytes_sent < 0) {
      RCLCPP_WARN(
        this->get_logger(),
        "Send failed (%s). Closing socket and waiting for reconnect.",
        std::strerror(errno));
      close_socket();
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Forwarded x=%.2f y=%.2f z=%.2f",
      message->x,
      message->y,
      message->z);
  }

  int socket_fd_;
  std::string host_;
  std::string port_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr subscription_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TcpXyzBridge>());
  rclcpp::shutdown();
  return 0;
}
