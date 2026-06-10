cmake_minimum_required(VERSION 3.8)
project(random_xyz_stream)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(random_xyz_publisher src/random_xyz_publisher.cpp)
ament_target_dependencies(random_xyz_publisher rclcpp geometry_msgs)

add_executable(random_xyz_subscriber src/random_xyz_subscriber.cpp)
ament_target_dependencies(random_xyz_subscriber rclcpp geometry_msgs)

add_executable(tcp_xyz_bridge src/tcp_xyz_bridge.cpp)
ament_target_dependencies(tcp_xyz_bridge rclcpp geometry_msgs)

install(TARGETS
  random_xyz_publisher
  random_xyz_subscriber
  tcp_xyz_bridge
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
