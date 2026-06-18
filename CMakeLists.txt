cmake_minimum_required(VERSION 3.14)
project(aether_tests)

find_package(GTest REQUIRED)

add_executable(run_tests tests/sample_test.cpp)
target_link_libraries(run_tests GTest::GTest GTest::Main)
