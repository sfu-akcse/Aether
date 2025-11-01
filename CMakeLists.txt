cmake_minimum_required(VERSION 3.22)
project(
    Aether 
    VERSION 1.0 
    DESCRIPTION "Robot Arm controller project using ESP 32 and Raspberry Pi" 
    LANGUAGES C CXX
)

include_directories(include)
file(GLOB SOURCES "src/*.c")
add_executable(
    aether 
    ${SOURCES}
)