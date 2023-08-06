cmake_minimum_required(VERSION 3.0.2)
project(simple_relative_navigator)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rosbag
    sensor_msgs
    nav_msgs
    relative_navigator_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES simple_relative_navigator
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

add_executable(local_goal_generator_node src/local_goal_generator_node.cpp src/local_goal_generator.cpp)
add_dependencies(local_goal_generator_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    local_goal_generator_node
    ${catkin_LIBRARIES}
)
