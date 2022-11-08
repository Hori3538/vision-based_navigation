cmake_minimum_required(VERSION 3.0.2)
project(relative_navigator)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    relative_navigator_msgs
    )

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES relative_navigator
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

add_executable(reference_trajectory_handler_node src/reference_trajectory_handler_node.cpp src/reference_trajectory_handler.cpp)
add_dependencies(reference_trajectory_handler_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    reference_trajectory_handler_node
    ${catkin_LIBRARIES}
    )
