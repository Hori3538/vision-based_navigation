cmake_minimum_required(VERSION 3.0.2)
project(relative_navigator)
add_compile_options(-std=c++2a)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rosbag
    sensor_msgs
    nav_msgs
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

add_executable(local_goal_generator_node src/local_goal_generator_node.cpp src/local_goal_generator.cpp)
add_dependencies(local_goal_generator_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    local_goal_generator_node
    ${catkin_LIBRARIES}
)

add_executable(local_path_planner_node src/local_path_planner_node.cpp src/local_path_planner.cpp)
add_dependencies(local_path_planner_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    local_path_planner_node
    ${catkin_LIBRARIES}
    )
