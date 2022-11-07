cmake_minimum_required(VERSION 3.0.2)
project(relative_navigator_msgs)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    message_generation
    )

add_message_files(
  FILES
  ImageWithPose.msg
  ImageWithPoseArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs  
  geometry_msgs
  sensor_msgs
)


catkin_package(
    CATKIN_DEPENDS std_msgs sensor_msgs geometry_msgs message_runtime
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)
