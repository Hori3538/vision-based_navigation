#ifndef REFERENCE_TRAJECTORY_HANDLER
#define REFERENCE_TRAJECTORY_HANDLER

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <tf2/utils.h>

#include <optional>

namespace relative_navigator
{
    struct Pose2D {
        double x;
        double y;
        double yaw;
    };

    struct Param
    {
        int hz;
        std::string bagfile_path;
        std::string image_topic_name;
        std::string odom_topic_name;
        float trajectory_resolution_trans;
        float trajectory_resolution_yaw;
    };

    struct ReferencePoint
    {
        sensor_msgs::CompressedImage image;
        geometry_msgs::Pose pose;
    };

    // class ReferenceTrajectory
    // {
    //     public:
    //         ReferenceTrajectory(std::vector<ReferencePoint> reference_points);
    //     private:
    //         std::vector<ReferencePoint> reference_points_;
    //         int current_index = 0;
    // };

    class ReferenceTrajectoryHandler
    {
        public:
            ReferenceTrajectoryHandler(ros::NodeHandle &nh, ros::NodeHandle &pnh);
            void process();
        private:
            std::vector<ReferencePoint> generate_reference_trajectory();
            geometry_msgs::Pose calc_relative_pose(geometry_msgs::Pose from_pose, geometry_msgs::Pose to_pose);
            nav_msgs::Odometry calc_relative_odom(nav_msgs::Odometry from_odom, nav_msgs::Odometry to_odom);
            Pose2D generate_pose_2d(geometry_msgs::Pose pose);

            Param param_;

            std::vector<ReferencePoint> reference_trajectory_;
            int current_index = 0;
    };
}

#endif
