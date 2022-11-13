#ifndef REFERENCE_TRAJECTORY_HANDLER
#define REFERENCE_TRAJECTORY_HANDLER

#include "ros/publisher.h"
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <tf2/utils.h>
#include <image_transport/image_transport.h>

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

            void set_points_to_marker(visualization_msgs::Marker& marker, std::vector<ReferencePoint> reference_trajectory);
            visualization_msgs::Marker generate_marker_of_reference_trajectory(std::vector<ReferencePoint> reference_trajectory);
            visualization_msgs::Marker generate_marker_of_reference_points(std::vector<ReferencePoint> reference_trajectory);
            void visualize_reference_trajectory(visualization_msgs::Marker marker_of_reference_trajectory);
            void visualize_reference_points(visualization_msgs::Marker marker_of_reference_points);
            sensor_msgs::CompressedImage get_image_from_trajectory(int index, std::vector<ReferencePoint> reference_trajectory);
            sensor_msgs::CompressedImage get_current_reference_image();
            void publish_reference_image();

            Param param_;

            std::vector<ReferencePoint> reference_trajectory_;
            visualization_msgs::Marker marker_of_reference_trajectory_;
            visualization_msgs::Marker marker_of_reference_points_;
            int current_index_ = 0;

            ros::Publisher reference_trajectory_pub_;
            ros::Publisher reference_points_pub_;
            // image_transport::Publisher reference_image_pub_;
            ros::Publisher reference_image_pub_;
    };
}

#endif
