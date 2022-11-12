#include "nav_msgs/Odometry.h"
#include "rosbag/bag.h"
#include "rosbag/message_instance.h"
#include "rosbag/query.h"
#include "rosbag/view.h"
#include "sensor_msgs/CompressedImage.h"
#include <cmath>
#include <cstddef>
#include <math.h>
#include <optional>
#include <reference_trajectory_handler/reference_trajectory_handler.hpp>

namespace relative_navigator
{
    ReferenceTrajectoryHandler::ReferenceTrajectoryHandler(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
    {
        private_nh.param<int>("hz", param_.hz, 10);
        private_nh.param<std::string>("bagfile_path", param_.bagfile_path, "~/bag/abstrelposnet/dkan_perimeter/2022-9-30-1500_dkan_perimeter_1.bag");
        private_nh.param<std::string>("image_topic_name", param_.image_topic_name, "/usb_cam/image_raw/compressed");
        private_nh.param<std::string>("odom_topic_name", param_.odom_topic_name, "/whill/odom");
        private_nh.param<float>("trajectory_resolution_trans", param_.trajectory_resolution_trans, 1.0);
        private_nh.param<float>("trajectory_resolution_yaw", param_.trajectory_resolution_yaw, 0.2);

        reference_trajectory_ = generate_reference_trajectory();
    }

    std::vector<ReferencePoint> ReferenceTrajectoryHandler::generate_reference_trajectory()
    {
        std::vector<ReferencePoint> reference_trajectory;
        std::optional<sensor_msgs::CompressedImage> image;
        std::optional<nav_msgs::Odometry> odom;
        std::optional<nav_msgs::Odometry> initial_odom;
        std::optional<nav_msgs::Odometry> prev_odom;
        rosbag::Bag bag;
        bag.open(param_.bagfile_path);

        std::vector<std::string> topics = {param_.image_topic_name, param_.odom_topic_name};
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        for(rosbag::MessageInstance const message: view)
        {
            sensor_msgs::CompressedImageConstPtr image_msg = message.instantiate<sensor_msgs::CompressedImage>();
            if(image_msg != NULL) image = *image_msg;
            
            nav_msgs::OdometryConstPtr odom_msg = message.instantiate<nav_msgs::Odometry>();
            if(odom_msg != NULL)
            {
                odom = *odom_msg;
                if(!initial_odom.has_value()) initial_odom = odom;
                odom = calc_relative_odom(initial_odom.value(), odom.value());
            }

            if(!image.has_value() || !odom.has_value()) continue;

            Pose2D delta_odom = {INFINITY, INFINITY, INFINITY};
            if(prev_odom.has_value())
                delta_odom = generate_pose_2d(calc_relative_pose(prev_odom->pose.pose, odom->pose.pose));
            double delta_trans = sqrt(pow(delta_odom.x, 2) + pow(delta_odom.y, 2));
            double delta_yaw = abs(delta_odom.yaw);

            if(delta_trans > param_.trajectory_resolution_trans || delta_yaw > param_.trajectory_resolution_yaw)
            {
                reference_trajectory.push_back({image.value(), odom->pose.pose});
                prev_odom = odom;
            }

            image.reset();
            odom.reset();
        }

        return reference_trajectory;
    }

    geometry_msgs::Pose ReferenceTrajectoryHandler::calc_relative_pose(geometry_msgs::Pose from_pose, geometry_msgs::Pose to_pose)
    {
    tf2::Transform from_tf;
    tf2::Transform to_tf;
    tf2::Transform relative_tf;
    geometry_msgs::Pose relative_pose;

    tf2::fromMsg(from_pose, from_tf);
    tf2::fromMsg(to_pose, to_tf);
    relative_tf = from_tf.inverse() * to_tf;
    tf2::toMsg(relative_tf, relative_pose);

    return relative_pose;
    }

    nav_msgs::Odometry ReferenceTrajectoryHandler::calc_relative_odom(nav_msgs::Odometry from_odom, nav_msgs::Odometry to_odom)
    {
        nav_msgs::Odometry relative_odom;
        relative_odom.pose.pose = calc_relative_pose(from_odom.pose.pose, to_odom.pose.pose);
        
        return relative_odom;
    }

    Pose2D ReferenceTrajectoryHandler::generate_pose_2d(geometry_msgs::Pose pose)
    {
        Pose2D pose_2d;
        pose_2d.x = pose.position.x;
        pose_2d.y = pose.position.y;
        pose_2d.yaw = tf2::getYaw(pose.orientation);

        return pose_2d;
    }

    void ReferenceTrajectoryHandler::process()
    {
        ros::Rate loop_rate(param_.hz);

        while (ros::ok())
        {
            ros::spinOnce();
            loop_rate.sleep();
        }
    }
}
