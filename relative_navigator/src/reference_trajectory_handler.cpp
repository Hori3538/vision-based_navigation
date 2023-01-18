#include <reference_trajectory_handler/reference_trajectory_handler.hpp>

namespace relative_navigator
{
    ReferenceTrajectoryHandler::ReferenceTrajectoryHandler(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
    {
        private_nh.param<int>("hz", param_.hz, 10);
        private_nh.param<std::string>("bagfile_path", param_.bagfile_path, "/home/amsl/bag/abstrelposnet/dkan_perimeter/for_reference/2022-12-2-1410_dkan_perimeter_for_reference_counterclockwise.bag");
        private_nh.param<std::string>("image_topic_name", param_.image_topic_name, "/usb_cam/image_raw/compressed");
        private_nh.param<std::string>("odom_topic_name", param_.odom_topic_name, "/whill/odom");
        private_nh.param<float>("trajectory_resolution_trans", param_.trajectory_resolution_trans, 1.0);
        private_nh.param<float>("trajectory_resolution_yaw", param_.trajectory_resolution_yaw, 0.2);

        reference_trajectory_ = generate_reference_trajectory();
        marker_of_reference_trajectory_ = generate_marker_of_reference_trajectory(reference_trajectory_);
        marker_of_reference_points_ = generate_marker_of_reference_points(reference_trajectory_);

        reference_trajectory_pub_ = nh.advertise<visualization_msgs::Marker>("/reference_trajectory", 1);
        reference_points_pub_ = nh.advertise<visualization_msgs::Marker>("/reference_points", 1);
        current_reference_point_pub_ = nh.advertise<visualization_msgs::Marker>("/current_reference_point", 1);

        reference_image_pub_ = nh.advertise<sensor_msgs::CompressedImage>("/reference_image/image_raw/compressed", 1);
        reaching_goal_flag_sub_ = nh.subscribe("/reaching_target_pose_flag", 1, &ReferenceTrajectoryHandler::reaching_goal_flag_callback, this);
    }
    void ReferenceTrajectoryHandler::reaching_goal_flag_callback(const std_msgs::BoolConstPtr &msg)
    {
        bool reaching_goal_flag = msg->data;
        if(reaching_goal_flag) current_index_++;
        
        if(current_index_ >= reference_trajectory_.size())
        {
            ROS_INFO("reaching_final_goal [reference_trajectory_handler]");
            current_index_--;
        }
    }

    std::vector<ReferencePoint> ReferenceTrajectoryHandler::generate_reference_trajectory()
    {
        ROS_INFO("Generating reference trajectory from bagfile [reference_trajectory_handler]");

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

        ROS_INFO("Done generating reference trajectory from bagfile");
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

    void ReferenceTrajectoryHandler::set_points_to_marker(visualization_msgs::Marker& marker, std::vector<ReferencePoint> reference_trajectory)
    {
        for(const auto& reference_point: reference_trajectory)
        {
            geometry_msgs::Point point;
            point.x = reference_point.pose.position.x;
            point.y = reference_point.pose.position.y;

            marker.colors.push_back(marker.color);
            marker.points.push_back(point);
            marker.pose.orientation.w = 1; // only used for avoiding quatenion warnings
        }
    }

    visualization_msgs::Marker ReferenceTrajectoryHandler::generate_marker_of_reference_trajectory(std::vector<ReferencePoint> reference_trajectory)
    {
        visualization_msgs::Marker marker;
        marker.type = marker.LINE_STRIP;
        marker.action = marker.ADD;
        marker.scale.x = 0.1; 
        marker.scale.y = 0.1; 
        marker.scale.z = 0.1; 
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;

        set_points_to_marker(marker, reference_trajectory);

        return marker;
    }

    visualization_msgs::Marker ReferenceTrajectoryHandler::generate_marker_of_reference_points(std::vector<ReferencePoint> reference_trajectory)
    {
        visualization_msgs::Marker marker;
        marker.type = marker.SPHERE_LIST;
        marker.action = marker.ADD;
        marker.scale.x = 0.2;
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;

        set_points_to_marker(marker, reference_trajectory);

        return marker;
    }

    visualization_msgs::Marker ReferenceTrajectoryHandler::generate_marker_of_current_reference_point()
    {
        geometry_msgs::Pose current_reference_point_pose = reference_trajectory_[current_index_].pose;
        visualization_msgs::Marker marker;
        marker.type = marker.ARROW;
        marker.scale.x = 1.0;
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.pose = current_reference_point_pose;

        return marker;
    }

    void ReferenceTrajectoryHandler::visualize_reference_trajectory(visualization_msgs::Marker marker_of_reference_trajectory)
    {
        marker_of_reference_trajectory.header.frame_id = "map";
        marker_of_reference_trajectory.header.stamp = ros::Time::now();
        marker_of_reference_trajectory.ns = "reference_trajectory";
        marker_of_reference_trajectory.id = 0;

        reference_trajectory_pub_.publish(marker_of_reference_trajectory);
    }

    void ReferenceTrajectoryHandler::visualize_reference_points(visualization_msgs::Marker marker_of_reference_points)
    {
        marker_of_reference_points.header.frame_id = "map";
        marker_of_reference_points.header.stamp = ros::Time::now();
        marker_of_reference_points.ns = "reference_points";
        marker_of_reference_points.id = 0;

        reference_points_pub_.publish(marker_of_reference_points);
    }

    void ReferenceTrajectoryHandler::visualize_current_reference_point()
    {
        visualization_msgs::Marker marker = generate_marker_of_current_reference_point();
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "current_reference_point";
        marker.id = 0;
        current_reference_point_pub_.publish(marker);
    }

    sensor_msgs::CompressedImage ReferenceTrajectoryHandler::get_image_from_trajectory(int index, std::vector<ReferencePoint> reference_trajectory)
    {
        sensor_msgs::CompressedImage target_image = reference_trajectory[index].image;

        return target_image;
    }

    sensor_msgs::CompressedImage ReferenceTrajectoryHandler::get_current_reference_image()
    {
        sensor_msgs::CompressedImage current_reference_image = get_image_from_trajectory(current_index_, reference_trajectory_);

        return current_reference_image;
    }

    void ReferenceTrajectoryHandler::publish_reference_image()
    {
        sensor_msgs::CompressedImage reference_image = get_current_reference_image();
        reference_image.header.stamp = ros::Time::now();
        reference_image_pub_.publish(reference_image);
    }
    

    void ReferenceTrajectoryHandler::process()
    {
        ros::Rate loop_rate(param_.hz);

        while (ros::ok())
        {
            visualize_reference_trajectory(marker_of_reference_trajectory_);
            visualize_reference_points(marker_of_reference_points_);
            visualize_current_reference_point();
            publish_reference_image();

            ros::spinOnce();
            loop_rate.sleep();
        }
    }
}
