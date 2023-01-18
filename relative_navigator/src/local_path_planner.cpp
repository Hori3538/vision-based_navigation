#include <local_path_planner/local_path_planner.hpp>

namespace relative_navigator
{
    LocalPathPlanner::LocalPathPlanner(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
    {
        private_nh.param<int>("hz", param_.hz, 10);

        local_goal_sub_ = nh.subscribe("/local_goal_generator/local_goal", 1, &LocalPathPlanner::local_goal_callback, this);
        odometry_sub_ = nh.subscribe("/whill/odom", 1, &LocalPathPlanner::odometry_callback, this);
        reaching_target_pose_flag_pub_ = nh.advertise<std_msgs::Bool>("reaching_target_pose_flag", 1);
    }

    void LocalPathPlanner::local_goal_callback(const geometry_msgs::PoseStampedConstPtr &msg)
    {
        if(!local_goal_.has_value() || reaching_target_pose_flag_)
            local_goal_ = *msg;

    }

    void LocalPathPlanner::odometry_callback(const nav_msgs::OdometryConstPtr &msg)
    {
        if(previous_odometry_.has_value())
            previous_odometry_ = current_odometry_;
        else
            previous_odometry_ = *msg;
        current_odometry_ = *msg;
        
        geometry_msgs::Pose previous_base_to_now_base = calc_previous_base_to_now_base();
        update_local_goal(previous_base_to_now_base);

    }

    double LocalPathPlanner::adjust_yaw(double yaw)
    {
        if(yaw > M_PI){yaw -= 2*M_PI;}
        if(yaw < -M_PI){yaw += 2*M_PI;}

        return yaw;
    }

    geometry_msgs::Pose LocalPathPlanner::calc_previous_base_to_now_base()
    {
        double dx = current_odometry_.pose.pose.position.x - 
            previous_odometry_.value().pose.pose.position.x;
        double dy = current_odometry_.pose.pose.position.y - 
            previous_odometry_.value().pose.pose.position.y;
        double current_yaw = tf2::getYaw(current_odometry_.pose.pose.orientation);
        double previous_yaw = tf2::getYaw(previous_odometry_.value().pose.pose.orientation);
        double dyaw = adjust_yaw(current_yaw - previous_yaw);
        double dtrans = sqrt(dx*dx + dy*dy);
        double drot1 = adjust_yaw(atan2(dy, dx) - previous_yaw);
        double drot2 = adjust_yaw(dyaw - drot1);

        geometry_msgs::Pose previous_base_to_now_base;
        previous_base_to_now_base.position.x = dtrans * cos(drot1);
        previous_base_to_now_base.position.y = dtrans * sin(drot1);
        tf::quaternionTFToMsg(tf::createQuaternionFromYaw(adjust_yaw(drot1 + drot2)), previous_base_to_now_base.orientation);

        return previous_base_to_now_base;
    }

    void LocalPathPlanner::update_local_goal(geometry_msgs::Pose previous_base_to_now_base)
    {
        tf2::Transform previous_base_to_now_base_tf, previous_base_to_local_goal_tf;
        tf2::fromMsg(previous_base_to_now_base, previous_base_to_now_base_tf);
        tf2::fromMsg(local_goal_.value().pose, previous_base_to_local_goal_tf);

        tf2::Transform now_base_to_local_goal_tf;
        now_base_to_local_goal_tf = previous_base_to_local_goal_tf.inverse() * previous_base_to_now_base_tf;
        tf2::toMsg(now_base_to_local_goal_tf, local_goal_.value().pose);
        local_goal_.value().header.stamp = ros::Time::now();
    }

    void LocalPathPlanner::process()
    {

    }
}
