#ifndef LOCAL_PATH_PLANNER
#define LOCAL_PATH_PLANNER

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Transform.h>
#include <std_msgs/Bool.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>
#include <tf/tf.h>

#include <optional>

namespace relative_navigator
{
    struct Param
    {
        int hz;
    };

    struct State
    {
        double x;
        double y;
        double yaw;
        double velocity;
        double yawrate;
    };

    class LocalPathPlanner
    {
        public:
            LocalPathPlanner(ros::NodeHandle &nh, ros::NodeHandle &private_nh);
            void process();
        private:
            void local_goal_callback(const geometry_msgs::PoseStampedConstPtr &msg);
            void odometry_callback(const nav_msgs::OdometryConstPtr &msg);
            geometry_msgs::Pose calc_previous_base_to_now_base();
            double adjust_yaw(double yaw);
            void update_local_goal(geometry_msgs::Pose previous_base_to_now_base);
            Param param_;

            std::optional<geometry_msgs::PoseStamped> local_goal_;
            bool reaching_target_point_flag_ = false;
            bool reaching_target_pose_flag_ = false;
            nav_msgs::Odometry current_odometry_;
            std::optional<nav_msgs::Odometry> previous_odometry_;

            ros::Subscriber local_goal_sub_;
            ros::Subscriber odometry_sub_;
            ros::Publisher reaching_target_pose_flag_pub_;
    };
}

#endif
