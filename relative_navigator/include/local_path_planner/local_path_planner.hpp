#ifndef LOCAL_PATH_PLANNER
#define LOCAL_PATH_PLANNER

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/Twist.h>
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
        double goal_dist_th;
        double goal_yaw_th;
        double predict_dt;
        double predict_time;
        double velocity_reso;
        double yawrate_reso;
        double heading_score_gain;
        double velocity_score_gain;
        double dist_score_gain;

        double robot_radius;
        double max_speed;
        double min_speed;
        double max_yawrate;
        double max_accel;
        double max_dyawrate;
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
            void robot_move(State &state, double velocity, double yawrate);
            std::vector<double> calc_dynamic_window();
            std::vector<State> calc_trajectory(double velocity, double yawrate);
            double calc_heading_score_to_target_point(std::vector<State> &trajectory);
            double calc_heading_score_to_target_pose(std::vector<State> &trajectory);
            std::pair<double, double> decide_input();
            void publish_control_input(double velocity, double yawrate);
            void visualize_trajectory(std::vector<State> &trajectory, ros::Publisher &publisher);
            void reaching_judge();
            void publish_reaching_flag();
            double calc_dist_from_pose(geometry_msgs::Pose pose);
            Param param_;

            std::optional<geometry_msgs::PoseStamped> local_goal_;
            bool reaching_target_point_flag_ = false;
            bool reaching_target_pose_flag_ = false;
            nav_msgs::Odometry current_odometry_;
            std::optional<nav_msgs::Odometry> previous_odometry_;
            std::pair<double, double> previous_input_;
            std::vector<std::vector<State>> trajectories_;
            std::vector<State> best_trajectory_;


            ros::Subscriber local_goal_sub_;
            ros::Subscriber odometry_sub_;
            ros::Publisher control_input_pub_;
            ros::Publisher reaching_target_pose_flag_pub_;
            ros::Publisher local_goal_pub_;
            ros::Publisher best_local_path_pub_;
            ros::Publisher candidate_local_path_pub_;
    };
}

#endif
