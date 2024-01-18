#ifndef LOCAL_PATH_PLANNER
#define LOCAL_PATH_PLANNER

#include <optional>
#include <string>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/LaserScan.h>

#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>
#include <tf/tf.h>

// To Do 障害物関連の処理がガバガバなのをどうにかする
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
        double approaching_score_gain;

        double robot_radius;
        double max_speed;
        double min_speed;
        double max_yawrate;
        double max_accel;
        double max_dyawrate;
        float collision_th;

        std::string odom_topic_name;
        std::string scan_topic_name;
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
            void scan_callback(const sensor_msgs::LaserScanConstPtr & msg);
            void reaching_goal_flag_callback(const std_msgs::BoolConstPtr & msg);

            geometry_msgs::Pose calc_previous_base_to_now_base() const;
            static double adjust_yaw(double yaw);
            void update_local_goal(geometry_msgs::Pose previous_base_to_now_base);
            static void robot_move(State &state, double velocity, double yawrate, double dt);
            std::vector<double> calc_dynamic_window() const;
            std::vector<State> calc_trajectory(double velocity, double yawrate) const;
            static double calc_heading_score_to_target_point(std::vector<State> &trajectory,
                                                             geometry_msgs::PoseStamped target_pose);
            static double calc_heading_score_to_target_pose(std::vector<State> &trajectory,
                                                     geometry_msgs::PoseStamped target_pose);
            static double calc_approaching_score(std::vector<State> &trajectory,
                                          geometry_msgs::PoseStamped target_pose);
            std::pair<double, double> decide_input();
            static void publish_control_input(double velocity, double yawrate,
                                              const ros::Publisher& publisher);
            static void visualize_trajectory(const std::vector<State> &trajectory,
                                      const ros::Publisher &publisher);
            std::pair<bool, bool> reaching_judge() const;
            static void publish_reaching_flag(const ros::Publisher &publisher, bool reaching_flag);
            static double calc_dist_from_pose(geometry_msgs::Pose pose);

            static geometry_msgs::PoseArray scan_to_obs_list(const sensor_msgs::LaserScan &scan);
            bool is_collision(const std::vector<State>& traj) const;

            Param param_;

            std::optional<geometry_msgs::PoseStamped> local_goal_;
            std::optional<sensor_msgs::LaserScan> scan_;
            std::optional<std_msgs::Bool> reaching_goal_flag_;
            nav_msgs::Odometry current_odometry_;
            std::optional<nav_msgs::Odometry> previous_odometry_;

            bool reaching_target_point_flag_ = false;
            bool reaching_target_pose_flag_ = false;
            std::pair<double, double> previous_input_;
            std::optional<geometry_msgs::PoseArray> obs_list_;

            std::vector<std::vector<State>> trajectories_;
            std::vector<State> best_trajectory_;

            ros::Subscriber local_goal_sub_;
            ros::Subscriber odometry_sub_;
            ros::Subscriber scan_sub_;
            ros::Subscriber reaching_goal_flag_sub_;

            ros::Publisher control_input_pub_;
            // ros::Publisher reaching_target_pose_flag_pub_;
            ros::Publisher local_goal_pub_;
            ros::Publisher best_local_path_pub_;
            ros::Publisher candidate_local_path_pub_;
    };
}

#endif
