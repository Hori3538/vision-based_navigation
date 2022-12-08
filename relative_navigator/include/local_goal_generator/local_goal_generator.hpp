#ifndef LOCAL_GOAL_GENERATOR
#define LOCAL_GOAL_GENERATOR

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <relative_navigator_msgs/AbstRelPose.h>
#include <tf/tf.h>

#include <optional>

namespace relative_navigator
{

    struct Param
    {
        int hz;

        // モデルの学習時に用いたラベリングの閾値同じ値を使う
        float dist_to_goal_x;
        float dist_to_goal_y;
        float angle_to_goal;
    };

    class LocalGoalGenerator
    {
        public:
            LocalGoalGenerator(ros::NodeHandle &nh, ros::NodeHandle &private_nh);
            void process();
        private:
            void abst_rel_pose_callback(const relative_navigator_msgs::AbstRelPoseConstPtr &msg);
            geometry_msgs::PoseStamped generate_local_goal_from_abst_rel_pose(relative_navigator_msgs::AbstRelPose abst_rel_pose);

            std::optional<relative_navigator_msgs::AbstRelPose> abst_rel_pose_;
            Param param_;

            ros::Subscriber abst_rel_pose_sub_;
            ros::Publisher local_goal_pub_;
    };
}

#endif 
