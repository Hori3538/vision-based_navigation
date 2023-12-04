#ifndef LOCAL_GOAL_GENERATOR
#define LOCAL_GOAL_GENERATOR

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <relative_navigator_msgs/RelPoseLabel.h>
#include <tf/tf.h>

#include <optional>

namespace relative_navigator
{

    struct Param
    {
        int hz;

        // モデルの学習時に用いたラベリングの閾値同じ値を使う
        float dist_to_local_goal;

        int bin_num;
        float bin_step_degree;
    };

    class LocalGoalGenerator
    {
        public:
            LocalGoalGenerator(ros::NodeHandle &nh, ros::NodeHandle &private_nh);
            void process();
        private:
            void rel_pose_label_callback(const relative_navigator_msgs::RelPoseLabelConstPtr &msg);
            static geometry_msgs::PoseStamped generate_local_goal_from_rel_pose_label(
                    relative_navigator_msgs::RelPoseLabel rel_pose_label,
                    int bin_num, float bin_step_degree, float dist_to_local_goal);

            std::optional<relative_navigator_msgs::RelPoseLabel> rel_pose_label_;
            Param param_;

            ros::Subscriber rel_pose_label_sub_;
            ros::Publisher local_goal_pub_;
    };
}

#endif 
