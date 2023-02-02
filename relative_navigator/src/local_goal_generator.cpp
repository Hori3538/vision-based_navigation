#include <local_goal_generator/local_goal_generator.hpp>

namespace relative_navigator
{
    LocalGoalGenerator::LocalGoalGenerator(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
    {
        private_nh.param<int>("hz", param_.hz, 10);
        private_nh.param<float>("dist_to_goal_x", param_.dist_to_goal_x, 1.5);
        private_nh.param<float>("dist_to_goal_y", param_.dist_to_goal_y, 1.5);
        private_nh.param<float>("angle_to_goal", param_.angle_to_goal, 0.2);

        abst_rel_pose_sub_ = nh.subscribe<relative_navigator_msgs::AbstRelPose>("/abstract_relative_pose", 10, &LocalGoalGenerator::abst_rel_pose_callback, this);
        local_goal_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/local_goal_generator/local_goal", 1);
    }

    void LocalGoalGenerator::abst_rel_pose_callback(const relative_navigator_msgs::AbstRelPoseConstPtr &msg)
    {
        abst_rel_pose_ = *msg;
    }

    geometry_msgs::PoseStamped LocalGoalGenerator::generate_local_goal_from_abst_rel_pose(relative_navigator_msgs::AbstRelPose abst_rel_pose)
    {
        geometry_msgs::PoseStamped local_goal;
        local_goal.header.stamp = ros::Time::now();
        local_goal.header.frame_id = "base_link";

        float goal_x = abst_rel_pose.x * param_.dist_to_goal_x;
        float goal_y = abst_rel_pose.y * param_.dist_to_goal_y;
        float goal_yaw = abst_rel_pose.yaw * param_.angle_to_goal;

        local_goal.pose.position.x = goal_x;
        local_goal.pose.position.y = goal_y;
        tf::quaternionTFToMsg(tf::createQuaternionFromYaw(goal_yaw), local_goal.pose.orientation);
        
        return local_goal;
    }

    void LocalGoalGenerator::process()
    {
        ros::Rate loop_rate(param_.hz);
        
        while(ros::ok())
        {
            if(abst_rel_pose_.has_value()){
                geometry_msgs::PoseStamped local_goal = generate_local_goal_from_abst_rel_pose(abst_rel_pose_.value());
                local_goal_pub_.publish(local_goal);
            }
            ros::spinOnce();
            loop_rate.sleep();
        }
    }
}
