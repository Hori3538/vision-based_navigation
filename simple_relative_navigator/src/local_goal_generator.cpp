#include <local_goal_generator/local_goal_generator.hpp>

namespace simple_relative_navigator
{
    LocalGoalGenerator::LocalGoalGenerator(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
    {
        private_nh.param<int>("hz", param_.hz, 10);
        private_nh.param<float>("dist_to_goal_x", param_.dist_to_goal_x, 1.5);
        private_nh.param<float>("angle_to_goal", param_.angle_to_goal, 0.2);

        abst_rel_pose_sub_ = nh.subscribe<relative_navigator_msgs::SimpleRelPoseLabel>("/simple_relative_pose_label", 10, &LocalGoalGenerator::abst_rel_pose_callback, this);
        local_goal_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/local_goal_generator/local_goal", 1);
    }

    void LocalGoalGenerator::abst_rel_pose_callback(const relative_navigator_msgs::SimpleRelPoseLabelConstPtr &msg)
    {
        abst_rel_pose_ = *msg;
    }

    geometry_msgs::PoseStamped LocalGoalGenerator::generate_local_goal_from_abst_rel_pose(relative_navigator_msgs::SimpleRelPoseLabel abst_rel_pose)
    {
        geometry_msgs::PoseStamped local_goal;
        local_goal.header.stamp = ros::Time::now();
        local_goal.header.frame_id = "base_link";
        
        int label = abst_rel_pose.label;
        float goal_yaw = 0;
        if(label == 1) local_goal.pose.position.x = param_.dist_to_goal_x;
        if(label == 2) goal_yaw = param_.angle_to_goal;
        if(label == 3) goal_yaw = -param_.angle_to_goal;

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
