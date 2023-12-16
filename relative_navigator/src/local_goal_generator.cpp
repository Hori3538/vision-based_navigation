#include <local_goal_generator/local_goal_generator.hpp>

namespace relative_navigator
{
    LocalGoalGenerator::LocalGoalGenerator(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
    {
        private_nh.param<int>("hz", param_.hz, 10);
        private_nh.param<float>("dist_to_local_goal", param_.dist_to_local_goal, 1.5);

        private_nh.param<int>("bin_num", param_.bin_num, 3);
        private_nh.param<float>("bin_step_degree", param_.bin_step_degree, 25);

        rel_pose_label_sub_ = nh.subscribe<relative_navigator_msgs::RelPoseLabel>("/rel_pose_label_estimator/rel_pose_label", 10, &LocalGoalGenerator::rel_pose_label_callback, this);
        local_goal_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/local_goal_generator/local_goal", 1);
    }

    void LocalGoalGenerator::rel_pose_label_callback(const relative_navigator_msgs::RelPoseLabelConstPtr &msg)
    {
        rel_pose_label_ = *msg;
    }

    std::vector<float> LocalGoalGenerator::calc_angle_for_each_label(int bin_num, float bin_step_degree)
    {
        std::vector<float> angle_for_each_angle;
        for(int i=0; i<bin_num; i++)
        {
            float angle_degree = bin_step_degree * int(bin_num / 2) - bin_step_degree * i;
            angle_for_each_angle.push_back(angle_degree * (M_PI/180)); // degree to radian
        }

        return angle_for_each_angle;
    }

    float LocalGoalGenerator::calc_weighted_mean_angle(std::vector<float> angle_for_each_labe,
                                                  std::vector<float> orientation_label_conf)
    {

    }

    geometry_msgs::PoseStamped LocalGoalGenerator::generate_local_goal_from_rel_pose_label(
            relative_navigator_msgs::RelPoseLabel rel_pose_label, int bin_num,
            float bin_step_degree, float dist_to_local_goal) 
    {
        auto label_to_angle_rad = [&](int label) -> std::optional<float> {
            if(label == bin_num) return std::nullopt;
            float angle_degree = bin_step_degree * int(bin_num / 2) - bin_step_degree * label;

            return angle_degree * (M_PI/180); // degree to radian
        };

        geometry_msgs::PoseStamped local_goal;
        local_goal.header.stamp = ros::Time::now();
        local_goal.header.frame_id = "base_link";

        const std::optional<float> direction_angle = label_to_angle_rad(rel_pose_label.direction_label);
        const std::optional<float> orientation_angle = label_to_angle_rad(rel_pose_label.orientation_label);
        if(direction_angle.has_value())
        {
            float local_goal_x = std::cos(direction_angle.value()) * dist_to_local_goal;
            float local_goal_y = std::sin(direction_angle.value()) * dist_to_local_goal;

            local_goal.pose.position.x = local_goal_x;
            local_goal.pose.position.y = local_goal_y;
        }

        else 
            tf::quaternionTFToMsg(tf::createQuaternionFromYaw(orientation_angle.value()), local_goal.pose.orientation);

        return local_goal;
    }

    void LocalGoalGenerator::process()
    {
        ros::Rate loop_rate(param_.hz);
        
        while(ros::ok())
        {
            if(rel_pose_label_.has_value()){
                geometry_msgs::PoseStamped local_goal = generate_local_goal_from_rel_pose_label(
                        rel_pose_label_.value(), param_.bin_num,
                        param_.bin_step_degree, param_.dist_to_local_goal);
                local_goal_pub_.publish(local_goal);
                rel_pose_label_.reset();
            }
            ros::spinOnce();
            loop_rate.sleep();
        }
    }
}
