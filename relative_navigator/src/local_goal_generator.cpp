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
        local_goal_pub_ = private_nh.advertise<geometry_msgs::PoseStamped>("local_goal", 1);
    }

    void LocalGoalGenerator::rel_pose_label_callback(const relative_navigator_msgs::RelPoseLabelConstPtr &msg)
    {
        rel_pose_label_ = *msg;
    }

    std::vector<float> LocalGoalGenerator::calc_angle_for_each_label(int bin_num, float bin_step_degree)
    {
        std::vector<float> angle_for_each_label;
        for(int i=0; i<bin_num; i++)
        {
            float angle_degree = bin_step_degree * int(bin_num / 2) - bin_step_degree * i;
            angle_for_each_label.push_back(angle_degree * (M_PI/180)); // degree to radian
        }

        return angle_for_each_label;
    }

    float LocalGoalGenerator::calc_weighted_mean_angle(std::vector<float> angle_for_each_label,
                                                  std::vector<float> orientation_label_conf)
    {
        float weighted_mean_angle = 0;
        for(int i=0; const auto& angle: angle_for_each_label)
        {
            weighted_mean_angle += angle * orientation_label_conf[i];
            i++;
        }

        return weighted_mean_angle;
    }

    geometry_msgs::Pose LocalGoalGenerator::calc_weighted_mean_pose(
            std::vector<float> angle_for_each_label, std::vector<float> direction_label_conf,
            float dist_to_local_goal)
    {
        float sum_of_positive_confs = std::accumulate(direction_label_conf.begin(),
                                                      direction_label_conf.end()-1, 0.0);
        // negative の conf が大きい時 local goal が近くなりすぎるため，
        // positive conf の合計が1になるようにスケーリングする
        std::for_each(direction_label_conf.begin(), direction_label_conf.end()-1,
                [sum_of_positive_confs](float& conf){conf /= sum_of_positive_confs;});

        float weighted_mean_x = 0;
        float weighted_mean_y = 0;
        for(int i=0; const auto& angle: angle_for_each_label)
        {
            weighted_mean_x += std::cos(angle) * dist_to_local_goal * direction_label_conf[i];
            weighted_mean_y += std::sin(angle) * dist_to_local_goal * direction_label_conf[i];
            i++;
        }

        geometry_msgs::Pose weighted_mean_pose;
        weighted_mean_pose.position.x = weighted_mean_x;
        weighted_mean_pose.position.y = weighted_mean_y;

        return weighted_mean_pose;
    }

    geometry_msgs::PoseStamped LocalGoalGenerator::generate_local_goal_from_rel_pose_label(
            relative_navigator_msgs::RelPoseLabel rel_pose_label, int bin_num,
            float bin_step_degree, float dist_to_local_goal) 
    {
        std::vector<float> angle_for_each_label = calc_angle_for_each_label(bin_num, bin_step_degree);
        float weighted_mean_angle = calc_weighted_mean_angle(
                angle_for_each_label,rel_pose_label.orientation_label_conf);

        geometry_msgs::Pose weighted_mean_pose = calc_weighted_mean_pose(
                angle_for_each_label, rel_pose_label.direction_label_conf, dist_to_local_goal);

        tf::quaternionTFToMsg(tf::createQuaternionFromYaw(
                    weighted_mean_angle), weighted_mean_pose.orientation);

        geometry_msgs::PoseStamped local_goal;
        local_goal.header.stamp = ros::Time::now();
        local_goal.header.frame_id = "base_link";

        local_goal.pose = weighted_mean_pose;

        return local_goal;
    }

    void LocalGoalGenerator::process()
    {
        ros::Rate loop_rate(param_.hz);
        
        while(ros::ok())
        {
            if(rel_pose_label_.has_value())
            {
                geometry_msgs::PoseStamped local_goal = generate_local_goal_from_rel_pose_label(
                        rel_pose_label_.value(), param_.bin_num,
                        param_.bin_step_degree, param_.dist_to_local_goal);

                if(rel_pose_label_.value().direction_label == 4)
                    ROS_WARN("Direction label is negative");
                local_goal_pub_.publish(local_goal);
                rel_pose_label_.reset();
            }
            ros::spinOnce();
            loop_rate.sleep();
        }
    }
}
