#include <local_path_planner/local_path_planner.hpp>

namespace relative_navigator
{
    LocalPathPlanner::LocalPathPlanner(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
    {
        private_nh.param<int>("hz", param_.hz, 10);
        private_nh.param<double>("goal_dist_th", param_.goal_dist_th, 0.4);
        private_nh.param<double>("goal_yaw_th", param_.goal_yaw_th, 0.16);
        private_nh.param<double>("predict_dt", param_.predict_dt, 0.1);
        private_nh.param<double>("predict_time", param_.predict_time, 3.0);
        private_nh.param<double>("velocity_reso", param_.velocity_reso, 0.05);
        private_nh.param<double>("heading_score_gain", param_.heading_score_gain, 0.05);
        private_nh.param<double>("velocity_score_gain", param_.velocity_score_gain, 1.0);
        private_nh.param<double>("dist_score_gain", param_.dist_score_gain, 0.3);
        private_nh.param<double>("yawrate_reso", param_.yawrate_reso, 0.1);
        private_nh.param<double>("robot_radius", param_.robot_radius, 0.5);
        private_nh.param<double>("max_speed", param_.max_speed, 0.3);
        private_nh.param<double>("min_speed", param_.min_speed, 0.0);
        private_nh.param<double>("max_yawrate", param_.max_yawrate, 1.0);
        private_nh.param<double>("max_accel", param_.max_accel, 3.0);
        private_nh.param<double>("max_dyawrate", param_.max_dyawrate, 5.0);

        local_goal_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("/local_goal_generator/local_goal", 1, &LocalPathPlanner::local_goal_callback, this);
        odometry_sub_ = nh.subscribe<nav_msgs::Odometry>("/whill/odom", 1, &LocalPathPlanner::odometry_callback, this);
        reaching_target_pose_flag_pub_ = nh.advertise<std_msgs::Bool>("reaching_target_pose_flag", 1);
        local_goal_pub_ = nh.advertise<geometry_msgs::PoseStamped>("local_path_planner/local_goal", 1);
        control_input_pub_ = nh.advertise<geometry_msgs::Twist>("local_path/cmd_vel", 1);
        best_local_path_pub_ = nh.advertise<nav_msgs::Path>("best_local_path", 1);
        candidate_local_path_pub_ = nh.advertise<nav_msgs::Path>("candidate_local_path", 1);
    }

    void LocalPathPlanner::local_goal_callback(const geometry_msgs::PoseStampedConstPtr &msg)
    {
        if(!local_goal_.has_value() || reaching_target_pose_flag_){
            local_goal_ = *msg;
            reaching_target_point_flag_ = false;
            reaching_target_pose_flag_ = false;
        }
    }

    void LocalPathPlanner::odometry_callback(const nav_msgs::OdometryConstPtr &msg)
    {
        if(previous_odometry_.has_value())
            previous_odometry_ = current_odometry_;
        else
            previous_odometry_ = *msg;
        current_odometry_ = *msg;
        
        geometry_msgs::Pose previous_base_to_now_base = calc_previous_base_to_now_base();
        if(local_goal_.has_value())
        {
            update_local_goal(previous_base_to_now_base);
        }
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
        now_base_to_local_goal_tf = previous_base_to_now_base_tf.inverse() * previous_base_to_local_goal_tf;
        tf2::toMsg(now_base_to_local_goal_tf, local_goal_.value().pose);
        local_goal_.value().header.stamp = ros::Time::now();
    }

    void LocalPathPlanner::robot_move(State &state, double velocity, double yawrate)
    {
        state.yaw += yawrate * param_.predict_dt;
        state.yaw = adjust_yaw(state.yaw);

        state.x += velocity * std::cos(state.yaw) * param_.predict_dt;
        state.y += velocity * std::sin(state.yaw) * param_.predict_dt;

        state.velocity = velocity;
        state.yawrate = yawrate;
    }

    std::vector<double> LocalPathPlanner::calc_dynamic_window()
    {
        std::vector<double> Vs = {param_.min_speed, param_.max_speed, -param_.max_yawrate, param_.max_yawrate};
        std::vector<double> Vd(4);
        std::vector<double> dynamic_window(4);

        Vd = {previous_input_.first - param_.max_accel * param_.predict_dt,
            previous_input_.first + param_.max_accel * param_.predict_dt,
            previous_input_.second - param_.max_dyawrate * param_.predict_dt,
            previous_input_.second + param_.max_dyawrate * param_.predict_dt};

        dynamic_window = {std::max(Vs[0], Vd[0]), std::min(Vs[1], Vd[1]),
                          std::max(Vs[2], Vd[2]), std::min(Vs[3], Vd[3])};

        return dynamic_window;
    }

    std::vector<State> LocalPathPlanner::calc_trajectory(double velocity, double yawrate)
    {
        State state = {0.0, 0.0, 0.0, 0.0, 0.0};
        std::vector<State> trajectory;
        for(double t=0.0; t<=param_.predict_time; t+=param_.predict_dt){
            robot_move(state, velocity, yawrate);
            trajectory.push_back(state);
        }

        return trajectory;
    }

    double LocalPathPlanner::calc_heading_score_to_target_point(std::vector<State> &trajectory)
    {
        State last_state = trajectory.back();
        double angle_to_target_point = std::atan2(local_goal_.value().pose.position.y - last_state.y,
                                          local_goal_.value().pose.position.x - last_state.x);
        angle_to_target_point -= last_state.yaw;
        double heading_score = M_PI - std::abs(adjust_yaw(angle_to_target_point));

        return heading_score;
    }

    double LocalPathPlanner::calc_heading_score_to_target_pose(std::vector<State> &trajectory)
    {
        State last_state = trajectory.back();
        double angle_to_target_pose = adjust_yaw(tf2::getYaw(local_goal_.value().pose.orientation) - last_state.yaw);
        double heading_score = M_PI - std::abs(angle_to_target_pose);

        return heading_score;
    }

    std::pair<double, double> LocalPathPlanner::decide_input()
    {
        std::pair<double, double> input{0.0, 0.0};
        if (reaching_target_pose_flag_) return input;
        
        std::vector<double> dynamic_window = calc_dynamic_window();
        double best_score = 0;

        double best_heading_score = 0;
        double best_velocity_score = 0;

        trajectories_.clear();

        for(double velocity=dynamic_window[0]; velocity<=dynamic_window[1]; velocity+=param_.velocity_reso){
            for(double yawrate=dynamic_window[2]; yawrate<=dynamic_window[3]; yawrate+=param_.yawrate_reso){
                std::vector<State> trajectory = calc_trajectory(velocity, yawrate);
                trajectories_.push_back(trajectory);

                double heading_score;
                double velocity_score;

                if(reaching_target_point_flag_)
                {
                    heading_score = calc_heading_score_to_target_pose(trajectory);
                    velocity_score = -std::abs(velocity);
                }
                else
                {
                    heading_score = param_.heading_score_gain * calc_heading_score_to_target_point(trajectory);
                    velocity_score = param_.velocity_score_gain * velocity;

                }
                // double sum_score = heading_score + std::abs(velocity_score);
                double sum_score = heading_score + velocity_score;

                if(sum_score > best_score){
                    best_score = sum_score;
                    input = {velocity, yawrate};
                    best_trajectory_ = trajectory;

                    best_heading_score = heading_score;
                    best_velocity_score = velocity_score;
                }
            }
        }
        previous_input_ = input;
        // std::cout << "best_velocity_score" << best_velocity_score << std::endl;

        return input;
    }

    void LocalPathPlanner::publish_control_input(double velocity, double yawrate)
    {
        geometry_msgs::Twist control_input;
        control_input.linear.x = velocity;
        control_input.angular.z = yawrate; 
        control_input_pub_.publish(control_input);
    }

    void LocalPathPlanner::visualize_trajectory(std::vector<State> &trajectory, ros::Publisher &publisher)
    {
        nav_msgs::Path local_path;
        local_path.header.frame_id = "base_link";
        local_path.header.stamp = ros::Time::now();

        for(const auto& state : trajectory){
            geometry_msgs::PoseStamped pose;
            pose.pose.position.x = state.x;
            pose.pose.position.y = state.y;
            local_path.poses.push_back(pose);
        }
        publisher.publish(local_path);
    }

    void LocalPathPlanner::reaching_judge()
    {
        double dist_to_target = calc_dist_from_pose(local_goal_.value().pose);
        double yaw_to_target = tf2::getYaw(local_goal_.value().pose.orientation);

        if(dist_to_target < param_.goal_dist_th)
        {
            reaching_target_point_flag_ = true;
            if(abs(yaw_to_target) < param_.goal_yaw_th) reaching_target_pose_flag_ = true;
        }
    }

    void LocalPathPlanner::publish_reaching_flag()
    {
        std_msgs::Bool reaching_target_pose_flag_msg;
        reaching_target_pose_flag_msg.data = reaching_target_pose_flag_;
        reaching_target_pose_flag_pub_.publish(reaching_target_pose_flag_msg);
    }

    double LocalPathPlanner::calc_dist_from_pose(geometry_msgs::Pose pose)
    {
        return std::sqrt(pow(pose.position.x, 2) + pow(pose.position.y, 2));
    }

    void LocalPathPlanner::process()
    {
        ros::Rate loop_rate(param_.hz);
        while(ros::ok())
        {
            ros::spinOnce();

            if(local_goal_.has_value())
            {
                reaching_judge();

                std::pair<double, double> input = decide_input();
                publish_control_input(input.first, input.second);
                publish_reaching_flag();

                for(auto& trajectory : trajectories_)
                {
                    visualize_trajectory(trajectory, candidate_local_path_pub_);
                }
                visualize_trajectory(best_trajectory_, best_local_path_pub_);
                local_goal_pub_.publish(local_goal_.value());
            }

            loop_rate.sleep();
        }
    }
}
