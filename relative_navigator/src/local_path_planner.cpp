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
        private_nh.param<double>("velocity_reso", param_.velocity_reso, 0.02);
        private_nh.param<double>("yawrate_reso", param_.yawrate_reso, 0.05);
        private_nh.param<double>("heading_score_gain", param_.heading_score_gain, 1.2);
        private_nh.param<double>("approaching_score_gain", param_.approaching_score_gain, 1.0);
        private_nh.param<double>("robot_radius", param_.robot_radius,  0.5);
        private_nh.param<double>("max_speed", param_.max_speed, 0.3);
        private_nh.param<double>("min_speed", param_.min_speed, 0.0);
        private_nh.param<double>("max_yawrate", param_.max_yawrate, 2.0);
        private_nh.param<double>("max_accel", param_.max_accel, 10.0);
        private_nh.param<double>("max_dyawrate", param_.max_dyawrate, 6.0);
        private_nh.param<float>("collision_th", param_.collision_th, 0.3);

        private_nh.param<std::string>("odom_topic_name", param_.odom_topic_name, "/odom");
        private_nh.param<std::string>("scan_topic_name", param_.scan_topic_name, "/scan");

        local_goal_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("/local_goal_generator/local_goal", 3, &LocalPathPlanner::local_goal_callback, this);
        odometry_sub_ = nh.subscribe<nav_msgs::Odometry>(param_.odom_topic_name, 3, &LocalPathPlanner::odometry_callback, this);
        scan_sub_ = nh.subscribe<sensor_msgs::LaserScan>(param_.scan_topic_name, 3, &LocalPathPlanner::scan_callback, this);
        reaching_goal_flag_sub_ = nh.subscribe<std_msgs::Bool>("/graph_path_planner/reaching_goal_flag", 3, &LocalPathPlanner::reaching_goal_flag_callback, this);

        local_goal_pub_ = nh.advertise<geometry_msgs::PoseStamped>("local_path_planner/local_goal", 3);
        control_input_pub_ = nh.advertise<geometry_msgs::Twist>("/local_path/cmd_vel", 3);
        best_local_path_pub_ = nh.advertise<nav_msgs::Path>("/best_local_path", 3);
        candidate_local_path_pub_ = nh.advertise<nav_msgs::Path>("/candidate_local_path", 3);
    }

    void LocalPathPlanner::local_goal_callback(const geometry_msgs::PoseStampedConstPtr &msg)
    {
        local_goal_ = *msg;
        reaching_target_point_flag_ = false;
        reaching_target_pose_flag_ = false;
        // if(!local_goal_.has_value() || reaching_target_pose_flag_){
        //     local_goal_ = *msg;
        //     reaching_target_point_flag_ = false;
        //     reaching_target_pose_flag_ = false;
        // }
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

    void LocalPathPlanner::scan_callback(const sensor_msgs::LaserScanConstPtr &msg)
    {
        scan_ = *msg;
        obs_list_ = scan_to_obs_list(scan_.value());
    }

    void LocalPathPlanner::reaching_goal_flag_callback(const std_msgs::BoolConstPtr& msg)
    {
        reaching_goal_flag_ = *msg;
    }

    double LocalPathPlanner::adjust_yaw(double yaw)
    {
        if(yaw > M_PI) return yaw - 2*M_PI;
        if(yaw < -M_PI) return yaw + 2*M_PI;

        return yaw;
    }

    geometry_msgs::Pose LocalPathPlanner::calc_previous_base_to_now_base() const
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

    void LocalPathPlanner::robot_move(State &state, double velocity, double yawrate, double dt)
    {
        state.yaw += yawrate * dt;
        state.yaw = adjust_yaw(state.yaw);

        state.x += velocity * std::cos(state.yaw) * dt;
        state.y += velocity * std::sin(state.yaw) * dt;

        state.velocity = velocity;
        state.yawrate = yawrate;
    }

    std::vector<double> LocalPathPlanner::calc_dynamic_window() const
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

    std::vector<State> LocalPathPlanner::calc_trajectory(double velocity, double yawrate) const
    {
        State state = {0.0, 0.0, 0.0, 0.0, 0.0};
        std::vector<State> trajectory;
        for(double t=0.0; t<=param_.predict_time; t+=param_.predict_dt){
            robot_move(state, velocity, yawrate, param_.predict_dt);
            trajectory.push_back(state);
        }

        return trajectory;
    }

    double LocalPathPlanner::calc_heading_score_to_target_point(std::vector<State> &trajectory,
                                                                geometry_msgs::PoseStamped target_pose)
    {
        State last_state = trajectory.back();
        double angle_to_target_point = std::atan2(target_pose.pose.position.y - last_state.y,
                                          target_pose.pose.position.x - last_state.x);
        angle_to_target_point -= last_state.yaw;
        double heading_score = M_PI - std::abs(adjust_yaw(angle_to_target_point));

        return heading_score;
    }

    double LocalPathPlanner::calc_heading_score_to_target_pose(std::vector<State> &trajectory,
                                                               geometry_msgs::PoseStamped target_pose)
    {
        State last_state = trajectory.back();
        double angle_to_target_pose = adjust_yaw(tf2::getYaw(target_pose.pose.orientation) - last_state.yaw);
        double heading_score = M_PI - std::abs(angle_to_target_pose);

        return heading_score;
    }

    double LocalPathPlanner::calc_approaching_score(std::vector<State> &trajectory,
                                                    geometry_msgs::PoseStamped target_pose)
    {
        State last_state = trajectory.back();
        double delta_x = target_pose.pose.position.x - last_state.x;
        double delta_y = target_pose.pose.position.y - last_state.y;
        double dist_paths_end_to_goal = std::hypot(delta_x, delta_y);

        double dist_paths_start_to_goal = calc_dist_from_pose(target_pose.pose);
        double approaching_score = dist_paths_start_to_goal - dist_paths_end_to_goal;

        return approaching_score;
    }

    std::pair<double, double> LocalPathPlanner::decide_input()
    {
        std::pair<double, double> input{0.0, 0.0};
        if(reaching_target_pose_flag_) return input;
        if(reaching_goal_flag_.has_value() && reaching_goal_flag_.value().data) return input;

        std::vector<double> dynamic_window = calc_dynamic_window();
        double best_score = -INFINITY;

        double best_heading_score = 0;
        double best_approaching_score = 0;

        trajectories_.clear();

        for(double velocity=dynamic_window[0]; velocity<=dynamic_window[1]; velocity+=param_.velocity_reso){
            for(double yawrate=dynamic_window[2]; yawrate<=dynamic_window[3]; yawrate+=param_.yawrate_reso){
                std::vector<State> trajectory = calc_trajectory(velocity, yawrate);
                trajectories_.push_back(trajectory);

                if(is_collision(trajectory)) continue;

                double heading_score;
                double approaching_score;

                if(reaching_target_point_flag_)
                {
                    heading_score = calc_heading_score_to_target_pose(trajectory, local_goal_.value());
                    approaching_score = -std::abs(calc_approaching_score(trajectory, local_goal_.value()));
                }
                else
                {
                    heading_score = param_.heading_score_gain * 
                                    calc_heading_score_to_target_point(trajectory, local_goal_.value());
                    approaching_score = param_.approaching_score_gain * 
                                        calc_approaching_score(trajectory, local_goal_.value());
                }
                double sum_score = heading_score + approaching_score;

                if(sum_score > best_score){
                    best_score = sum_score;
                    input = {velocity, yawrate};
                    best_trajectory_ = trajectory;

                    best_heading_score = heading_score;
                    best_approaching_score = approaching_score;
                }
            }
        }

        return input;
    }

    void LocalPathPlanner::publish_control_input(double velocity, double yawrate,
                                                 const ros::Publisher& publisher)
    {
        geometry_msgs::Twist control_input;
        control_input.linear.x = velocity;
        control_input.angular.z = yawrate; 
        publisher.publish(control_input);
    }

    void LocalPathPlanner::visualize_trajectory(const std::vector<State> &trajectory,
                                                const ros::Publisher &publisher)
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

    std::pair<bool, bool> LocalPathPlanner::reaching_judge() const
    {
        bool reaching_target_point_flag = false;
        bool reaching_target_pose_flag = false;

        double dist_to_target = calc_dist_from_pose(local_goal_.value().pose);
        double yaw_to_target = tf2::getYaw(local_goal_.value().pose.orientation);

        if(dist_to_target < param_.goal_dist_th)
        {
            reaching_target_point_flag = true;
            if(abs(yaw_to_target) < param_.goal_yaw_th) reaching_target_pose_flag = true;
        }

        return {reaching_target_point_flag, reaching_target_pose_flag};
    }

    void LocalPathPlanner::publish_reaching_flag(const ros::Publisher &publisher, 
                                                 bool reaching_flag)
    {
        std_msgs::Bool reaching_target_pose_flag_msg;
        reaching_target_pose_flag_msg.data = reaching_flag;
        publisher.publish(reaching_target_pose_flag_msg);
    }

    double LocalPathPlanner::calc_dist_from_pose(geometry_msgs::Pose pose)
    {
        return std::hypot(pose.position.x, pose.position.y);
    }

    geometry_msgs::PoseArray LocalPathPlanner::scan_to_obs_list(const sensor_msgs::LaserScan &scan)
    {
        geometry_msgs::PoseArray obs_list;
        obs_list.header = scan.header;
        for(float angle=scan.angle_min; const auto& range : scan.ranges)
        {
            geometry_msgs::Pose pose;
            pose.position.x = range * cos(angle);
            pose.position.y = range * sin(angle);
            obs_list.poses.push_back(pose);
            angle += scan.angle_increment;
        }

        return obs_list;
    }

    bool LocalPathPlanner::is_collision(const std::vector<State>& traj) const
    {
        if(!obs_list_.has_value()) return false;
        for (const auto &state : traj)
        {
            for (const auto &obs : obs_list_.value().poses)
            {
                float dist = std::hypot((state.x - obs.position.x), (state.y - obs.position.y));
                if (dist < param_.collision_th) return true;
            }
        }
        return false;
    }

    void LocalPathPlanner::process()
    {
        ros::Rate loop_rate(param_.hz);
        while(ros::ok())
        {
            if(local_goal_.has_value())
            {
                std::tie(reaching_target_point_flag_, reaching_target_pose_flag_) = reaching_judge();

                std::pair<double, double> input = decide_input();
                previous_input_ = input;
                publish_control_input(input.first, input.second, control_input_pub_);

                for(auto& trajectory : trajectories_)
                    visualize_trajectory(trajectory, candidate_local_path_pub_);

                visualize_trajectory(best_trajectory_, best_local_path_pub_);
                local_goal_pub_.publish(local_goal_.value());
                if(reaching_target_pose_flag_) local_goal_.reset();

            }

            ros::spinOnce();
            loop_rate.sleep();
        }
    }
}
