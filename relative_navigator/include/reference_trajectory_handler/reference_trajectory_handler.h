#ifndef REFERENCE_TRAJECTORY_HANDLER
#define REFERENCE_TRAJECTORY_HANDLER

#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <geometry_msgs/Pose.h>

namespace relative_navigator
{
    struct Param
    {
        int hz;
        std::string bagfile_path;
    };

    struct ReferencePoint
    {
        sensor_msgs::CompressedImage image;
        geometry_msgs::Pose pose;
    };

    // class ReferenceTrajectory
    // {
    //     public:
    //         ReferenceTrajectory(std::vector<ReferencePoint> reference_points);
    //     private:
    //         std::vector<ReferencePoint> reference_points_;
    //         int current_index = 0;
    // };

    class ReferenceTrajectoryHandler
    {
        public:
            ReferenceTrajectoryHandler(ros::NodeHandle &nh, ros::NodeHandle &pnh);
            void process();
        private:
            std::vector<ReferencePoint> generate_reference_trajectory();

            Param param_;

            std::vector<ReferencePoint> reference_trajectory_;
            int current_index = 0;

    };
}

#endif
