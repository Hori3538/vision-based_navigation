#include <reference_trajectory_handler/reference_trajectory_handler.h>

namespace relative_navigator
{
    ReferenceTrajectoryHandler::ReferenceTrajectoryHandler(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
    {
        private_nh.param<int>("hz", param_.hz, 10);
        private_nh.param<std::string>("bagfile_path", param_.bagfile_path, "~/bag/abstrelposnet/dkan_perimeter/2022-9-30-1500_dkan_perimeter_1.bag");

        reference_trajectory_ = generate_reference_trajectory();
    }

    std::vector<ReferencePoint> ReferenceTrajectoryHandler::generate_reference_trajectory()
    {
        std::vector<ReferencePoint> reference_trajectory;

        return reference_trajectory;
    }


    void ReferenceTrajectoryHandler::process()
    {
        ros::Rate loop_rate(param_.hz);

        while (ros::ok())
        {
            ros::spinOnce();
            loop_rate.sleep();
        }
    }
}
