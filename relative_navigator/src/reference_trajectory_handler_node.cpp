#include <reference_trajectory_handler/reference_trajectory_handler.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "reference_trajectory_handler_node");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    relative_navigator::ReferenceTrajectoryHandler reference_trajectory_handler(nh, private_nh);

    reference_trajectory_handler.process();
    return 0;
}
