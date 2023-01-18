#include <local_path_planner/local_path_planner.hpp>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_path_planner_node");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    relative_navigator::LocalPathPlanner local_path_planner(nh, private_nh);

    local_path_planner.process();
    return 0;
}
