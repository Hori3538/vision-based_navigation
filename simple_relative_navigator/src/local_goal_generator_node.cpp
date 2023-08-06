#include <local_goal_generator/local_goal_generator.hpp>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_goal_generator_node");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    simple_relative_navigator::LocalGoalGenerator local_goal_generator(nh, private_nh);

    local_goal_generator.process();
    return 0;
}
