#!/usr/bin/env python3

import rospy
from modules.graph_path_planner import GraphPathPlanner

def main() -> None:
    try:
        graph_path_planner = GraphPathPlanner()
        graph_path_planner.process()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()

