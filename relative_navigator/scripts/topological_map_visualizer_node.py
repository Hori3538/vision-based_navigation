#!/usr/bin/env python3

import rospy
from modules.topological_map_visualizer import TopologicalMapVisualizer


def main() -> None:
    try:
        topological_map_visualizer = TopologicalMapVisualizer()
        topological_map_visualizer.process()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

