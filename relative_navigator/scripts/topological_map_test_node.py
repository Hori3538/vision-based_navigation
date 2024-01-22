#!/usr/bin/env python3

import rospy
from modules.topological_map_test import TopologicalMapTest


def main() -> None:
    try:
        topological_map_test = TopologicalMapTest()
        topological_map_test.process()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

