#!/usr/bin/env python3

import rospy
from modules.topological_mapper import TopologicalMapper


def main() -> None:
    try:
        topological_mapper = TopologicalMapper()
        topological_mapper.process()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
