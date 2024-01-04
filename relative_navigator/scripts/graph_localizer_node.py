#!/usr/bin/env python3

import rospy
from modules.graph_localizer import GraphLocalizer


def main() -> None:
    try:
        graph_localizer = GraphLocalizer()
        graph_localizer.process()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

