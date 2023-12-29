#!/usr/bin/python3

import rospy

from modules.abstract_relative_pose_estimator import AbstractRelativePoseEstimator


def main() -> None:

    abstract_relative_pose_estimator = AbstractRelativePoseEstimator()

    try:
        abstract_relative_pose_estimator.process()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
