#!/usr/bin/python3

import rospy

from relative_pose_label_estimator import RelativePoseLabelEstimator


def main() -> None:

    relative_pose_label_estimator = RelativePoseLabelEstimator()

    try:
        relative_pose_label_estimator.process()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
