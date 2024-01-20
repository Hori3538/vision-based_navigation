#!/usr/bin/python3

import rospy

from modules.rel_pose_estimator import RelPoseEstimator


def main() -> None:

    rel_pose_label_estimator = RelPoseEstimator()

    try:
        rel_pose_label_estimator.process()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
