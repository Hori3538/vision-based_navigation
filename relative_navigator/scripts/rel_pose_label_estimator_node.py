#!/usr/bin/python3

import rospy

from rel_pose_label_estimator import RelPoseLabelEstimator


def main() -> None:

    rel_pose_label_estimator = RelPoseLabelEstimator()

    try:
        rel_pose_label_estimator.process()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
