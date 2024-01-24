#!/usr/bin/python3

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from relative_navigator_msgs.msg import NodeInfoArray

from dataclasses import dataclass
import math
import torch
from typing import Optional, Tuple, List, cast

from .utils import compressed_image_to_tensor, tensor_to_compressed_image, infer
from transformutils import get_msg_from_array_2d

@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int
    observed_image_width: int
    observed_image_height: int

    hz: float
    local_goal_min_th: float

    rel_pose_net_path: str
    observed_image_topic_name: str

class RelPoseEstimator:
    def __init__(self) -> None:
        rospy.init_node("rel_pose_estimator")

        self._param: Param = Param(
                cast(int, rospy.get_param("/common/image_width")),
                cast(int, rospy.get_param("/common/image_height")),
                cast(int, rospy.get_param("/common/observed_image_width")),
                cast(int, rospy.get_param("/common/observed_image_height")),

                cast(float, rospy.get_param("~hz")),
                cast(float, rospy.get_param("~local_goal_min_th")),

                cast(str, rospy.get_param("~rel_pose_net_path")),
                cast(str, rospy.get_param("~observed_image_topic_name")),
            )

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        # self._device: str = "cpu"

        self._rel_pose_net: torch.ScriptModule = torch.jit.load(self._param.rel_pose_net_path).eval().to(self._device)

        self._observed_image: Optional[torch.Tensor] = None
        self._waypoints: Optional[NodeInfoArray] = None

        self._observed_image_sub: rospy.Subscriber = rospy.Subscriber(
                self._param.observed_image_topic_name,
                CompressedImage, self._observed_image_callback, queue_size=1)

        self._waypoint_img_pub: rospy.Publisher = rospy.Publisher("/rel_pose_label_estimator/waypoint_img/image_raw/compressed", CompressedImage,
                queue_size=1, tcp_nodelay=True)

        self._waypoints_sub: rospy.Subscriber = rospy.Subscriber(
                "/graph_path_planner/waypoints",
                NodeInfoArray, self._waypoints_callback, queue_size=1)

        self._local_goal_pub_: rospy.Publisher = rospy.Publisher("/local_goal_generator/local_goal",
                PoseStamped,  queue_size=1)

    def _observed_image_callback(self, msg: CompressedImage) -> None:
        self._observed_image = compressed_image_to_tensor(msg,
                (self._param.image_height, self._param.image_width))

    def _waypoints_callback(self, msg: NodeInfoArray) -> None:
        self._waypoints = msg

    def _predict_rel_pose(self) -> Tuple[PoseStamped, torch.Tensor]:

        for waypoint in self._waypoints.node_infos:
            waypoint_img: torch.Tensor = compressed_image_to_tensor(waypoint.image,
                (self._param.image_height, self._param.image_width))
            rel_pose: List[float] = infer(self._rel_pose_net, self._device,
                    cast(torch.Tensor, self._observed_image), cast(torch.Tensor, waypoint_img),
                    use_softmax=False).squeeze().tolist()

            if math.hypot(rel_pose[0], rel_pose[1]) > self._param.local_goal_min_th: break

        rel_pose_msg = PoseStamped()
        rel_pose_msg.header.stamp = rospy.Time.now()
        rel_pose_msg.header.frame_id = "base_link"

        rel_pose_msg.pose = get_msg_from_array_2d(rel_pose)


        return rel_pose_msg, waypoint_img

    def process(self) -> None:
        rate = rospy.Rate(self._param.hz)

        while not rospy.is_shutdown():
            if self._observed_image is None or self._waypoints is None: continue
            rel_pose_msg, waypoint_img_tensor = self._predict_rel_pose()

            waypoint_img_msg: CompressedImage = tensor_to_compressed_image(waypoint_img_tensor,
                    (self._param.observed_image_width, self._param.observed_image_height)
                    )
            self._waypoint_img_pub.publish(waypoint_img_msg)
            self._local_goal_pub_.publish(rel_pose_msg)

            self._observed_image = None
            self._waypoints = None

            rate.sleep()
