#!/usr/bin/python3

import rospy
from sensor_msgs.msg import CompressedImage
from relative_navigator_msgs.msg import RelPoseLabel, NodeInfoArray

from dataclasses import dataclass
import torch
from typing import Optional, Tuple, cast

from my_models import DirectionNet, OrientationNet
from .utils import compressed_image_to_tensor, tensor_to_compressed_image, infer

@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int
    observed_image_width: int
    observed_image_height: int

    hz: float

    direction_net_weight_path: str
    orientation_net_weight_path: str
    observed_image_topic_name: str
    # reference_image_topic_name: str

class RelPoseLabelEstimator:
    def __init__(self) -> None:
        rospy.init_node("rel_pose_label_estimator")

        self._param: Param = Param(
                cast(int, rospy.get_param("/common/image_width")),
                cast(int, rospy.get_param("/common/image_height")),
                cast(int, rospy.get_param("/common/observed_image_width")),
                cast(int, rospy.get_param("/common/observed_image_height")),

                cast(float, rospy.get_param("~hz")),

                cast(str, rospy.get_param("~direction_net_weight_path")),
                cast(str, rospy.get_param("~orientation_net_weight_path")),
                cast(str, rospy.get_param("~observed_image_topic_name")),
                # cast(str, rospy.get_param("~reference_image_topic_name")),
            )

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        # self._device: str = "cpu"

        self._direction_net: DirectionNet = DirectionNet().to(self._device)
        self._direction_net.load_state_dict(torch.load(self._param.direction_net_weight_path, map_location=torch.device(self._device)))
        self._direction_net.eval()

        self._orientation_net: OrientationNet = OrientationNet().to(self._device)
        self._orientation_net.load_state_dict(torch.load(self._param.orientation_net_weight_path, map_location=torch.device(self._device)))
        self._orientation_net.eval()

        self._observed_image: Optional[torch.Tensor] = None
        # self._reference_image: Optional[torch.Tensor] = None
        self._waypoints: Optional[NodeInfoArray] = None

        self._observed_image_sub: rospy.Subscriber = rospy.Subscriber(
                self._param.observed_image_topic_name,
                CompressedImage, self._observed_image_callback, queue_size=1)

        # self._reference_image_sub: rospy.Subscriber = rospy.Subscriber(
        #         self._param.reference_image_topic_name,
        #         CompressedImage, self._reference_image_callback, queue_size=1)

        self._waypoint_img_pub: rospy.Publisher = rospy.Publisher("~waypoint_img/image_raw/compressed", CompressedImage,
                queue_size=1, tcp_nodelay=True)

        self._waypoints_sub: rospy.Subscriber = rospy.Subscriber(
                "/graph_path_planner/waypoints",
                NodeInfoArray, self._waypoints_callback, queue_size=1)

        self._rel_pose_label_pub: rospy.Publisher = rospy.Publisher(
                "~rel_pose_label", RelPoseLabel, queue_size=1)

    def _observed_image_callback(self, msg: CompressedImage) -> None:
        self._observed_image = compressed_image_to_tensor(msg,
                (self._param.image_height, self._param.image_width))

    # def _reference_image_callback(self, msg: CompressedImage) -> None:
    #     self._reference_image = compressed_image_to_tensor(msg,
    #             (self._param.image_height, self._param.image_width))

    def _waypoints_callback(self, msg: NodeInfoArray) -> None:
        self._waypoints = msg

    def _predict_rel_pose_label(self) -> Tuple[RelPoseLabel, torch.Tensor]:

        # direction_probs: torch.Tensor = infer(self._direction_net, self._device,
        #         cast(torch.Tensor, self._observed_image), cast(torch.Tensor, self._reference_image)).squeeze()
        # orientation_probs: torch.Tensor = infer(self._orientation_net, self._device,
        #         cast(torch.Tensor, self._observed_image), cast(torch.Tensor, self._reference_image)).squeeze()
        #
        # direction_max_idx = direction_probs.max(0).indices
        # orientation_max_idx = orientation_probs.max(0).indices
        #
        for waypoint in self._waypoints.node_infos:
            waypoint_img: torch.Tensor = compressed_image_to_tensor(waypoint.image,
                (self._param.image_height, self._param.image_width))
            direction_probs: torch.Tensor = infer(self._direction_net, self._device,
                    cast(torch.Tensor, self._observed_image), cast(torch.Tensor, waypoint_img)).squeeze()
            orientation_probs: torch.Tensor = infer(self._orientation_net, self._device,
                    cast(torch.Tensor, self._observed_image), cast(torch.Tensor, waypoint_img)).squeeze()
            direction_max_idx = direction_probs.max(0).indices
            orientation_max_idx = orientation_probs.max(0).indices

            if direction_max_idx!=3 and orientation_max_idx!=1: break


        rel_pose_label_msg = RelPoseLabel()
        rel_pose_label_msg.header.stamp = rospy.Time.now()

        rel_pose_label_msg.direction_label = direction_max_idx
        rel_pose_label_msg.orientation_label = orientation_max_idx

        rel_pose_label_msg.direction_label_conf = direction_probs.tolist()
        rel_pose_label_msg.orientation_label_conf = orientation_probs.tolist()

        return rel_pose_label_msg, waypoint_img

    def process(self) -> None:
        rate = rospy.Rate(self._param.hz)

        while not rospy.is_shutdown():
            # if self._observed_image is None or self._reference_image is None: continue
            if self._observed_image is None or self._waypoints is None: continue
            abst_rel_pose_msg, waypoint_img_tensor = self._predict_rel_pose_label()
            self._rel_pose_label_pub.publish(abst_rel_pose_msg)

            waypoint_img_msg: CompressedImage = tensor_to_compressed_image(waypoint_img_tensor,
                    (self._param.observed_image_width, self._param.observed_image_height)
                    )
            self._waypoint_img_pub.publish(waypoint_img_msg)

            self._observed_image = None
            # self._reference_image = None
            self._waypoints = None

            rate.sleep()
