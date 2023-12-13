#!/usr/bin/python3

import rospy
from sensor_msgs.msg import CompressedImage
from relative_navigator_msgs.msg import RelPoseLabel

from dataclasses import dataclass
import torch
import numpy as np
import cv2
from typing import Optional

import torch.nn.functional as F
from model import AbstRelPosNet

@dataclass(frozen=True)
class Param:
    hz: float
    weight_path: str
    image_width: int
    image_height: int
    observed_image_topic_name: str

class RelPoseLabelEstimator:
    def __init__(self) -> None:
        rospy.init_node("relative_pose_label_estimator")

        self._param: Param = Param(
                rospy.get_param("~hz", 10),
                rospy.get_param("~weight_path", "/home/amsl/catkin_ws/src/vision-based_navigation/dnn_models/abstrelposnet/weights/dkan_perimeter_0130_duplicate_test_20000/best_loss.pt"),
                rospy.get_param("~image_width", 224),
                rospy.get_param("~image_height", 224),
                rospy.get_param("~observed_image_topic_name", "/usb_cam/image_raw/compressed"),
            )
        # self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device: str = "cpu"
        self._model: AbstRelPosNet = AbstRelPosNet().to(self._device)
        self._model.load_state_dict(torch.load(self._param.weight_path, map_location=torch.device(self._device)))
        self._model.eval()

        self._observed_image: Optional[torch.Tensor] = None
        self._reference_image: Optional[torch.Tensor] = None

        self._observed_image_sub: rospy.Subscriber = rospy.Subscriber(
                self._param.observed_image_topic_name,
                CompressedImage, self._observed_image_callback, queue_size=1)

        self._reference_image_sub: rospy.Subscriber = rospy.Subscriber(
                "/reference_image/image_raw/compressed",
                CompressedImage, self._reference_image_callback, queue_size=1)

        self._rel_pose_label_pub: rospy.Publisher = rospy.Publisher(
                "/rel_pose_label_estimator/rel_pose_label", RelPoseLabel, queue_size=1)

    def _observed_image_callback(self, msg: CompressedImage) -> None:
        self._observed_image = self._compressed_image_to_tensor(msg)

    def _reference_image_callback(self, msg: CompressedImage) -> None:
        self._reference_image = self._compressed_image_to_tensor(msg)

    def _compressed_image_to_tensor(self, msg: CompressedImage) -> torch.Tensor:

        np_image: np.ndarray = cv2.imdecode(np.frombuffer(
            msg.data, np.uint8), cv2.IMREAD_COLOR)
        np_image = cv2.resize(
            np_image, (self._param.image_height, self._param.image_width))
        image = torch.tensor(
            np_image, dtype=torch.float32).to(self._device)
        image = image.permute(2, 0, 1).unsqueeze(dim=0)
        image = image / 255

        return image

    def _predict_rel_pose_label(self) -> RelPoseLabel:
        models_output: torch.Tensor = self._model(self._observed_image.to(self._device),
                                                  self._reference_image.to(self._device)).squeeze()
        bin_num: int = models_output.size(dim=0) // 2
        direction_probs: torch.Tensor = F.softmax(models_output[:bin_num+1], 0)
        orientation_probs: torch.Tensor = F.softmax(models_output[bin_num+1:], 0)

        direction_max_idx = direction_probs.max(0).indices
        orientation_max_idx = orientation_probs.max(0).indices

        rel_pose_label_msg = RelPoseLabel()
        rel_pose_label_msg.header.stamp = rospy.Time.now()

        rel_pose_label_msg.direction_label = direction_max_idx
        rel_pose_label_msg.orientation_label = orientation_max_idx

        rel_pose_label_msg.direction_label_conf = direction_probs[direction_max_idx]
        rel_pose_label_msg.orientation_label_conf = orientation_probs[orientation_max_idx]

        return rel_pose_label_msg

    def process(self) -> None:
        rate = rospy.Rate(self._param.hz)

        while not rospy.is_shutdown():
            if self._observed_image is None or self._reference_image is None: continue
            abst_rel_pose_msg: RelPoseLabel = self._predict_rel_pose_label()
            self._observed_image = None
            self._rel_pose_label_pub.publish(abst_rel_pose_msg)

            rate.sleep()