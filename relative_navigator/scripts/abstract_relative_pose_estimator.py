#!/usr/bin/python3

from dataclasses import dataclass
import rospy
from sensor_msgs.msg import CompressedImage
import torch
import numpy as np
import cv2

import sys
# sys.path.append("../..")
from dnn_models.abstrelposnet.model import AbstRelPosNet
from typing import Optional, List

from relative_navigator_msgs.msg import AbstRelPose

@dataclass(frozen=True)
class Param:
    hz: float
    weight_path: str
    image_width: int
    image_height: int

    # 連続値であるモデルの出力をラベルに変換するときの閾値
    label_th_x: float 
    label_th_y: float
    label_th_yaw: float

class AbstractRelativePoseEstimator:
    def __init__(self) -> None:
        rospy.init_node("abstract_relative_pose_estimator")

        self._param: Param = Param(
                rospy.get_param("hz", 10),
                rospy.get_param("weight_path", "/home/amsl/catkin_ws/src/vision-based_navigation/dnn_models/abstrelposnet/weights/dkan_perimeter_0930_1010/best_loss.pt"),
                rospy.get_param("image_width", 224),
                rospy.get_param("image_height", 224),
                rospy.get_param("label_th_x", 0.33),
                rospy.get_param("label_th_y", 0.33),
                rospy.get_param("label_th_yaw", 0.33),
            )
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        # self._device: str = "cpu"
        self._model: AbstRelPosNet = AbstRelPosNet().to(self._device)
        self._model.load_state_dict(torch.load(self._param.weight_path, map_location=torch.device(self._device)))
        self._model.eval()
        self._observed_image: Optional[torch.Tensor] = None
        self._reference_image: Optional[torch.Tensor] = None

        self._observed_image_sub: rospy.Subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",
            CompressedImage, self._observed_image_callback, queue_size=1)
        self._reference_image_sub: rospy.Subscriber = rospy.Subscriber("/reference_image/image_raw/compressed",
            CompressedImage, self._reference_image_callback, queue_size=1)
        self._abstract_relative_pose_pub: rospy.Publisher = rospy.Publisher("/abstract_relative_pose", AbstRelPose, queue_size=1)

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

    def _predict_abstract_relative_pose(self) -> AbstRelPose:
        abst_rel_pose_tensor: torch.Tensor = self._model(self._observed_image.to(self._device),self._reference_image.to(self._device)).squeeze()
        abst_rel_pose_label: List[int] = self._output_tensor_to_label_list(abst_rel_pose_tensor)
        abst_rel_pose_msg = AbstRelPose()
        rospy.loginfo("abst rel pose: %.3f,%.3f,%.3f", float(abst_rel_pose_tensor[0]),
                float(abst_rel_pose_tensor[1]), float(abst_rel_pose_tensor[2]))
        abst_rel_pose_msg.x, abst_rel_pose_msg.y, abst_rel_pose_msg.yaw = abst_rel_pose_label
        abst_rel_pose_msg.header.stamp = rospy.Time.now()

        return abst_rel_pose_msg

    def _output_tensor_to_label_list(self, abst_rel_pose: torch.Tensor) -> List[int]:
        label_list = [0] * 3
        label_list[0] = 1 if abst_rel_pose[0] > self._param.label_th_x else (-1 if abst_rel_pose[0] < -self._param.label_th_x else 0)
        label_list[1] = 1 if abst_rel_pose[1] > self._param.label_th_y else (-1 if abst_rel_pose[1] < -self._param.label_th_y else 0)
        label_list[2] = 1 if abst_rel_pose[2] > self._param.label_th_yaw else (-1 if abst_rel_pose[2] < -self._param.label_th_yaw else 0)
        
        return label_list

    def process(self) -> None:
        rate = rospy.Rate(self._param.hz)

        while not rospy.is_shutdown():
            if self._observed_image is None or self._reference_image is None: continue
            abst_rel_pose_msg = self._predict_abstract_relative_pose()
            self._abstract_relative_pose_pub.publish(abst_rel_pose_msg)
            rate.sleep()
