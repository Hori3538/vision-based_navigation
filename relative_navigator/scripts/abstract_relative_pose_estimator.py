from dataclasses import dataclass
import rospy
from sensor_msgs.msg import CompressedImage
import torch
import numpy as np
import cv2

import sys
sys.path.append("...")
from dnn_models.abstrelposnet.model import AbstRelPosNet
from typing import Optional, List

from relative_navigator_msgs import AbstRelPose

@dataclass(frozen=True)
class Param:
    hz: float
    weight_path: str
    image_width: int
    image_height: int

class AbstractRelativePoseEstimator:
    def __init__(self) -> None:
        rospy.init_node("abstract_relative_pose_estimator")

        self._param: Param = Param(
                rospy.get_param("hz", 10),
                rospy.get_param("weight_path", "~/catkin_ws/src/vision-based_navigation/dnn_models/abstrelposnet/weights/dkan_perimeter_0930_1010/best_loss.pt"),
                rospy.get_param("image_width", 224),
                rospy.get_param("image_height", 224)
            )
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: AbstRelPosNet = AbstRelPosNet().to(self._device)
        self._model.load_state_dict(torch.load(self._param.weight_path, map_location=torch.device(self._device)))
        self._model.eval()
        self._observed_image: Optional[torch.Tensor] = None
        self._reference_image: Optional[torch.Tensor] = None

        self._observed_image_sub: rospy.Subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",
            CompressedImage, self._observed_image_callback, queue_size=1)
        self._reference_image_sub: rospy.Subscriber = rospy.Subscriber("/reference_image/image_raw/compressed",
            CompressedImage, self._reference_image_callback, queue_size=1)

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
        abst_rel_pose_msg: AbstRelPose
        # abst_rel_pose_msg.x, abst_rel_pose_msg.y, abst_rel_pose_msg.yaw = 
