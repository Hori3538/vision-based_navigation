from dataclasses import dataclass
import rospy
from sensor_msgs.msg import CompressedImage
from rospy.client import get_param
import torch

import sys
sys.path.append("...")
from dnn_models.abstrelposnet.model import AbstRelPosNet
from typing import Optional, List

@dataclass(frozen=True)
class Param:
    hz: float
    weight_path: str

class AbstractRelativePoseEstimator:
    def __init__(self) -> None:
        rospy.init_node("abstract_relative_pose_estimator")

        self._param: Param = Param(
                rospy.get_param("hz", 10),
                rospy.get_param("weight_path", "~/catkin_ws/src/vision-based_navigation/dnn_models/abstrelposnet/weights/dkan_perimeter_0930_1010/best_loss.pt")
            )
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: AbstRelPosNet = AbstRelPosNet().to(self._device)
        self._model.load_state_dict(torch.load(self._param.weight_path, map_location=torch.device(self._device)))
        self._model.eval()
        self._observed_image: Optional[torch.Tensor] = None
        self._reference_image: Optional[CompressedImage] = None

        self._observed_image_sub: rospy.Subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",
            CompressedImage, self._observed_image_callback, queue_size=1)
        self._reference_image_sub: rospy.Subscriber = rospy.Subscriber("/reference_image/image_raw/compressed",
            CompressedImage, self._reference_image_callback, queue_size=1)

    def _observed_image_callback(self, msg: CompressedImage) -> None:
        pass
    def _reference_image_callback(self, msg: CompressedImage) -> None:
        pass
