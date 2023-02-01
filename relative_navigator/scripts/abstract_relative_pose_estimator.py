#!/usr/bin/python3

import rospy
from sensor_msgs.msg import CompressedImage
from relative_navigator_msgs.msg import AbstRelPose
from std_msgs.msg import Bool

from dataclasses import dataclass
import torch
import numpy as np
import cv2
from typing import Optional, List

from model import AbstRelPosNet
from onehot_conversion import create_onehot_from_output, onehot_decoding


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
                rospy.get_param("weight_path", "/home/amsl/catkin_ws/src/vision-based_navigation/dnn_models/abstrelposnet/weights/dkan_perimeter_0130_duplicate_test_20000/best_loss.pt"),
                rospy.get_param("image_width", 224),
                rospy.get_param("image_height", 224),
            )
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: AbstRelPosNet = AbstRelPosNet().to(self._device)
        self._model.load_state_dict(torch.load(self._param.weight_path, map_location=torch.device(self._device)))
        self._model.eval()

        self._observed_image: Optional[torch.Tensor] = None
        self._reference_image: Optional[torch.Tensor] = None
        self._reaching_target_pose_flag: Optional[bool] = None

        self._observed_image_sub: rospy.Subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",
            CompressedImage, self._observed_image_callback, queue_size=1)
        self._reference_image_sub: rospy.Subscriber = rospy.Subscriber("/reference_image/image_raw/compressed",
            CompressedImage, self._reference_image_callback, queue_size=1)
        self._reaching_target_pose_flag_sub: rospy.Subscriber = rospy.Subscriber("/reaching_target_pose_flag",
                Bool, self._reaching_target_pose_flag_callback, queue_size=1)
        self._abstract_relative_pose_pub: rospy.Publisher = rospy.Publisher("/abstract_relative_pose", AbstRelPose, queue_size=1)
        self._reaching_goal_flag_pub: rospy.Publisher = rospy.Publisher("/reaching_goal_flag", Bool, queue_size=1)

    def _observed_image_callback(self, msg: CompressedImage) -> None:
        self._observed_image = self._compressed_image_to_tensor(msg)

    def _reference_image_callback(self, msg: CompressedImage) -> None:
        self._reference_image = self._compressed_image_to_tensor(msg)

    def _reaching_target_pose_flag_callback(self, msg: Bool) -> None:
        self._reaching_target_pose_flag = msg.data

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
        models_output: torch.Tensor = self._model(self._observed_image.to(self._device),self._reference_image.to(self._device))
        relative_pose_label: List[int] = self._models_output_to_label_list(models_output)
        abst_rel_pose_msg = AbstRelPose()
        rospy.loginfo("relative_pose_label: %d,%d,%d", relative_pose_label[0],relative_pose_label[1], relative_pose_label[2])
        abst_rel_pose_msg.x, abst_rel_pose_msg.y, abst_rel_pose_msg.yaw = relative_pose_label
        abst_rel_pose_msg.header.stamp = rospy.Time.now()

        return abst_rel_pose_msg

    def _models_output_to_label_list(self, models_output: torch.Tensor) -> List[int]:
        one_hot_encoded_output: torch.Tensor = create_onehot_from_output(models_output)
        decoded_output: torch.Tensor = onehot_decoding(one_hot_encoded_output)
        
        return decoded_output.squeeze().tolist()

    def process(self) -> None:
        rate = rospy.Rate(self._param.hz)

        while not rospy.is_shutdown():
            if self._observed_image is None or self._reference_image is None: continue
            if self._reaching_target_pose_flag == None or self._reaching_target_pose_flag:
                abst_rel_pose_msg = self._predict_abstract_relative_pose()
                reaching_goal_flag: bool = (abst_rel_pose_msg.x, abst_rel_pose_msg.y, abst_rel_pose_msg.yaw) == (0, 0, 0)
                if reaching_goal_flag:
                    reaching_goal_flag_msg: Bool = Bool()
                    reaching_goal_flag_msg.data = reaching_goal_flag
                    self._reaching_goal_flag_pub.publish(reaching_goal_flag_msg)

                else:
                    self._abstract_relative_pose_pub.publish(abst_rel_pose_msg)
                    self._reaching_target_pose_flag = False;

            rate.sleep()
