from typing import List, Tuple, Type, Union, TypeVar, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose

from directionnet import DirectionNet
from orientationnet import OrientationNet

def compressed_image_to_tensor(
        compressed_image: CompressedImage, image_shape: Tuple[int, int]) -> torch.Tensor:

    image: np.ndarray = cv2.imdecode(np.frombuffer(
        compressed_image.data, np.uint8), cv2.IMREAD_COLOR)  # type: ignore
    image: np.ndarray = cv2.resize(image, image_shape)
    image_tensor = torch.tensor(
        image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255

    return image_tensor

def tensor_to_compressed_image(
        tensor: torch.Tensor, image_shape: Tuple[int, int]) -> CompressedImage:

    image: np.ndarray = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
    image: np.ndarray = cv2.resize(image, image_shape)
    image_msg = CompressedImage()
    image_msg.format = "jpeg"
    image_msg.data = np.array(cv2.imencode(".jpg", image)[1]).tostring()

    return image_msg

def infer(model: Union[DirectionNet, OrientationNet], device: str, 
        src_img: torch.Tensor, tgt_img: torch.Tensor) -> torch.Tensor:
    model_output = model(src_img.to(device), tgt_img.to(device)).squeeze()
    output_probs = F.softmax(model_output, 0)

    return output_probs

# T = TypeVar("T", Odometry, PoseStamped)
# def msg_to_pose(msg: T) -> Pose:
#     if isinstance(msg, Odometry): return msg.pose.pose
#     if isinstance(msg, PoseStamped): return msg.pose
#     print("hoge")

def msg_to_pose(msg: Any, type: str) -> Pose:
    if type == "Odometry": return msg.pose.pose
    if type == "PoseStamped": return msg.pose
    assert False, "Unknown message type"
