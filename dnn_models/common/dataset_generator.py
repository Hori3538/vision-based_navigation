from dataclasses import dataclass
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rosbag import Bag
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, cast
import numpy as np
from copy import copy
import torch
import cv2
import os

from smartargparse import BaseConfig
from transformutils import calc_relative_pose, get_array_2d_from_msg

from training_data import TrainingData

@dataclass(frozen=True)
class Config(BaseConfig):
    bagfiles_dir: str
    output_dir: str
    image_topic_name: str = "/usb_cam/image_raw/compressed"
    odom_topic_name: str = "/odom"
    image_width: int = 224
    image_height: int = 224
    reso_dist: float = 0.1
    reso_yaw: float = 0.1

@dataclass(frozen=True)
class ReferencePoint:
    image: CompressedImage
    pose: Pose

class DatasetGenerator(metaclass=ABCMeta):
    _config: Config
    _bag: Bag
    _reference_points: List[ReferencePoint]
    _data_count: int
    _bag_id: int

    def __init__(self, config: Config, bag: Bag, bag_id: int=0) -> None:
        self._config = config
        self._bag = bag
        self._reference_points  = []
        self._data_count = 0
        self._bag_id = bag_id

    def __call__(self) -> None:
        self._generate_reference_data()
        reference_points1 = copy(self._reference_points)
        reference_points2 = copy(self._reference_points)

        for i, reference_point1 in enumerate(reference_points1):
            for j, reference_point2 in enumerate(reference_points2):
                self._generate_data_from_reference_points(reference_point1, reference_point2)
        

    @staticmethod
    def _calc_relative_odom(odom_from: Odometry, odom_to: Odometry) -> Odometry:
        relative_odom = Odometry()
        relative_odom.pose.pose = calc_relative_pose(odom_from.pose.pose, odom_to.pose.pose)

        return relative_odom

    def _generate_reference_data(self) -> None:
        image: Optional[CompressedImage] = None
        odom: Optional[Odometry] = None
        initial_odom: Optional[Odometry] = None
        prev_odom: Optional[Odometry] = None

        for topic, msg, _ in self._bag.read_messages(
                topics=[self._config.image_topic_name, self._config.odom_topic_name]):
            if topic == self._config.image_topic_name:
                image = cast(CompressedImage, msg) # ?

            if topic == self._config.odom_topic_name:
                odom = cast(Odometry, msg) # ?
                if initial_odom is None:
                    initial_odom = odom
                odom = self._calc_relative_odom(initial_odom, odom)

            if image is None or odom is None: continue

            if prev_odom is None: delta_odom = [np.inf] * 3
            else:
                delta_pose = calc_relative_pose(prev_odom.pose.pose, odom.pose.pose)
                delta_odom = get_array_2d_from_msg(delta_pose)

            delta_dist = np.linalg.norm(delta_odom[:2])
            delta_yaw = abs(delta_odom[2])
            if delta_dist >= self._config.reso_dist or delta_yaw >= self._config.reso_yaw:
                self._reference_points.append(ReferencePoint(image, odom.pose.pose))
                prev_odom = odom

            image = None
            odom = None

    @abstractmethod
    def _generate_data_from_reference_points(self, reference_point1: ReferencePoint,
            reference_point2: ReferencePoint) -> None:
        raise NotImplementedError()

    def _compressed_image_to_tensor(self, compressed_image: CompressedImage) -> torch.Tensor:
        image: np.ndarray = cv2.imdecode(np.frombuffer(compressed_image.data, np.uint8),
                cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self._config.image_height, self._config.image_width))

        return torch.tensor(image)

    def _save_data(self, images: Tuple[torch.Tensor, ...], label: torch.Tensor) -> None:
        self._data_count += 1
        training_data = TrainingData(images[0], images[1], label)
        save_dir = os.path.join(self._config.output_dir, str(self._bag_id))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(training_data, os.path.join(save_dir, f"{self._data_count}.pt"))
