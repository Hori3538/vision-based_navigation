from numpy import dtype, float32
from rosbag import Bag
from typing import List
import torch

from transformutils import (calc_relative_pose, get_array_2d_from_msg)

import sys
sys.path.append("../../common")
from dataset_generator import Config, DatasetGenerator, ReferencePoint

class ConfigForAbstRelPosNet(Config):
    dist_gap_th: float = 5.0
    yaw_gap_th: float = 0.8
    dist_labelling_th: float = 1.0
    yaw_labelling_th: float = 0.3

class DatasetGeneratorForAbstRelPosNet(DatasetGenerator):
    def __init__(self, config: ConfigForAbstRelPosNet, bag: Bag, bag_id: int) -> None:
        super().__init__(config, bag, bag_id)

    def _generate_data_from_reference_points(self, reference_point1: ReferencePoint,
            reference_point2: ReferencePoint) -> None:
        images = (self._compressed_image_to_tensor(reference_point1.image),
                self._compressed_image_to_tensor(reference_point2.image))

        relative_pose = calc_relative_pose(reference_point1.pose, reference_point2)
        relative_odom = get_array_2d_from_msg(relative_pose)
        abst_relative_odom = self._to_abstract(relative_odom)
        label = abst_relative_odom + relative_odom

        self._save_data(images, torch.tensor(label, dtype=torch.float32))

    def _to_abstract(self, relative_odom: List[float]) -> List[float]:
        abst_relative_odom = []
        for i, value in enumerate(relative_odom):
            if i != 2:
                dist_th = self._config.dist_labelling_th
                if value > dist_th: abst_relative_odom.append(1)
                if -dist_th <= value <= dist_th: abst_relative_odom.append((0))
                if value < -dist_th: abst_relative_odom.append(-1)
            else:
                yaw_th = self._config.yaw_labelling_th
                if value > yaw_th: abst_relative_odom.append(1)
                if -yaw_th <= value <= yaw_th: abst_relative_odom.append((0))
                if value < -yaw_th: abst_relative_odom.append(-1)
        
        return abst_relative_odom


