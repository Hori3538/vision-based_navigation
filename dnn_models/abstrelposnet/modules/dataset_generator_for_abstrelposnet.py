from rosbag import Bag
from typing import List
import torch
import numpy as np

from transformutils import (calc_relative_pose, get_array_2d_from_msg)

import sys
sys.path.append("../../common")
from dataset_generator import Config, DatasetGenerator, ReferencePoint

class ConfigForAbstRelPosNet(Config):
    dist_gap_th: float = 4.5
    # dist_gap_th: float = 6.0
    yaw_gap_th: float = 0.6
    dist_labelling_th: float = 1.5
    yaw_labelling_th: float = 0.2
    total_travel_dist_gap_th: float = 18.0

class DatasetGeneratorForAbstRelPosNet(DatasetGenerator):
    def __init__(self, config: ConfigForAbstRelPosNet, bag: Bag, bag_id: int) -> None:
        super().__init__(config, bag, bag_id)
        self._used_index_list: list = []

    def _generate_data_from_reference_points(self, reference_point1: ReferencePoint,
            reference_point2: ReferencePoint) -> None:
        # if reference_point1.point_index in self._used_index_list or reference_point2.point_index in self._used_index_list: return

        images = (self._compressed_image_to_tensor(reference_point1.image),
                self._compressed_image_to_tensor(reference_point2.image))

        relative_pose = calc_relative_pose(reference_point1.pose, reference_point2.pose)
        relative_odom = get_array_2d_from_msg(relative_pose)

        dist_gap = np.linalg.norm(relative_odom[:2])
        yaw_gap = abs(relative_odom[2])
        if dist_gap > self._config.dist_gap_th or yaw_gap > self._config.yaw_gap_th \
            or abs(reference_point1.total_travel_distance - reference_point2.total_travel_distance) > self._config.total_travel_dist_gap_th:
            return 

        abst_relative_odom = self._to_abstract(relative_odom)
        label = abst_relative_odom + relative_odom

        self._save_data(images, torch.tensor(label, dtype=torch.float32))
        self._used_index_list += [reference_point1.point_index, reference_point2.point_index]

    def _to_abstract(self, relative_odom: List[float]) -> List[float]:
        abst_relative_odom = []
        for i, value in enumerate(relative_odom):
            if i != 2:
                dist_th = self._config.dist_labelling_th
                if value > dist_th: abst_relative_odom.append(1)
                elif value < -dist_th: abst_relative_odom.append(-1)
                else: abst_relative_odom.append((0))
            else:
                yaw_th = self._config.yaw_labelling_th
                if value > yaw_th: abst_relative_odom.append(1)
                elif value < -yaw_th: abst_relative_odom.append(-1)
                else: abst_relative_odom.append((0))
        
        return abst_relative_odom
