from rosbag import Bag
from typing import List
import torch
import numpy as np

from transformutils import (calc_relative_pose, get_array_2d_from_msg)

from dataset_generator import Config, DatasetGenerator, ReferencePoint

class ConfigForSimpleAbstRelPosNet(Config):
    dist_gap_th: float = 9.0
    yaw_gap_th: float = 0.6
    dist_labelling_th: float = 1.5
    yaw_labelling_th: float = 0.2
    total_travel_dist_gap_th: float = 10.0
    label_ratio_th: float = 0.2525

class DatasetGeneratorForSimpleAbstRelPosNet(DatasetGenerator):
    def __init__(self, config: ConfigForSimpleAbstRelPosNet, bag: Bag, bag_id: int) -> None:
        super().__init__(config, bag, bag_id)
        self._used_index_list: list = []
        self._label_counts: list = [0, 0, 0, 0]

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

        label = self._to_label(relative_odom)
        if label ==  -1: return
        if not self._label_using_judge(label): return

        self._label_counts[label] += 1

        # print(f"label_counts: {self._label_counts}")
        label = [label] + relative_odom
        self._save_data(images, torch.tensor(label, dtype=torch.float32))
        self._used_index_list += [reference_point1.point_index, reference_point2.point_index]

    # label: 0:stop, 1:forwad, 2:left_turn, 3:right_turn, -1:out_range
    def _to_label(self, relative_odom: List[float]) -> int:
        relative_x = relative_odom[0]
        relative_y = relative_odom[1]
        relative_yaw = relative_odom[2]

        dist_th = self._config.dist_labelling_th
        yaw_th = self._config.yaw_labelling_th
        if abs(relative_x) < dist_th and abs(relative_y) < dist_th and abs(relative_yaw) < yaw_th: return 0
        if relative_x >= dist_th and abs(relative_y) < dist_th and abs(relative_yaw) < yaw_th: return 1
        if abs(relative_x) < dist_th and abs(relative_y) < dist_th and relative_yaw >= yaw_th: return 2
        if abs(relative_x) < dist_th and abs(relative_y) < dist_th and relative_yaw <= -yaw_th: return 3

        return -1

    def _label_using_judge(self, label: int) -> bool:
        if sum(self._label_counts) < 20: return True
        if self._label_counts[label] / sum(self._label_counts) > self._config.label_ratio_th: return False
        return True
