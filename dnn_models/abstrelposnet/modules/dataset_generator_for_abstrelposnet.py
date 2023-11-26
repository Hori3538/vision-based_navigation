from rosbag import Bag
from typing import List
import torch
import numpy as np
from math import radians, atan2
import random

from transformutils import (calc_relative_pose, get_array_2d_from_msg)

from dataset_generator import Config, DatasetGenerator, ReferencePoint

class ConfigForAbstRelPosNet(Config):
    # dist_gap_th: float = 4.5
    dist_gap_th: float = 6.0
    # yaw_gap_th: float = 0.6
    dist_labelling_th: float = 1.5
    yaw_labelling_th: float = 0.2
    total_travel_dist_gap_th: float = 10.0

    # 以下のパラメータは fov_degree/bin_step_degree>=3 になるように設定すること
    fov_degree: int = 78
    bin_step_degree: int = 25

    # set param by user-defined param
    # def __post_init__(self):
        # bin_num: int = self.fov_degree // self.bin_step_degree
        # if bin_num%2 == 0: bin_num -= 1 # bin_numが奇数じゃなければ奇数にする 
        # self._bin_num = bin_num
        # self.yaw_gap_th = radians((bin_num // 2 + 0.5) * self.bin_step_degree)

class DatasetGeneratorForAbstRelPosNet(DatasetGenerator):
    # param determin by user-defined param
    _bin_num: int
    _yaw_gap_th: float

    def __init__(self, config: ConfigForAbstRelPosNet, bag: Bag, bag_id: int) -> None:
        super().__init__(config, bag, bag_id)
        self._config:ConfigForAbstRelPosNet = config
        self._used_index_list: list = []

        # set param by user-defined param
        bin_num: int = config.fov_degree // config.bin_step_degree
        if bin_num%2 == 0: bin_num -= 1 # bin_numが奇数じゃなければ奇数にする 
        self._bin_num = bin_num
        self._yaw_gap_th = radians((bin_num // 2 + 0.5) * config.bin_step_degree)

    def __call__(self) -> None:
        self._generate_reference_data()
        for reference_point1 in self._reference_points:
            furthest_id = self._search_furthest_id(reference_point1)

            while True:
                point2_id: int = random.randint(reference_point1.point_index, furthest_id)
                reference_point2: ReferencePoint = self._reference_points[point2_id]
                if self._is_appropriate_pair(reference_point1, reference_point2): break

            self._generate_data_from_reference_points(reference_point1, reference_point2)

    def _search_furthest_id(self, reference_point: ReferencePoint) -> int:
        for index in range(reference_point.point_index, len(self._reference_points)):
            if abs(self._reference_points[index].total_travel_distance - reference_point.total_travel_distance) > self._config.total_travel_dist_gap_th:
                return index - 1
        return len(self._reference_points) - 1

    def _generate_data_from_reference_points(self, reference_point1: ReferencePoint,
            reference_point2: ReferencePoint) -> None:
        # if reference_point1.point_index in self._used_index_list or reference_point2.point_index in self._used_index_list: return

        images = (self._compressed_image_to_tensor(reference_point1.image),
                self._compressed_image_to_tensor(reference_point2.image))

        relative_odom = get_array_2d_from_msg(calc_relative_pose(reference_point1.pose, reference_point2.pose))

        if not self._is_appropriate_pair(reference_point1, reference_point2): return

        direction_label: List[float] = self._to_direction_label(relative_odom)
        orientation_label: List[float] = self._to_orientation_label(relative_odom)

        self._save_data(images,
                torch.tensor(direction_label, dtype=torch.float32),
                torch.tensor(orientation_label, dtype=torch.float32),
                torch.tensor(relative_odom, dtype=torch.float32))
        self._used_index_list += [reference_point1.point_index, reference_point2.point_index]

    # def _to_abstract(self, relative_odom: List[float]) -> List[float]:
    #     abst_relative_odom = []
    #     for i, value in enumerate(relative_odom):
    #         if i != 2:
    #             dist_th = self._config.dist_labelling_th
    #             if value > dist_th: abst_relative_odom.append(1)
    #             elif value < -dist_th: abst_relative_odom.append(-1)
    #             else: abst_relative_odom.append((0))
    #         else:
    #             yaw_th = self._config.yaw_labelling_th
    #             if value > yaw_th: abst_relative_odom.append(1)
    #             elif value < -yaw_th: abst_relative_odom.append(-1)
    #             else: abst_relative_odom.append((0))
    #
    #     return abst_relative_odom
    # direction label はどのbinが勾配方向かをone-hot形式で表す 勾配がない場合,[-1]=1
    def _to_direction_label(self, relative_pose: List[float]) -> List[float]:
        direction_label: List[float] = [0] * (self._bin_num+1)
        
        dist_gap: np.float32 = np.linalg.norm(relative_pose[:2])
        if dist_gap < self._config.dist_labelling_th:
            direction_label[-1] = 1
            return direction_label

        bin_no: int = self._allocate_yaw_to_bin(atan2(relative_pose[1], relative_pose[0]))
        direction_label[bin_no] = 1

        return direction_label

    # orientation label はどのbinが回転方向かをone-hot形式で表す
    def _to_orientation_label(self, relative_pose: List[float]) -> List[float]:
        orientation_label: List[float] = [0] * self._bin_num

        bin_no: int = self._allocate_yaw_to_bin(relative_pose[2])
        orientation_label[bin_no] = 1

        return orientation_label
    
    # def _is_appropriate_pair(self, relative_pose: List[float]) -> bool:
    def _is_appropriate_pair(self, reference_point1: ReferencePoint, reference_point2: ReferencePoint) -> bool:
        # この場合atan2がバグるので先に処理
        if reference_point1.point_index == reference_point2.point_index: return True

        if abs(reference_point1.total_travel_distance - reference_point2.total_travel_distance) > \
            self._config.total_travel_dist_gap_th: return False

        relative_pose= get_array_2d_from_msg(calc_relative_pose(reference_point1.pose, reference_point2.pose))
        dist_gap: np.float32 = np.linalg.norm(relative_pose[:2])
        if dist_gap > self._config.dist_gap_th: return False

        direction_yaw: float = atan2(relative_pose[1], relative_pose[0])
        if abs(direction_yaw) > self._yaw_gap_th: return False

        orientation_yaw: float = relative_pose[2]
        if abs(orientation_yaw) > self._yaw_gap_th: return False
        
        return True

    # bin no は0はじまり
    # yaw がでかいほうが若いbin no に割り当てられる
    # bin の数は必ず奇数であり，真ん中のbinのrangeは[bin_range/2, bin_range/2)
    # 各binの範囲について，最大値側は閉区間，最初値側は開区間．ただし，最後のbinは両方閉区間
    def _allocate_yaw_to_bin(self, yaw: float) -> int:
        # この例外は_is_appropriate_pair()が弾かれるのでいまのところ起きない
        if abs(yaw) > self._yaw_gap_th: return -1

        # binの割当に使うboarderをbin no 0 の最小値で初期化
        boarder_yaw: float = radians((self._bin_num // 2 - 0.5) * self._config.bin_step_degree)
        for bin_no in range(self._bin_num):
            if bin_no < self._bin_num-1 and yaw > boarder_yaw or \
                    bin_no == self._bin_num-1 and yaw >= boarder_yaw:
                return bin_no 
            boarder_yaw -= radians(self._config.bin_step_degree)
        
        return -1 # ここに到達することはないがlinterが怒るので適当に返す
