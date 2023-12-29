from dataclasses import dataclass
from enum import Flag, auto
from typing import Any, cast

import networkx as nx
import numpy as np
import rosbag
import rospy
import torch
import torch.nn.functional as F
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from torch.jit._script import ScriptModule
from torch.jit._serialization import load as load_jit
from transformutils import calc_relative_pose, get_array_2d_from_msg

from directionnet import DirectionNet
from orientationnet import OrientationNet

from .topological_map_io import load_topological_map, save_topological_map
from .utils import compressed_image_to_tensor
@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int
    # torch_script_path: str
    direction_net_weight_path: str
    orientation_net_weight_path: str
    bagfile_path: str
    map_save_path: str
    use_gt_pose: bool 
    image_topic_name: str
    odom_topic_name: str
    gt_pose_topic_name: str

class TopologicalMapper:
    def __init__(self) -> None:
        rospy.init_node("topological_mapper")
        self._param = Param(
            cast(int, rospy.get_param("~image_width", 224)),
            cast(int, rospy.get_param("~image_height", 224)),
            cast(str, rospy.get_param("~direction_net_weight_path")),
            cast(str, rospy.get_param("~orientation_net_weight_path")),
            cast(str, rospy.get_param("~bagfile_dir")),
            cast(str, rospy.get_param("~map_save_path")),
            cast(bool, rospy.get_param("~use_gt_pose", False)),
            cast(str, rospy.get_param("~image_topic_name")),
            cast(str, rospy.get_param("~odom_topic_name")),
            cast(str, rospy.get_param("~gt_pose_topic_name", default="")),
        )

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._direction_net: DirectionNet = DirectionNet().to(self._device)
        self._direction_net.load_state_dict(torch.load(self._param.direction_net_weight_path, map_location=torch.device(self._device)))
        self._direction_net.eval()

        self._orientation_net: OrientationNet = OrientationNet().to(self._device)
        self._orientation_net.load_state_dict(torch.load(self._param.orientation_net_weight_path, map_location=torch.device(self._device)))
        self._orientation_net.eval()

        self._graph = nx.DiGraph()

        def _are
