from dataclasses import dataclass
from enum import Flag, auto
from typing import Any, cast, List, Optional
from glob import iglob
import os
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import rospy
import torch
import torch.nn.functional as F
from torch.jit._script import ScriptModule
from torch.jit._serialization import load as load_jit

from rosbag import Bag
import rosnode
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry

from transformutils import calc_relative_pose, get_array_2d_from_msg

from directionnet import DirectionNet
from orientationnet import OrientationNet

from .topological_map_io import save_topological_map
from .utils import compressed_image_to_tensor, infer, msg_to_pose
@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int
    # torch_script_path: str
    direction_net_weight_path: str
    orientation_net_weight_path: str
    bagfiles_dir: str
    map_save_path: str
    # use_gt_pose: bool 
    image_topic_name: str
    pose_topic_name: str
    pose_topic_type: str
    # odom_topic_name: str
    # gt_pose_topic_name: str

class TopologicalMapper:
    def __init__(self) -> None:
        rospy.init_node("topological_mapper")
        self._param = Param(
            cast(int, rospy.get_param("~image_width", 224)),
            cast(int, rospy.get_param("~image_height", 224)),
            cast(str, rospy.get_param("~direction_net_weight_path")),
            cast(str, rospy.get_param("~orientation_net_weight_path")),
            cast(str, rospy.get_param("~bagfiles_dir")),
            cast(str, rospy.get_param("~map_save_path")),
            # cast(bool, rospy.get_param("~use_gt_pose", False)),
            cast(str, rospy.get_param("~image_topic_name", "/grasscam/image_raw/compressed")),
            cast(str, rospy.get_param("~pose_topic_name", "/whill/odom")),
            cast(str, rospy.get_param("~pose_topic_type", "Odometry")),
            # cast(str, rospy.get_param("~odom_topic_name")),
            # cast(str, rospy.get_param("~gt_pose_topic_name")),
        )

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._direction_net: DirectionNet = DirectionNet().to(self._device)
        self._direction_net.load_state_dict(torch.load(self._param.direction_net_weight_path, map_location=torch.device(self._device)))
        self._direction_net.eval()

        self._orientation_net: OrientationNet = OrientationNet().to(self._device)
        self._orientation_net.load_state_dict(torch.load(self._param.orientation_net_weight_path, map_location=torch.device(self._device)))
        self._orientation_net.eval()

        # self._topics_to_be_read = [self._param.image_topic_name]
        # if self._param.use_gt_pose: self._topics_to_be_read += [self._param.gt_pose_topic_name]
        # else: self._topics_to_be_read += [self._param.odom_topic_name]

        self._graph = nx.DiGraph()
        # self._node_count = 0

    def _are_diff_nodes(self, src_img: torch.Tensor, tgt_img: torch.Tensor) -> bool:
        direction_probs: torch.Tensor = infer(self._direction_net, self._device,
                                              src_img, tgt_img)

        direction_max_idx = direction_probs.max(0).indices
        
        # ラベルが変位ありの場合違うノードとする
        if direction_max_idx <= 2: return True
        return False

    # def _add_node(graph: nx.DiGraph, img: torch.Tensor, pose: List[float]):
    #     graph.add_node()

    def _add_nodes_from_bag(self, graph: nx.DiGraph, bag: Bag, bag_id: int) -> None:
    # def _add_nodes_from_bag(bag: Bag):
        # prev_img: Optional[CompressedImage] = None
        prev_img: Optional[torch.Tensor] = None
        # img: Optional[CompressedImage] = None
        img: Optional[torch.Tensor] = None
        # odom: Optional[Odometry] = None
        pose: Optional[Pose] = None
        # initial_odom: Optional[Odometry] = None
        initial_pose: Optional[Pose] = None
        # prev_odom: Optional[Odometry] = None
        node_count = 0

        for topic, msg, _ in bag.read_messages(
                topics=[self._param.image_topic_name, self._param.pose_topic_name]):
            if topic == self._param.image_topic_name:
                # img = cast(CompressedImage, msg) # Optional[CompressedImage] -> CompressedImage
                # img = msg # Optional[CompressedImage] -> CompressedImage
                img = compressed_image_to_tensor(msg,
                        (self._param.image_height, self._param.image_width))

            if topic == self._param.pose_topic_name:
                # odom = cast(Odometry, msg) # Optional[Odometry] -> Odometry
                pose = msg_to_pose(msg, self._param.pose_topic_type)
                if initial_pose is None: initial_pose = pose

                pose = calc_relative_pose(initial_pose, pose)

            if img is None or pose is None: continue

            if prev_img is None:
                prev_img  = img
                continue
            
            if self._are_diff_nodes(cast(torch.Tensor, prev_img), cast(torch.Tensor, img)):
                pose_list = get_array_2d_from_msg(pose)
                node_id = str(bag_id) + "_" + str(node_count)

                graph.add_node(node_id, img=img, pose=pose_list)

                node_count +=1
                prev_img = img

            img = None
            pose = None
        rospy.loginfo(f"bug id: {bag_id} is finished\n {node_count} nodes is added.")

    def process(self) -> None:
        for i, bagfile_path in enumerate(iglob(os.path.join(self._param.bagfiles_dir, "*"))):
            bag: Bag = Bag(bagfile_path)
            self._add_nodes_from_bag(self._graph, bag, i)
        
        save_topological_map(self._param.map_save_path, self._graph)
        
        nx.draw_networkx(self._graph)
        plt.show()
        rospy.loginfo("Process fnished")
        rosnode.kill_nodes("topological_mapper")
