#!/usr/bin/python3

from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Union, List, cast
import networkx as nx

import rospy
from sensor_msgs.msg import CompressedImage
from visualization_msgs.msg import Marker
from relative_navigator_msgs.msg import RelPoseLabel


from directionnet import DirectionNet
from orientationnet import OrientationNet
from .utils import compressed_image_to_tensor, infer
from .topological_map_io import load_topological_map

@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int

    direction_net_weight_path: str
    orientation_net_weight_path: str
    observed_image_topic_name: str

    hz: float
    map_path: str
    batch_size: int

class GraphLocalizer:
    def __init__(self) -> None:
        rospy.init_node("graph_localizer")

        self._param: Param = Param(

                rospy.get_param("~image_width", 224),
                rospy.get_param("~image_height", 224),
                rospy.get_param("~direction_net_weight_path", ""),
                rospy.get_param("~orientation_net_weight_path", ""),
                rospy.get_param("~observed_image_topic_name", "/usb_cam/image_raw/compressed"),
                rospy.get_param("~hz", 10),
                cast(str, rospy.get_param("~map_path")),
                rospy.get_param("~batch_size", 32),
            )

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._direction_net: DirectionNet = DirectionNet().to(self._device)
        self._direction_net.load_state_dict(torch.load(self._param.direction_net_weight_path, map_location=torch.device(self._device)))
        self._direction_net.eval()

        self._observed_image: Optional[torch.Tensor] = None
        self._observed_image_sub: rospy.Subscriber = rospy.Subscriber(
                self._param.observed_image_topic_name,
                CompressedImage, self._observed_image_callback, queue_size=1)

        self._graph: Union[nx.DiGraph, nx.Graph] = load_topological_map(self._param.map_path)

        self._nearest_node_pub = rospy.Publisher("~nearest_node", Marker, queue_size=1, tcp_nodelay=True)
        self._goal_node_pub = rospy.Publisher("~goal_node", Marker, queue_size=1, tcp_nodelay=True)

    def _observed_image_callback(self, msg: CompressedImage) -> None:
        self._observed_image = compressed_image_to_tensor(msg,
                (self._param.image_height, self._param.image_width))

    def _create_img_tensor_from_nodes(self, nodes: List[str]) -> torch.Tensor:
        img_tensor = torch.stack([self._graph.nodes[node]['img'].squeeze() for node in nodes])

        return img_tensor

    def _localize_node(self, tgt_node: str, nodes_pool: List[str]) -> str:

        tgt_img: torch.Tensor = self._graph.nodes[tgt_node]['img'].squeeze()
        all_imgs_pool: torch.Tensor = self._create_img_tensor_from_nodes(nodes_pool)
        all_same_confs: List[float] = []

        for partial_imgs_pool in torch.split(all_imgs_pool, self._param.batch_size, dim=0):
            tgt_imgs: torch.Tensor = tgt_img.repeat(partial_imgs_pool.shape[0], 1, 1, 1)
            output_probs: torch.Tensor = infer(self._direction_net, self._device,
                                               tgt_imgs, partial_imgs_pool)
            partial_same_confs: List[float] = output_probs[:, 3].tolist()
            all_same_confs += partial_same_confs

        max_conf_idx = all_same_confs.index(max(all_same_confs))

        return nodes_pool[max_conf_idx]

    def process(self) -> None:
        print(f"{self._localize_node('0_0', list(self._graph.nodes))}")
        pass
