#!/usr/bin/python3

from dataclasses import dataclass
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as F
from typing import Optional, Union, List, Tuple, cast
import networkx as nx

import rospy
from sensor_msgs.msg import CompressedImage
from visualization_msgs.msg import Marker
from std_msgs.msg import String

from my_models import DirectionNet
from .utils import compressed_image_to_tensor, tensor_to_compressed_image, infer
from .topological_map_io import load_topological_map

@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int
    observed_image_width: int
    observed_image_height: int

    hz: float
    batch_size: int
    global_localize_conf_th: float
    candidate_neibors_num: int
    global_candidate_neibors_num: int

    direction_net_weight_path: str
    orientation_net_weight_path: str
    goal_img_path: str
    map_path: str
    observed_image_topic_name: str

class GraphLocalizer:
    def __init__(self) -> None:
        rospy.init_node("graph_localizer")

        self._param: Param = Param(
                cast(int, rospy.get_param("/common/image_width")),
                cast(int, rospy.get_param("/common/image_height")),
                cast(int, rospy.get_param("/common/observed_image_width")),
                cast(int, rospy.get_param("/common/observed_image_height")),

                cast(float, rospy.get_param("~hz")),
                cast(int, rospy.get_param("~batch_size")),
                cast(float, rospy.get_param("~global_localize_conf_th")),
                cast(int, rospy.get_param("~candidate_neibors_num")),
                cast(int, rospy.get_param("~global_candidate_neibors_num")),

                cast(str, rospy.get_param("~direction_net_weight_path")),
                cast(str, rospy.get_param("~orientation_net_weight_path")),
                cast(str, rospy.get_param("~goal_img_path")),
                cast(str, rospy.get_param("~map_path")),
                cast(str, rospy.get_param("~observed_image_topic_name")),
            )

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        # self._device: str = "cpu"

        self._direction_net: DirectionNet = DirectionNet().to(self._device)
        self._direction_net.load_state_dict(torch.load(self._param.direction_net_weight_path, map_location=torch.device(self._device)))
        self._direction_net.eval()

        self._observed_image: Optional[torch.Tensor] = None
        self._observed_image_sub: rospy.Subscriber = rospy.Subscriber(
                self._param.observed_image_topic_name,
                CompressedImage, self._observed_image_callback, queue_size=1)

        self._nearest_node_img_pub = rospy.Publisher("~nearest_node_img/image_raw/compressed",
                CompressedImage, queue_size=1, tcp_nodelay=True)
        self._goal_img_pub = rospy.Publisher("~goal_img/image_raw/compressed",
                CompressedImage, queue_size=1, tcp_nodelay=True)
        self._goal_node_img_pub = rospy.Publisher("~goal_node_img/image_raw/compressed",
                CompressedImage, queue_size=1, tcp_nodelay=True)


        self._nearest_node_marker_pub = rospy.Publisher("~nearest_node_marker", Marker, queue_size=1, tcp_nodelay=True)
        self._goal_node_marker_pub = rospy.Publisher("~goal_node_marker", Marker, queue_size=1, tcp_nodelay=True)

        self._nearest_node_id_pub = rospy.Publisher("~nearest_node_id", String, queue_size=1, tcp_nodelay=True)
        self._goal_node_id_pub = rospy.Publisher("~goal_node_id", String, queue_size=1, tcp_nodelay=True)

        # self._graph: Union[nx.DiGraph, nx.Graph] = load_topological_map(self._param.map_path)
        self._graph: Union[nx.DiGraph, nx.Graph] = load_topological_map(self._param.map_path)
        self._before_node: Optional[str] = None

        goal_img: torch.Tensor = read_image(path=self._param.goal_img_path)[[2,1,0],:,:].unsqueeze(0)/255
        self._goal_image: torch.Tensor = F.resize(img=goal_img,
                size=[self._param.image_height, self._param.image_width])
        # self._goal_image: torch.Tensor = self._graph.nodes["0_180"]['img']

    def _observed_image_callback(self, msg: CompressedImage) -> None:
        self._observed_image = compressed_image_to_tensor(msg,
                (self._param.image_height, self._param.image_width))

    def _create_img_tensor_from_nodes(self, nodes: List[str]) -> torch.Tensor:
        img_tensor = torch.stack([self._graph.nodes[node]['img'].squeeze() for node in nodes])

        return img_tensor

    def _localize_node(self, tgt_img: torch.Tensor, nodes_pool: List[str]) -> Tuple[str, float]:

        tgt_img = tgt_img.squeeze()
        all_imgs_pool: torch.Tensor = self._create_img_tensor_from_nodes(nodes_pool)
        all_same_confs: List[float] = []

        for partial_imgs_pool in torch.split(all_imgs_pool, self._param.batch_size, dim=0):
            tgt_imgs: torch.Tensor = tgt_img.repeat(partial_imgs_pool.shape[0], 1, 1, 1)
            output_probs: torch.Tensor = infer(self._direction_net, self._device,
                                               tgt_imgs, partial_imgs_pool)
            reverse_output_probs: torch.Tensor = infer(self._direction_net, self._device,
                                               partial_imgs_pool, tgt_imgs)
            # partial_same_confs: List[float] = (output_probs[:, 3]).tolist()
            partial_same_confs: List[float] = ((output_probs[:, 3] + reverse_output_probs[:, 3])/2).tolist()
            all_same_confs += partial_same_confs

        max_conf: float = max(all_same_confs)
        max_conf_idx = all_same_confs.index(max_conf)

        return nodes_pool[max_conf_idx], max_conf

    def _predict_nearest_node(self, observed_img: torch.Tensor) -> str:
        nodes_pool: List[str]
        if self._before_node == None:
            nodes_pool = list(self._graph.nodes)
            rospy.loginfo("first global localizing...")
        else:
            nodes_pool = self._get_neibors(self._before_node, self._param.candidate_neibors_num)

        nearest_node, conf = self._localize_node(observed_img, nodes_pool)
        if  conf < self._param.global_localize_conf_th:
            rospy.loginfo("global localizing...")
            nodes_pool = self._get_neibors(self._before_node, self._param.global_candidate_neibors_num)
            nearest_node, _ = self._localize_node(observed_img, nodes_pool)

        # rospy.loginfo(f"conf: {conf}")
        return nearest_node

    def _get_neibors(self, node: str, search_depth: int) -> List[str]:
        nodes_pool = [node]
        for _ in range(search_depth):
            nodes_pool_tmp: List[str] = []
            for node in nodes_pool: 
                nodes_pool_tmp += list(self._graph.succ[node])
                nodes_pool_tmp += list(self._graph.pred[node])
            nodes_pool_tmp = list(set(nodes_pool_tmp))
            nodes_pool += nodes_pool_tmp
            nodes_pool = list(dict.fromkeys(nodes_pool))

        return nodes_pool

    def _predict_goal_node(self, observed_img: torch.Tensor) -> str:
        nodes_pool: List[str] = list(self._graph.nodes)
        goal_node, _ = self._localize_node(observed_img, nodes_pool)

        return goal_node

    def _publish_img(self, img: torch.Tensor, publisher: rospy.Publisher) -> None:
        img_msg: CompressedImage = tensor_to_compressed_image(img,
                # (self._param.image_height, int(self._param.image_height*self._param.original_image_width/self._param.original_image_height)))
                (self._param.observed_image_width, self._param.observed_image_height))
        publisher.publish(img_msg)

    def _create_nearest_marker_node(self, node_name: str) -> Marker:
        marker = Marker()
        marker.type = marker.SPHERE
        marker.action = marker.ADD

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "nearest_node"

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        x, y, _ = self._graph.nodes[node_name]['pose']
        marker.pose.position.x, marker.pose.position.y = x, y
        marker.pose.orientation.w = 1
        
        return marker

    def _create_goal_marker_node(self, node_name: str) -> Marker:
        marker = Marker()
        marker.type = marker.SPHERE
        marker.action = marker.ADD

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal_node"

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        x, y, _ = self._graph.nodes[node_name]['pose']
        marker.pose.position.x, marker.pose.position.y = x, y
        marker.pose.orientation.w = 1
        
        return marker

    def process(self) -> None:

        rospy.loginfo("localizing goal node...")
        goal_node:str = self._predict_goal_node(self._goal_image)
        rospy.loginfo("goal node is localized")

        goal_node_marker: Marker = self._create_goal_marker_node(goal_node)
        goal_node_img: torch.Tensor = self._graph.nodes[goal_node]['img']

        rate = rospy.Rate(self._param.hz)
        while not rospy.is_shutdown():

            self._publish_img(self._goal_image, self._goal_img_pub)
            self._goal_node_id_pub.publish(goal_node)
            self._goal_node_marker_pub.publish(goal_node_marker)
            self._publish_img(goal_node_img, self._goal_node_img_pub)

            if self._observed_image is None: continue

            nearest_node: str = self._predict_nearest_node(self._observed_image)
            # print(f"normal: {infer(self._direction_net, self._device, self._observed_image, self._graph.nodes[nearest_node]['img'])}")
            # print(f"reverwe: {infer(self._direction_net, self._device, self._graph.nodes[nearest_node]['img'],  self._observed_image)}")
            self._publish_img(self._graph.nodes[nearest_node]['img'], self._nearest_node_img_pub)
            nearest_node_marker: Marker = self._create_nearest_marker_node(nearest_node)
            self._nearest_node_marker_pub.publish(nearest_node_marker)
            self._nearest_node_id_pub.publish(nearest_node)

            self._before_node = nearest_node
            self._observed_image = None

            rate.sleep()
