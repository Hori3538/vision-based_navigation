#!/usr/bin/python3

from dataclasses import dataclass
import copy
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
    reserving_node_num: int

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
                cast(int, rospy.get_param("~reserving_node_num")),

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

        self._graph: Union[nx.DiGraph, nx.Graph] = load_topological_map(self._param.map_path)
        self._before_nodes: Optional[List[str]] = None

        goal_img: torch.Tensor = read_image(path=self._param.goal_img_path)[[2,1,0],:,:].unsqueeze(0)/255
        self._goal_image: torch.Tensor = F.resize(img=goal_img,
                size=[self._param.image_height, self._param.image_width])

    def _observed_image_callback(self, msg: CompressedImage) -> None:
        self._observed_image = compressed_image_to_tensor(msg,
                (self._param.image_height, self._param.image_width))

    def _create_img_tensor_from_nodes(self, nodes: List[str]) -> torch.Tensor:
        img_tensor = torch.stack([self._graph.nodes[node]['img'].squeeze() for node in nodes])

        return img_tensor

    def _localize_node(self, tgt_img: torch.Tensor, nodes_pool: List[str]) -> Tuple[List[str], List[float]]:

        all_imgs_pool: torch.Tensor = self._create_img_tensor_from_nodes(nodes_pool)
        all_same_confs = self._pred_same_confs(tgt_img, all_imgs_pool)

        all_same_confs_tensor = torch.tensor(all_same_confs)
        max_confs, max_indices  = torch.topk(all_same_confs_tensor, self._param.reserving_node_num)
        # print(f"max n nodes: {[nodes_pool[i] for i in max_indices]}")
        # print(f"max n nodes conf: {max_confs}")


        return [nodes_pool[i] for i in max_indices], max_confs

    def _localize_node2(self, tgt_img: torch.Tensor, nodes_pool: List[str]) -> Tuple[List[str], List[float]]:

        if(len(nodes_pool) != self._graph.number_of_nodes()):
            nodes_pool_including_neighbor: List[str] = self._get_neighbors(nodes_pool, 1)
        else: 
            nodes_pool_including_neighbor: List[str] = nodes_pool

        all_imgs_pool: torch.Tensor = self._create_img_tensor_from_nodes(nodes_pool_including_neighbor)
        all_same_confs = self._pred_same_confs(tgt_img, all_imgs_pool)
        self._register_conf_to_nodes(nodes_pool_including_neighbor, all_same_confs)

        avg_same_confs = self._calc_avg_confs_of_neighbors(nodes_pool)
        avg_same_confs_tensor = torch.tensor(avg_same_confs)

        max_confs, max_indices  = torch.topk(avg_same_confs_tensor, 5)
        print(f"max 5 nodes avg: {[nodes_pool[i] for i in max_indices]}")
        print(f"max 5 nodes avgconf: {max_confs}")

        return [nodes_pool[i] for i in max_indices], max_confs

    def _predict_nearest_nodes(self, observed_img: torch.Tensor) -> List[str]:
        nodes_pool: List[str]
        if self._before_nodes == None:
            nodes_pool = list(self._graph.nodes)
            rospy.loginfo("first global localizing...")
        else:
            nodes_pool = self._get_neighbors(self._before_nodes, self._param.candidate_neibors_num)

        nearest_nodes, confs = self._localize_node(observed_img, nodes_pool)
        # nearest_nodes, confs = self._localize_node2(observed_img, nodes_pool)
        if  confs[0] < self._param.global_localize_conf_th:
            rospy.loginfo("wider localizing...")
            nodes_pool = self._get_neighbors(self._before_nodes, self._param.global_candidate_neibors_num)
            nearest_nodes, _ = self._localize_node(observed_img, nodes_pool)
            # nearest_nodes, _ = self._localize_node2(observed_img, nodes_pool)

        # rospy.loginfo(f"conf: {confs[0]}")
        return nearest_nodes

    def _get_neighbors(self, node: List[str], search_depth: int) -> List[str]:
        nodes_pool = copy.deepcopy(node)
        for _ in range(search_depth):
            nodes_pool_tmp: List[str] = []
            for node in nodes_pool: 
                nodes_pool_tmp += list(self._graph.succ[node])
                nodes_pool_tmp += list(self._graph.pred[node])
            nodes_pool_tmp = list(set(nodes_pool_tmp))
            nodes_pool += nodes_pool_tmp
            nodes_pool = list(dict.fromkeys(nodes_pool))

        return nodes_pool

    def _register_conf_to_nodes(self, nodes: List[str], confs: List[float]) -> None:
        for node, conf in zip(nodes, confs):
            self._graph.nodes[node]["conf"] = conf

    def _calc_avg_confs_of_neighbors(self, nodes: List[str]) -> List[float]:
        avg_confs = []
        for node in nodes:
            sum_conf: float = 0.0
            neibghbors: List[str] = self._get_neighbors([node], 1)
            for neighbor_node in neibghbors:
                sum_conf += self._graph.nodes[neighbor_node]["conf"]

            avg_confs.append(sum_conf / len(neibghbors))

        return avg_confs

    def _pred_same_confs(self, src_img: torch.Tensor, tgt_imgs: torch.Tensor) -> List[float]:

        src_img = src_img.squeeze()
        all_same_confs: List[float] = []

        for partial_tgt_imgs in torch.split(tgt_imgs, self._param.batch_size, dim=0):
            src_imgs: torch.Tensor = src_img.repeat(partial_tgt_imgs.shape[0], 1, 1, 1)
            output_probs: torch.Tensor = infer(self._direction_net, self._device,
                                               src_imgs, partial_tgt_imgs)
            reverse_output_probs: torch.Tensor = infer(self._direction_net, self._device,
                                               partial_tgt_imgs, src_imgs)
            partial_same_confs: List[float] = ((output_probs[:, 3] + reverse_output_probs[:, 3])/2).tolist()
            all_same_confs += partial_same_confs

        return all_same_confs

    def _predict_goal_node(self, observed_img: torch.Tensor) -> str:
        nodes_pool: List[str] = list(self._graph.nodes)
        candidate_goal_nodes, _ = self._localize_node(observed_img, nodes_pool)
        # candidate_goal_nodes, _ = self._localize_node2(observed_img, nodes_pool)

        return candidate_goal_nodes[0]

    def _publish_img(self, img: torch.Tensor, publisher: rospy.Publisher) -> None:
        img_msg: CompressedImage = tensor_to_compressed_image(img,
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

            nearest_nodes: List[str] = self._predict_nearest_nodes(self._observed_image)
            nearest_node: str = nearest_nodes[0]
            self._publish_img(self._graph.nodes[nearest_node]['img'], self._nearest_node_img_pub)
            nearest_node_marker: Marker = self._create_nearest_marker_node(nearest_node)
            self._nearest_node_marker_pub.publish(nearest_node_marker)
            self._nearest_node_id_pub.publish(nearest_node)

            self._before_nodes = nearest_nodes
            self._observed_image = None

            rate.sleep()
