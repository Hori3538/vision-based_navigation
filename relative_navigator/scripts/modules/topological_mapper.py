from dataclasses import dataclass
import re
from typing import cast, Optional, Union, List
import os
import math
from glob import iglob

import networkx as nx
import rospy
import torch

from rosbag import Bag
import rosnode
from geometry_msgs.msg import Pose

from transformutils import get_array_2d_from_msg, calc_relative_pose

from .topological_map_io import save_topological_map, load_topological_map, save_nodes_as_img
from .utils import compressed_image_to_tensor, infer, msg_to_pose

@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int

    direction_net_path: str
    orientation_net_path: str
    bagfiles_dir: str
    map_save_dir: str
    map_name: str

    image_topic_name: str
    pose_topic_name: str
    pose_topic_type: str

    use_model: int

    divide_conf_th: float # node生成時，隣接画像間の "same" confがこれより小さい時nodeを追加する
    connect_conf_th: float # edge 生成時，node間の "same" conf がこれより大きい時edgeを追加する
    edge_num_th: int # >= 1

    divide_gt_dist_th: float
    connect_gt_dist_th: float

class TopologicalMapper:
    def __init__(self) -> None:
        rospy.init_node("topological_mapper")
        self._param = Param(
            cast(int, rospy.get_param("~image_width", 224)),
            cast(int, rospy.get_param("~image_height", 224)),

            cast(str, rospy.get_param("~direction_net_path")),
            cast(str, rospy.get_param("~orientation_net_path")),
            cast(str, rospy.get_param("~bagfiles_dir")),
            cast(str, rospy.get_param("~map_save_dir")),
            cast(str, rospy.get_param("~map_name")),

            cast(str, rospy.get_param("~image_topic_name", "/grasscam/image_raw/compressed")),
            cast(str, rospy.get_param("~pose_topic_name", "/whill/odom")),
            cast(str, rospy.get_param("~pose_topic_type", "Odometry")),

            cast(int, rospy.get_param("~use_model", 1)),

            cast(float, rospy.get_param("~divide_conf_th", 0.7)),
            cast(float, rospy.get_param("~connect_conf_th", 0.6)),
            cast(int, rospy.get_param("~edge_num_th", 3)),

            cast(float, rospy.get_param("~divide_gt_dist_th")),
            cast(float, rospy.get_param("~connect_gt_dist_th")),
        )

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._direction_net: torch.ScriptModule = torch.jit.load(self._param.direction_net_path).eval().to(self._device)
        self._orientation_net: torch.ScriptModule = torch.jit.load(self._param.orientation_net_path).eval().to(self._device)
 

        # self._graph = nx.DiGraph()
        self._graph: Optional[nx.DiGraph] = None

    def _are_different_enough(self, src_img: torch.Tensor, tgt_img: torch.Tensor) -> bool:

        direction_probs: torch.Tensor = infer(self._direction_net, self._device,src_img, tgt_img).squeeze()
        orientation_probs: torch.Tensor = infer(self._orientation_net, self._device,src_img, tgt_img).squeeze()
        if direction_probs[3] < self._param.divide_conf_th: return True
        if orientation_probs[1] < self._param.divide_conf_th: return True

        return False

    def _add_nodes_from_bag(self, graph: Union[nx.DiGraph, nx.Graph], bag: Bag, bag_id: int) -> None:

        rospy.loginfo(f"start adding node of bag id: {bag_id}")
        prev_img: Optional[torch.Tensor] = None
        img: Optional[torch.Tensor] = None
        pose: Optional[Pose] = None
        node_count = 0

        for topic, msg, _ in bag.read_messages(
                topics=[self._param.image_topic_name, self._param.pose_topic_name]):
            if topic == self._param.image_topic_name:
                img = compressed_image_to_tensor(msg,(self._param.image_height, self._param.image_width))

            if topic == self._param.pose_topic_name:
                pose = msg_to_pose(msg, self._param.pose_topic_type)

            if img is None or pose is None: continue
            if prev_img is None:
                prev_img  = img
                continue
            
            if self._are_different_enough(cast(torch.Tensor, prev_img), cast(torch.Tensor, img)):
                pose_list = get_array_2d_from_msg(pose)
                node_id = str(bag_id) + "_" + str(node_count)

                graph.add_node(node_id, img=img, pose=pose_list)

                node_count +=1
                prev_img = img

            img = None
            pose = None
        rospy.loginfo(f"bag id: {bag_id} is finished. {node_count} nodes is added.")

    def _add_nodes_from_bag_by_gt_pose(self, graph: nx.Graph, bag: Bag, bag_id: int) -> None:

        rospy.loginfo(f"start adding node of bag id: {bag_id}")
        pose: Optional[Pose] = None
        before_pose: Optional[Pose] = None
        node_count = 0

        for topic, msg, _ in bag.read_messages(
                topics=[self._param.pose_topic_name]):

            if topic == self._param.pose_topic_name:
                pose = msg_to_pose(msg, self._param.pose_topic_type)
            if pose is None: continue

            if before_pose is None:
                before_pose = pose
                continue

            relative_pose = calc_relative_pose(before_pose, pose)
            if math.hypot(relative_pose.position.x, relative_pose.position.y) > self._param.divide_gt_dist_th:
            # if self._are_different_enough(cast(torch.Tensor, prev_img), cast(torch.Tensor, img)):
                pose_list = get_array_2d_from_msg(pose)
                node_id = str(bag_id) + "_" + str(node_count)

                graph.add_node(node_id, pose=pose_list)

                node_count +=1
                before_pose = pose

            pose = None
        rospy.loginfo(f"bag id: {bag_id} is finished. {node_count} nodes is added.")

    # pred conf of "same" label
    def _pred_same_conf(self, src_img: torch.Tensor, tgt_img: torch.Tensor) -> float:

        direction_probs: torch.Tensor = infer(self._direction_net, self._device,src_img, tgt_img).squeeze()

        return float(direction_probs[3])

    def _add_edges(self, graph: Union[nx.DiGraph, nx.Graph]) -> None:

        rospy.loginfo(f"start adding normal edge")
        for src_node, src_img in dict(graph.nodes.data('img')).items():
            for tgt_node, tgt_img in dict(graph.nodes.data('img')).items():
                if src_node == tgt_node or graph.has_edge(src_node, tgt_node): continue

                same_conf: float = self._pred_same_conf(src_img, tgt_img)
                reverse_same_conf: float = self._pred_same_conf(tgt_img, src_img)
                if same_conf < self._param.connect_conf_th or reverse_same_conf < self._param.connect_conf_th: continue
                if graph.has_edge(tgt_node, src_node):
                    if graph.edges[tgt_node, src_node]['required'] == True: continue
                    if graph.edges[tgt_node, src_node]['weight'] < 1-same_conf: continue
                    else: graph.remove_edge(tgt_node, src_node)

                graph.add_edge(src_node, tgt_node, weight=1-same_conf, required=False)

        rospy.loginfo(f"{len(list(graph.edges))} edges is added")

    def _add_edges_by_gt_pose(self, graph: nx.Graph) -> None:

        rospy.loginfo(f"start adding normal edge")
        for src_node, src_pose in dict(graph.nodes.data('pose')).items():
            for tgt_node, tgt_pose in dict(graph.nodes.data('pose')).items():
                if src_node == tgt_node or graph.has_edge(src_node, tgt_node): continue

                dist: float = math.hypot(tgt_pose[0] - src_pose[0], tgt_pose[1] - src_pose[1])
                if dist > self._param.connect_gt_dist_th: continue

                graph.add_edge(src_node, tgt_node, weight=dist, required=False)

        rospy.loginfo(f"{len(list(graph.edges))} edges is added")

    def _add_minimum_required_edges(self, graph: Union[nx.DiGraph, nx.Graph]) -> None:
        for src_node, src_img in dict(graph.nodes.data('img')).items():
            bag_idx, src_node_idx = src_node.split('_')
            next_node: str = "_".join([bag_idx, str(int(src_node_idx) + 1)])
            if not next_node in graph: continue
            if graph.has_edge(src_node, next_node): continue

            next_img: torch.Tensor = graph.nodes[next_node]['img']
            same_conf = self._pred_same_conf(src_img, next_img)
            graph.add_edge(src_node, next_node, weight=1-same_conf, required=True)

    def _edge_pruning(self, graph: Union[nx.DiGraph, nx.Graph]) -> None:
        
        rospy.loginfo(f"start pruning edge")
        pruned_count: int=0
        for src_node in graph.nodes:
            tgt_nodes = graph.succ[src_node]
            surplus_edge_count = len(tgt_nodes) - self._param.edge_num_th

            sorted_tgt_nodes = sorted(dict(tgt_nodes), key=lambda x: tgt_nodes[x]['weight'], reverse=True)
            for tgt_node in sorted_tgt_nodes:
                if surplus_edge_count < 1: break
                if graph.edges[src_node, tgt_node]['required'] == True: continue

                graph.remove_edge(src_node, tgt_node)
                rospy.loginfo(f"edge between {src_node} and {tgt_node} is removed")
                surplus_edge_count-=1
                pruned_count+=1
        rospy.loginfo(f"{pruned_count} edges are pruned")

    def _delete_node_without_cycle(self, graph: Union[nx.DiGraph, nx.Graph]) -> None:
        rospy.loginfo("deleting node without cycle")
        deletion_target_nodes: List[str] = []
        for node in graph.nodes:
            try:
                nx.find_cycle(graph, source=node, orientation="original")
            except:
                deletion_target_nodes.append(node)

        rospy.loginfo(f"deletion targets: {deletion_target_nodes}")
        graph.remove_nodes_from(deletion_target_nodes)
        rospy.loginfo(f"{len(deletion_target_nodes)} nodes are deleted.")

    def _create_graph_by_model(self) -> nx.DiGraph:
        graph: Optional[nx.DiGraph] = None
        # for i, bagfile_path in enumerate(iglob(os.path.join(self._param.bagfiles_dir, "*.bag"))):
        #     bag: Bag = Bag(bagfile_path)
        #     self._add_nodes_from_bag(graph, bag, i)
        #
        # self._add_minimum_required_edges(graph)
        # self._add_edges(graph)
        # self._edge_pruning(graph)
        # self._delete_node_without_cycle(graph)

        for i, bagfile_path in enumerate(iglob(os.path.join(self._param.bagfiles_dir, "*.bag"))):
            bag: Bag = Bag(bagfile_path)
            mono_graph = nx.DiGraph()
            self._add_nodes_from_bag(mono_graph, bag, i)
            self._add_minimum_required_edges(mono_graph)
            self._add_edges(mono_graph)
            self._delete_node_without_cycle(mono_graph)

            if graph == None: graph = mono_graph
            else: graph = nx.union(graph, mono_graph)
        self._add_edges(cast(nx.DiGraph, graph))
        self._edge_pruning(cast(nx.DiGraph, graph))

        return cast(nx.DiGraph, graph)

    def _create_graph_by_gt_pose(self) -> nx.DiGraph:
        # graph: Optional[nx.DiGraph] = None
        graph = nx.DiGraph()

        for i, bagfile_path in enumerate(iglob(os.path.join(self._param.bagfiles_dir, "*.bag"))):
            bag: Bag = Bag(bagfile_path)
            self._add_nodes_from_bag_by_gt_pose(graph, bag, i)

        self._add_edges_by_gt_pose(graph)

        return graph

    def process(self) -> None:
        
        self._graph = self._create_graph_by_model() if self._param.use_model else self._create_graph_by_gt_pose()

        rospy.loginfo(f"node num: {cast(nx.DiGraph, self._graph).number_of_nodes()}")
        rospy.loginfo(f"edge num: {cast(nx.DiGraph, self._graph).number_of_edges()}")

        save_topological_map(os.path.join(self._param.map_save_dir, self._param.map_name) + ".pkl", cast(nx.DiGraph, self._graph))

        if self._param.use_model:
            os.makedirs(os.path.join(self._param.map_save_dir, self._param.map_name+"_node_images"), exist_ok=True)
            save_nodes_as_img(self._graph, os.path.join(self._param.map_save_dir, self._param.map_name+"_node_images"))
        # print(f"edges: {dict(self._graph.edges)}")

        rospy.loginfo("process fnished")
        rosnode.kill_nodes("topological_mapper")
