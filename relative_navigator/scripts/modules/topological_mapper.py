from dataclasses import dataclass
import re
from typing import Tuple, cast, List, Optional, Union
import os
import matplotlib.pyplot as plt
from glob import iglob

import networkx as nx
import rospy
import torch

from rosbag import Bag
import rosnode
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose

from transformutils import calc_relative_pose, get_array_2d_from_msg

from directionnet import DirectionNet
from orientationnet import OrientationNet

from .topological_map_io import save_topological_map, load_topological_map, save_nodes_as_img
from .utils import compressed_image_to_tensor, infer, msg_to_pose

@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int

    direction_net_weight_path: str
    orientation_net_weight_path: str
    bagfiles_dir: str
    map_save_dir: str
    map_name: str

    image_topic_name: str
    pose_topic_name: str
    pose_topic_type: str

    divide_conf_th: float # node生成時，隣接画像間の "same" confがこれより小さい時nodeを追加する
    connect_conf_th: float # edge 生成時，node間の "same" conf がこれより大きい時edgeを追加する
    edge_num_th: int # >= 1

    #  edgeの種類は dir or ori となり，ori edgeのほうがdir edge に比べ移動が容易であると考えられる．
    #  そのため，最短経路の計算時に使用するweight はori edgeのほうが小さく設定する
    orientation_edge_weigth_ratio: float

class TopologicalMapper:
    def __init__(self) -> None:
        rospy.init_node("topological_mapper")
        self._param = Param(
            cast(int, rospy.get_param("~image_width", 224)),
            cast(int, rospy.get_param("~image_height", 224)),
            cast(str, rospy.get_param("~direction_net_weight_path")),
            cast(str, rospy.get_param("~orientation_net_weight_path")),
            cast(str, rospy.get_param("~bagfiles_dir")),
            cast(str, rospy.get_param("~map_save_dir")),
            cast(str, rospy.get_param("~map_name")),
            cast(str, rospy.get_param("~image_topic_name", "/grasscam/image_raw/compressed")),
            cast(str, rospy.get_param("~pose_topic_name", "/whill/odom")),
            cast(str, rospy.get_param("~pose_topic_type", "Odometry")),

            cast(float, rospy.get_param("~divide_conf_th", 0.7)),
            cast(float, rospy.get_param("~connect_conf_th", 0.6)),
            cast(int, rospy.get_param("~edge_num_th", 3)),
            cast(float, rospy.get_param("~orientation_edge_weigth_ratio", 0.2)),
        )

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._direction_net: DirectionNet = DirectionNet().to(self._device)
        self._direction_net.load_state_dict(torch.load(self._param.direction_net_weight_path, map_location=torch.device(self._device)))
        self._direction_net.eval()

        self._orientation_net: OrientationNet = OrientationNet().to(self._device)
        self._orientation_net.load_state_dict(torch.load(self._param.orientation_net_weight_path, map_location=torch.device(self._device)))
        self._orientation_net.eval()

        self._graph = nx.DiGraph()
        # self._graph = nx.Graph()

    # def _are_diff_nodes(self, src_img: torch.Tensor, tgt_img: torch.Tensor) -> bool:
    #
    #     direction_probs: torch.Tensor = infer(self._direction_net, self._device,src_img, tgt_img).squeeze()
    #     orientation_probs: torch.Tensor = infer(self._orientation_net, self._device,src_img, tgt_img).squeeze()
    #
    #     direction_max_idx = direction_probs.max(0).indices
    #     orientation_max_idx = orientation_probs.max(0).indices
    #     
    #     # ラベルが変位ありの場合違うノードとする
    #     if direction_max_idx <= 2: return True
    #
    #     # 変位なしで方位が違う場合も違うノードとする
    #     if direction_max_idx == 3 and (orientation_max_idx == 0 or orientation_max_idx == 2):
    #         return True
    #
    #     return False

    def _are_different_enough(self, src_img: torch.Tensor, tgt_img: torch.Tensor) -> bool:

        direction_probs: torch.Tensor = infer(self._direction_net, self._device,src_img, tgt_img).squeeze()
        if direction_probs[3] < self._param.divide_conf_th: return True

        orientation_probs: torch.Tensor = infer(self._orientation_net, self._device,src_img, tgt_img).squeeze()
        if orientation_probs[1] < self._param.divide_conf_th: return True
        # orientation_max_idx = orientation_probs.max(0).indices
        # if orientation_max_idx == 0 or orientation_max_idx == 2: return True


        # direction_max_idx = direction_probs.max(0).indices
        return False

    def _add_nodes_from_bag(self, graph: Union[nx.DiGraph, nx.Graph], bag: Bag, bag_id: int) -> None:

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
                # if initial_pose is None: initial_pose = pose
                # pose = calc_relative_pose(initial_pose, pose)

            if img is None or pose is None: continue
            if prev_img is None:
                prev_img  = img
                continue
            
            # if self._are_diff_nodes(cast(torch.Tensor, prev_img), cast(torch.Tensor, img)):
            if self._are_different_enough(cast(torch.Tensor, prev_img), cast(torch.Tensor, img)):
                pose_list = get_array_2d_from_msg(pose)
                node_id = str(bag_id) + "_" + str(node_count)

                graph.add_node(node_id, img=img, pose=pose_list)

                node_count +=1
                prev_img = img

            img = None
            pose = None
        rospy.loginfo(f"bag id: {bag_id} is finished\n {node_count} nodes is added.")

    # pred label and conf
    # def _pred_edge(self, src_img: torch.Tensor, tgt_img: torch.Tensor) -> Optional[Tuple[str, float]]:
    #
    #     # 計算量を抑えるためdirectionの予測だけで確定する場合を先に処理する
    #     direction_probs: torch.Tensor = infer(self._direction_net, self._device,src_img, tgt_img).squeeze()
    #     direction_max_idx = int(direction_probs.max(0).indices)
    #     direction_label_conf = float(direction_probs[direction_max_idx])
    #
    #     if direction_max_idx == 4: return None # negativeラベルの時はエッジ作らない
    #     if direction_max_idx < 3: return "dir"+str(direction_max_idx), direction_label_conf
    #
    #     # direction label がsame(3)の時の処理
    #     orientation_probs: torch.Tensor = infer(self._orientation_net, self._device,src_img, tgt_img).squeeze()
    #     orientation_max_idx = int(orientation_probs.max(0).indices)
    #     orientation_label_conf = float(orientation_probs[orientation_max_idx])
    #
    #     if orientation_max_idx == 1: return None # orientationに変化がない時はエッジ作らない
    #     return "ori"+str(orientation_max_idx), orientation_label_conf

    # pred conf of "same" label
    # def _pred_same_conf(self, src_img: torch.Tensor, tgt_img: torch.Tensor) -> Optional[float]:
    def _pred_same_conf(self, src_img: torch.Tensor, tgt_img: torch.Tensor) -> float:

        direction_probs: torch.Tensor = infer(self._direction_net, self._device,src_img, tgt_img).squeeze()
        direction_max_idx = int(direction_probs.max(0).indices)

        # if direction_max_idx != 3: return None
        return float(direction_probs[3])

    # return node_name and conf_of_label
    # def _find_node_connected_by_designated_label(self, src_node, label: str) -> Optional[Tuple[str, float]]:
    #     for tgt_node, attribute in self._graph.succ[src_node].items():
    #         if attribute['label'] == label:
    #             return tgt_node, attribute['conf']
    #
    #     return None

    # def _add_edges(self, graph: nx.DiGraph) -> None:
    #
    #     for src_node, src_img in dict(graph.nodes.data('img')).items():
    #         for tgt_node, tgt_img in dict(graph.nodes.data('img')).items():
    #             edge_info: Optional[Tuple[str, float]] = self._pred_edge(src_img, tgt_img)
    #             if edge_info == None: continue
    #
    #             label, conf = edge_info
    #             if conf < self._param.label_conf_th: continue
    #             edge_weigth = self._param.orientation_edge_weigth_ratio if "ori" in label else 1
    #
    #             competing_node_info: Optional[Tuple[str, float]] = \
    #                 self._find_node_connected_by_designated_label(src_node, label)
    #
    #             if competing_node_info == None:
    #                 graph.add_edge(src_node, tgt_node,
    #                                label=label, conf=conf, weight=edge_weigth, required=False)
    #                 continue
    #
    #             competing_node, competing_edge_conf = competing_node_info
    #             if conf > competing_edge_conf:
    #                 graph.add_edge(src_node, tgt_node, label=label, conf=conf, weight=edge_weigth,
    #                                required=False)
    #                 graph.remove_edge(src_node, competing_node)

    def _add_edges2(self, graph: Union[nx.DiGraph, nx.Graph]) -> None:

        for src_node, src_img in dict(graph.nodes.data('img')).items():
            for tgt_node, tgt_img in dict(graph.nodes.data('img')).items():
                if src_node == tgt_node or graph.has_edge(src_node, tgt_node): continue

                # same_conf: Optional[float] = self._pred_same_conf(src_img, tgt_img)
                same_conf: float = self._pred_same_conf(src_img, tgt_img)
                reverse_same_conf: float = self._pred_same_conf(tgt_img, src_img)
                # if same_conf == None or same_conf < self._param.connect_conf_th: continue
                if same_conf < self._param.connect_conf_th or reverse_same_conf < self._param.connect_conf_th: continue

                graph.add_edge(src_node, tgt_node, weight=1-same_conf, required=False)
        rospy.loginfo(f"{len(list(self._graph.edges))} edges is added")

    def _add_minimum_required_edges(self, graph: Union[nx.DiGraph, nx.Graph]) -> None:
        for src_node, src_img in dict(graph.nodes.data('img')).items():
            bag_idx, src_node_idx = src_node.split('_')
            next_node: str = "_".join([bag_idx, str(int(src_node_idx) + 1)])
            if not next_node in graph: continue

            next_img: torch.Tensor = graph.nodes[next_node]['img']
            same_conf = self._pred_same_conf(src_img, next_img)
            graph.add_edge(src_node, next_node, weight=1-same_conf, required=True)

    def _edge_pruning(self, graph: Union[nx.DiGraph, nx.Graph]) -> None:
        for src_node in graph.nodes:
            tgt_nodes = graph.succ[src_node]
            surplus_edge_count = len(tgt_nodes) - self._param.edge_num_th

            sorted_tgt_nodes = sorted(dict(tgt_nodes), key=lambda x: tgt_nodes[x]['weight'], reverse=True)
            for tgt_node in sorted_tgt_nodes:
                if surplus_edge_count < 1: break
                if graph.edges[src_node, tgt_node]['required'] == True: continue

                graph.remove_edge(src_node, tgt_node)
                surplus_edge_count -= 1


    def process(self) -> None:

        # add nodes process
        # for i, bagfile_path in enumerate(iglob(os.path.join(self._param.bagfiles_dir, "*"))):
        #     bag: Bag = Bag(bagfile_path)
        #     self._add_nodes_from_bag(self._graph, bag, i)

        self._graph = load_topological_map(os.path.join(self._param.map_save_dir, self._param.map_name))
        # add edges process
        # self._add_edges(self._graph)
        
        self._graph.remove_edges_from(list(self._graph.edges))
        self._add_minimum_required_edges(self._graph)
        self._add_edges2(self._graph)
        self._edge_pruning(self._graph)
        save_topological_map(os.path.join(self._param.map_save_dir, self._param.map_name), self._graph)
        # self._graph = load_topological_map(os.path.join(self._param.map_save_dir, self._param.map_name))
        os.makedirs(os.path.join(self._param.map_save_dir, "node_images"), exist_ok=True)
        # save_nodes_as_img(self._graph, os.path.join(self._param.map_save_dir, "node_images"))

        print(dict(self._graph.edges))
        # print(self._pred_same_conf(self._graph.nodes["0_250"]['img'],
        #                            self._graph.nodes["0_21"]['img']))
        # print(infer(self._direction_net, self._device, self._graph.nodes["0_382"]['img'], 
        #     self._graph.nodes["0_100"]['img']))
        # print(nx.shortest_path(self._graph, source="0_0", target="0_200", weight="weigth"))
        
        # nx.draw_networkx(self._graph)
        plt.show()
        rospy.loginfo("Process fnished")
        rosnode.kill_nodes("topological_mapper")
