#!/usr/bin/python3

from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, cast
import networkx as nx

import rospy
import torch
from visualization_msgs.msg import Marker
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage
from relative_navigator_msgs.msg import NodeInfo, NodeInfoArray

from .topological_map_io import load_topological_map
from .utils import tensor_to_compressed_image
from transformutils import get_msg_from_array_2d

@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int
    observed_image_width: int
    observed_image_height: int

    hz: float
    waypoint_num: int
    goal_th: int

    map_path: str

class GraphPathPlanner:
    def __init__(self) -> None:
        rospy.init_node("graph_path_planner")

        self._param: Param = Param(
                cast(int, rospy.get_param("/common/image_width")),
                cast(int, rospy.get_param("/common/image_height")),
                cast(int, rospy.get_param("/common/observed_image_width")),
                cast(int, rospy.get_param("/common/observed_image_height")),

                cast(float, rospy.get_param("~hz")),
                cast(int, rospy.get_param("~waypoint_num")),
                cast(int, rospy.get_param("~goal_th")),

                cast(str, rospy.get_param("~map_path")),
            )

        self._goal_node_id: Optional[str] = None
        self._nearest_node_id: Optional[str] = None

        self._goal_node_id_sub: rospy.Subscriber = rospy.Subscriber("/graph_localizer/goal_node_id",
                String, self._goal_node_callback, queue_size=3)
        self._nearest_node_id_sub: rospy.Subscriber = rospy.Subscriber("/graph_localizer/nearest_node_id",
                String, self._nearest_node_callback, queue_size=3)

        self._waypoints_pub = rospy.Publisher("~waypoints", NodeInfoArray, queue_size=3, tcp_nodelay=True)
        self._shortest_path_pub = rospy.Publisher("~shortest_path", Marker, queue_size=3, tcp_nodelay=True)
        self._reaching_goal_flag_pub = rospy.Publisher("~reaching_goal_flag", Bool, queue_size=3, tcp_nodelay=True)

        self._graph: Union[nx.DiGraph, nx.Graph] = load_topological_map(self._param.map_path)

    def _goal_node_callback(self, msg: String) -> None:
        self._goal_node_id = msg.data

    def _nearest_node_callback(self, msg: String) -> None:
        self._nearest_node_id = msg.data

    def _calc_shortest_path(self, start: str, goal: str) -> Optional[List[str]]:
        try:
            shortest_path: List[str] = cast(List[str],
                    nx.shortest_path(self._graph, source=start, target=goal,weight="weight"))
            return shortest_path
        except:
            rospy.logwarn(f"No path between {start} and {goal}")
            return None

    def _generate_marker_of_path(self, path_nodes: List[str]) -> Marker:

        marker = Marker()
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.pose.orientation.w = 1
        
        src_node: str = path_nodes.pop(0)
        for tgt_node in path_nodes:
            self._add_edge_to_marker(marker, (src_node, tgt_node))
            src_node = tgt_node

        return marker

    def _add_edge_to_marker(self, marker: Marker, edge: Tuple[str, str]) -> None:

        src_node, tgt_node = edge
        src_x, src_y, _ = self._graph.nodes[src_node]['pose']
        tgt_x, tgt_y, _ = self._graph.nodes[tgt_node]['pose']

        src_point = Point()
        src_point.x, src_point.y = src_x, src_y

        tgt_point = Point()
        tgt_point.x, tgt_point.y = tgt_x, tgt_y

        marker.points.append(src_point)
        marker.points.append(tgt_point)

    def _visualize_path(self, marker: Marker) -> None:

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "path"
        marker.id = 0
        self._shortest_path_pub.publish(marker)

    def _create_node_info_msg(self, node_name: str) -> NodeInfo:
        node_msg: NodeInfo = NodeInfo()
        node_msg.node_name = node_name

        node_img_tensor: torch.Tensor = self._graph.nodes[node_name]['img']
        node_img_msg: CompressedImage = tensor_to_compressed_image(
                node_img_tensor,(self._param.image_width, self._param.image_height))
        node_msg.image = node_img_msg

        node_msg.pose = get_msg_from_array_2d(self._graph.nodes[node_name]['pose'])
        
        return node_msg

    def _create_node_info_array_msg(self, node_names: List[str]) -> NodeInfoArray:
        node_array_msg: NodeInfoArray = NodeInfoArray()
        node_array_msg.header.frame_id = "map"
        node_array_msg.header.stamp = rospy.Time.now()

        for node_name in node_names:
            node_msg = self._create_node_info_msg(node_name)
            node_array_msg.node_infos.append(node_msg)

        return node_array_msg

    def process(self) -> None:
        rate = rospy.Rate(self._param.hz)
        while not rospy.is_shutdown():
            if self._goal_node_id is None or self._nearest_node_id is None: continue
            # if self._goal_node_id == self._nearest_node_id:
            #     rospy.loginfo(f"reaching goal")
            #     self._nearest_node_id = None
            #     continue

            shortest_path: Optional[List[str]] = self._calc_shortest_path(self._nearest_node_id, self._goal_node_id)
            if shortest_path is None:
                # self._nearest_node_id = None
                continue

            reaching_goal_flag = Bool()
            if len(shortest_path) <= self._param.goal_th:
                rospy.loginfo(f"reaching goal")
                reaching_goal_flag.data = True
                self._reaching_goal_flag_pub.publish(reaching_goal_flag)
                continue

            reaching_goal_flag.data = False
            self._reaching_goal_flag_pub.publish(reaching_goal_flag)

            shortest_path_marker: Marker = self._generate_marker_of_path(shortest_path)
            self._visualize_path(shortest_path_marker)

            waypoint_num: int = min(len(shortest_path), self._param.waypoint_num)
            node_array_msg: NodeInfoArray = self._create_node_info_array_msg(shortest_path[:waypoint_num])
            self._waypoints_pub.publish(node_array_msg)

            self._nearest_node_id = None

            rate.sleep()
