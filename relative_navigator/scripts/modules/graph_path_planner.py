#!/usr/bin/python3

from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, cast
import networkx as nx

import rospy
import torch
from visualization_msgs.msg import Marker
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage

from .topological_map_io import load_topological_map
from .utils import tensor_to_compressed_image

@dataclass(frozen=True)
class Param:
    image_width: int
    image_height: int
    observed_image_width: int
    observed_image_height: int

    hz: float
    first_waypoint_dist: int

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
                cast(int, rospy.get_param("~first_waypoint_dist")),

                cast(str, rospy.get_param("~map_path")),
            )

        self._goal_node_id: Optional[str] = None
        self._nearest_node_id: Optional[str] = None

        self._goal_node_id_sub: rospy.Subscriber = rospy.Subscriber("/graph_localizer/goal_node_id",
                String, self._goal_node_callback, queue_size=1)
        self._nearest_node_id_sub: rospy.Subscriber = rospy.Subscriber("/graph_localizer/nearest_node_id",
                String, self._nearest_node_callback, queue_size=1)

        self._first_waypoint_img_pub = rospy.Publisher("~first_waypoint_img/image_raw/compressed",
                CompressedImage, queue_size=1, tcp_nodelay=True)
        self._first_waypoint_id_pub = rospy.Publisher("~first_waypoint_id", String, queue_size=1, tcp_nodelay=True)
        self._shortest_path_pub = rospy.Publisher("~shortest_path", Marker, queue_size=1, tcp_nodelay=True)

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

    def process(self) -> None:
        rate = rospy.Rate(self._param.hz)
        while not rospy.is_shutdown():
            if self._goal_node_id is None or self._nearest_node_id is None: continue
            if self._goal_node_id == self._nearest_node_id:
                rospy.loginfo(f"reaching goal")
                self._nearest_node_id = None
                continue

            shortest_path: Optional[List[str]] = self._calc_shortest_path(self._nearest_node_id, self._goal_node_id)
            if shortest_path is None:
                self._nearest_node_id = None
                continue

            shortest_path_marker: Marker = self._generate_marker_of_path(shortest_path)
            self._visualize_path(shortest_path_marker)

            first_waypoint_dist: int = min(len(shortest_path)-1, self._param.first_waypoint_dist)
            first_waypoint_id: str = shortest_path[first_waypoint_dist]
            first_waypoint_img_tensor: torch.Tensor = self._graph.nodes[first_waypoint_id]['img']
            first_waypoint_img_msg: CompressedImage = tensor_to_compressed_image(
                    first_waypoint_img_tensor,
                    (self._param.observed_image_width, self._param.observed_image_height)
                    )

            self._first_waypoint_id_pub.publish(first_waypoint_id)
            self._first_waypoint_img_pub.publish(first_waypoint_img_msg)

            self._nearest_node_id = None

            rate.sleep()
