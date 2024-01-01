from dataclasses import dataclass
from typing import Tuple, Union, cast

import networkx as nx
import rospy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


from .topological_map_io import load_topological_map

@dataclass(frozen=True)
class Param:
    hz: float
    map_path: str

class TopologicalMapVisualizer:
    def __init__(self) -> None:
        rospy.init_node("topological_map_visualizer")
        self._param = Param(
            cast(float, rospy.get_param("~hz")),
            cast(str, rospy.get_param("~map_path")),
        )
        self._graph: Union[nx.DiGraph, nx.Graph] = load_topological_map(self._param.map_path)
        self._nodes_pub = rospy.Publisher("~nodes", Marker, queue_size=1, tcp_nodelay=True)
        self._edges_pub = rospy.Publisher("~edges", Marker, queue_size=1, tcp_nodelay=True)

    def _generate_marker_of_nodes(self) -> Marker:

        marker = Marker()
        marker.type = marker.SPHERE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color.a = 1.0
        marker.color.b = 1.0

        marker.pose.orientation.w = 1
        
        for node in self._graph.nodes:
            self._add_node_to_marker(marker, node)

        return marker

    def _generate_marker_of_edges(self) -> Marker:

        marker = Marker()
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        # marker.scale.x = 0.2
        # marker.scale.y = 0.2
        # marker.scale.z = 0.2
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0

        marker.pose.orientation.w = 1
        
        for edge in self._graph.edges:
            self._add_edge_to_marker(marker, edge)

        return marker

    def _add_node_to_marker(self, marker: Marker, node_name: str) -> None:
        x, y, _ = self._graph.nodes[node_name]['pose']
        point = Point()
        point.x, point.y = x, y
        marker.colors.append(marker.color)
        marker.points.append(point)

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

    def _visualize_nodes(self, marker: Marker) -> None:

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "nodes"
        marker.id = 0
        self._nodes_pub.publish(marker)

    def _visualize_edges(self, marker: Marker) -> None:

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "edges"
        marker.id = 0
        self._edges_pub.publish(marker)

    def process(self) -> None:

        rate = rospy.Rate(self._param.hz)

        marker_of_nodes: Marker = self._generate_marker_of_nodes()
        marker_of_edges: Marker = self._generate_marker_of_edges()

        while not rospy.is_shutdown():
            self._visualize_nodes(marker_of_nodes)
            self._visualize_edges(marker_of_edges)
            rate.sleep()
