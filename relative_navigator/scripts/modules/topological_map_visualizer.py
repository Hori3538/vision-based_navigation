from dataclasses import dataclass
from typing import Tuple, Union, cast

import networkx as nx
import rospy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

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
        self._nodes_sphere_pub = rospy.Publisher("~nodes_sphere", Marker, queue_size=1, tcp_nodelay=True)
        self._nodes_text_pub = rospy.Publisher("~nodes_text", MarkerArray, queue_size=1, tcp_nodelay=True)
        self._edges_pub = rospy.Publisher("~edges", Marker, queue_size=1, tcp_nodelay=True)

    def _generate_marker_of_nodes(self) -> Tuple[Marker, MarkerArray]:

        marker_sphere = Marker()
        marker_sphere.type = marker_sphere.SPHERE_LIST
        marker_sphere.action = marker_sphere.ADD
        marker_sphere.scale.x = 0.4
        marker_sphere.scale.y = 0.4
        marker_sphere.scale.z = 0.4
        marker_sphere.color.a = 1.0
        marker_sphere.color.b = 1.0

        marker_sphere.pose.orientation.w = 1

        markers_text = MarkerArray()
        
        for marker_id, node in enumerate(self._graph.nodes):
            self._add_node_to_marker(marker_sphere, node)
            markers_text.markers.append(self._create_text_marker_of_node(node, marker_id))

        return marker_sphere, markers_text

    def _generate_marker_of_edges(self) -> Marker:

        marker = Marker()
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        # marker.scale.x = 0.2
        # marker.scale.y = 0.2
        # marker.scale.z = 0.2
        marker.scale.x = 0.07
        marker.scale.y = 0.07
        marker.scale.z = 0.07
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

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
    
    def _create_text_marker_of_node(self, node_name: str, marker_id: int) -> Marker:
        marker = Marker()
        marker.type = marker.TEXT_VIEW_FACING
        marker.action = marker.ADD

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "nodes_text"

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        marker.scale.z = 0.7

        marker.id = marker_id

        x, y, _ = self._graph.nodes[node_name]['pose']
        marker.pose.position.x, marker.pose.position.y = x, y
        marker.pose.orientation.w = 1
        marker.text = node_name
        
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

    def _visualize_nodes_sphere(self, marker: Marker) -> None:

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "nodes_sphere"
        marker.id = 0
        self._nodes_sphere_pub.publish(marker)

    def _visualize_edges(self, marker: Marker) -> None:

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "edges"
        marker.id = 0
        self._edges_pub.publish(marker)

    def process(self) -> None:

        rate = rospy.Rate(self._param.hz)

        marker_of_nodes_sphere, marker_of_nodes_text = self._generate_marker_of_nodes()
        marker_of_edges: Marker = self._generate_marker_of_edges()

        while not rospy.is_shutdown():
            self._visualize_nodes_sphere(marker_of_nodes_sphere)
            self._visualize_edges(marker_of_edges)
            self._nodes_text_pub.publish(marker_of_nodes_text)
            rate.sleep()
