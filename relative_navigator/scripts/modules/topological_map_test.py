from dataclasses import dataclass
import math
from glob import iglob
import os
from typing import Tuple, Union, Optional, List, cast

import networkx as nx
import rospy
from rosbag import Bag

from geometry_msgs.msg import Point, Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path

from transformutils import get_array_2d_from_msg
from .topological_map_io import load_topological_map
from .utils import msg_to_pose

@dataclass(frozen=True)
class Param:
    hz: float
    
    bagfiles_dir: str
    map_path: str
    pose_topic_name: str
    pose_topic_type: str

class TopologicalMapTest:
    def __init__(self) -> None:
        rospy.init_node("topological_map_test")
        self._param = Param(
            cast(float, rospy.get_param("~hz")),

            cast(str, rospy.get_param("~bagfiles_dir")),
            cast(str, rospy.get_param("~map_path")),
            cast(str, rospy.get_param("~pose_topic_name")),
            cast(str, rospy.get_param("~pose_topic_type")),
        )
        self._graph: Union[nx.DiGraph, nx.Graph] = load_topological_map(self._param.map_path)
        self._nodes_sphere_pub = rospy.Publisher("/topological_map_visualizer/nodes_sphere", Marker, queue_size=1, tcp_nodelay=True)
        self._nodes_text_pub = rospy.Publisher("/topological_map_visualizer/nodes_text", MarkerArray, queue_size=1, tcp_nodelay=True)
        self._edges_pub = rospy.Publisher("/topological_map_visualizer/edges", Marker, queue_size=1, tcp_nodelay=True)
        self._shortest_path_pub = rospy.Publisher("/graph_path_planner/shortest_path", Marker, queue_size=1, tcp_nodelay=True)

        self._nearest_node_marker_pub = rospy.Publisher("/graph_localizer/nearest_node_marker", Marker, queue_size=3, tcp_nodelay=True)
        self._goal_node_marker_pub = rospy.Publisher("/graph_localizer/goal_node_marker", Marker, queue_size=3, tcp_nodelay=True)
        self._gt_pose_trajs_pub = rospy.Publisher("~gt_pose_trajs", Path, queue_size=3, tcp_nodelay=True)

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

    def _calc_dist_of_path(self, path_nodes: List[str]) -> float:
        total_dist: float = -1
        before_x, before_y, _ = self._graph.nodes[path_nodes[0]]["pose"]
        for node in path_nodes:
            if total_dist == -1:
                total_dist = 0
                continue
            x, y, _ = self._graph.nodes[node]["pose"]
            total_dist += math.hypot(x-before_x, y-before_y)
            before_x, before_y = x, y
        
        return total_dist

    def _connect_check(self) -> None:
        for start in self._graph.nodes:
            for goal in self._graph.nodes:
                try:
                    cast(List[str], nx.shortest_path(self._graph, source=start, target=goal,weight="weight"))
                except:
                    rospy.logwarn(f"No path between {start} and {goal}")

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

    def _calc_gt_pose_traj_from_bag(self, bag: Bag, bag_id: int) -> Path:

        rospy.loginfo(f"start calculating gt pose trajectory of bag id: {bag_id}")
        pose: Optional[Pose] = None
        traj = Path()
        traj.header.stamp = rospy.Time.now()
        traj.header.frame_id = "map"

        for topic, msg, _ in bag.read_messages(
                topics=[self._param.pose_topic_name]):
            if topic == self._param.pose_topic_name:
                pose = msg_to_pose(msg, self._param.pose_topic_type)
            if pose is None: continue
            
                # pose_list = get_array_2d_from_msg(pose)
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            traj.poses.append(pose_stamped)

            pose = None
        rospy.loginfo(f"bag id: {bag_id} is finished")
        return traj

    def _calc_gt_pose_trajs_from_bag_dir(self, bag_dir: str) -> List[Path]:
        trajs: List[Path] = []
        for i, bagfile_path in enumerate(iglob(os.path.join(bag_dir, "*.bag"))):
            rospy.loginfo(f"bagfile_path is {bagfile_path}")
            bag: Bag = Bag(bagfile_path)
            traj = self._calc_gt_pose_traj_from_bag(bag, i)
            trajs.append(traj)

        return trajs

    def _publish_gt_pose_trajs(self, publisher: rospy.Publisher, gt_pose_trajs: List[Path]) -> None:
        for traj in gt_pose_trajs: publisher.publish(traj)

    # def _search_nearest_node(self, pose_list: List[float], graph: Union[nx.DiGraph, nx.Graph]) -> str:



    def process(self) -> None:

        rospy.loginfo(f"node num: {self._graph.number_of_nodes()}")
        rospy.loginfo(f"edge num: {self._graph.number_of_edges()}")
        # print(f"edges: {dict(self._graph.edges)}")
        rate = rospy.Rate(self._param.hz)

        marker_of_nodes_sphere, marker_of_nodes_text = self._generate_marker_of_nodes()
        marker_of_edges: Marker = self._generate_marker_of_edges()
        # self._connect_check()

        start_node, goal_node = "1_128", "1_185" # 1
        start_node_marker: Marker = self._create_nearest_marker_node(start_node)
        goal_node_marker: Marker = self._create_goal_marker_node(goal_node)
        shortest_path: Optional[List[str]] = self._calc_shortest_path(start_node, goal_node)

        dist_of_path: float = self._calc_dist_of_path(shortest_path)
        print(f"dist of path: {dist_of_path}")
        shortest_path_marker: Marker = self._generate_marker_of_path(shortest_path)

        gt_pose_trajs: List[Path] = self._calc_gt_pose_trajs_from_bag_dir(self._param.bagfiles_dir)

        while not rospy.is_shutdown():
            self._visualize_nodes_sphere(marker_of_nodes_sphere)
            self._visualize_edges(marker_of_edges)
            self._nodes_text_pub.publish(marker_of_nodes_text)

            self._nearest_node_marker_pub.publish(start_node_marker)
            self._goal_node_marker_pub.publish(goal_node_marker)

            self._publish_gt_pose_trajs(self._gt_pose_trajs_pub, gt_pose_trajs)

            # shortest_path: Optional[List[str]] = self._calc_shortest_path("2_20", "2_75")#old
            # shortest_path: Optional[List[str]] = self._calc_shortest_path("3_140", "3_210")# 2
            # shortest_path: Optional[List[str]] = self._calc_shortest_path("2_85", "2_170")# 3
            # shortest_path: Optional[List[str]] = self._calc_shortest_path("0_145", "2_270")# 4
            # shortest_path: Optional[List[str]] = self._calc_shortest_path("1_80", "3_90")# 5

            # shortest_path: Optional[List[str]] = self._calc_shortest_path("1_110", "4_81")# 1
            # shortest_path: Optional[List[str]] = self._calc_shortest_path("0_5", "0_42")# 2
            # shortest_path: Optional[List[str]] = self._calc_shortest_path("3_55", "0_10")# 3
            # shortest_path: Optional[List[str]] = self._calc_shortest_path("0_60", "3_146")# 4
            # shortest_path: Optional[List[str]] = self._calc_shortest_path("3_80", "2_150")# 5
            # shortest_path: Optional[List[str]] = ["3_"+str(i) for i in range(80, 136)]
            # shortest_path: Optional[List[str]] = ["0_"+str(i) for i in range(60, 121)]
            # shortest_path: Optional[List[str]] = ["3_"+str(i) for i in range(55, 120)]

            self._visualize_path(shortest_path_marker)

            rate.sleep()
