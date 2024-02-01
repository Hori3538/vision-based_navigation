from dataclasses import dataclass
import math
from glob import iglob
import os
from typing import Tuple, Union, Optional, List, cast

import networkx as nx
import rospy
from rosbag import Bag

from geometry_msgs.msg import Point, Pose, PoseStamped, PoseWithCovarianceStamped
from torch import dist
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from std_msgs.msg import String

from transformutils import get_array_2d_from_msg
from .topological_map_io import load_topological_map
from .utils import msg_to_pose

@dataclass(frozen=True)
class Param:
    hz: float
    
    bagfiles_dir: str
    map_path: str
    gt_map_path: str
    pose_topic_name: str
    pose_topic_type: str

class TopologicalMapTest:
    def __init__(self) -> None:
        rospy.init_node("topological_map_test")
        self._param = Param(
            cast(float, rospy.get_param("~hz")),

            cast(str, rospy.get_param("~bagfiles_dir")),
            cast(str, rospy.get_param("~map_path")),
            cast(str, rospy.get_param("~gt_map_path")),
            cast(str, rospy.get_param("~pose_topic_name")),
            cast(str, rospy.get_param("~pose_topic_type")),
        )
        self._graph: Union[nx.DiGraph, nx.Graph] = load_topological_map(self._param.map_path)
        self._gt_graph: Union[nx.DiGraph, nx.Graph] = load_topological_map(self._param.gt_map_path)
        self._nearest_node_id: Optional[str] = None
        self._gt_pose: Optional[PoseWithCovarianceStamped] = None

        self._nearest_node_id_sub: rospy.Subscriber = rospy.Subscriber("/graph_localizer/nearest_node_id",
                String, self._nearest_node_callback, queue_size=3)
        self._gt_pose_sub: rospy.Subscriber = rospy.Subscriber("/localized_pose",
                PoseWithCovarianceStamped, self._gt_pose_callback, queue_size=3)

        self._nodes_sphere_pub = rospy.Publisher("/topological_map_visualizer/nodes_sphere", Marker, queue_size=1, tcp_nodelay=True)
        self._nodes_text_pub = rospy.Publisher("/topological_map_visualizer/nodes_text", MarkerArray, queue_size=1, tcp_nodelay=True)
        self._edges_pub = rospy.Publisher("/topological_map_visualizer/edges", Marker, queue_size=1, tcp_nodelay=True)
        self._shortest_path_pub = rospy.Publisher("/graph_path_planner/shortest_path", Marker, queue_size=1, tcp_nodelay=True)

        self._nearest_node_marker_pub = rospy.Publisher("/graph_localizer/nearest_node_marker", Marker, queue_size=3, tcp_nodelay=True)
        self._goal_node_marker_pub = rospy.Publisher("/graph_localizer/goal_node_marker", Marker, queue_size=3, tcp_nodelay=True)
        self._gt_pose_trajs_pub = rospy.Publisher("~gt_pose_trajs", Path, queue_size=3, tcp_nodelay=True)

    def _nearest_node_callback(self, msg: String) -> None:
        self._nearest_node_id = msg.data

    def _gt_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._gt_pose = msg

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

    def _calc_shortest_path(self, graph: Union[nx.Graph, nx.DiGraph], start: str, goal: str) -> Optional[List[str]]:
        try:
            shortest_path: List[str] = cast(List[str],
                    nx.shortest_path(graph, source=start, target=goal,weight="weight"))
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

    def _calc_dist_of_path(self, graph: Union[nx.Graph, nx.DiGraph], path_nodes: List[str]) -> float:
        total_dist: float = -1
        before_node: str = path_nodes[0]
        for node in path_nodes:
            if total_dist == -1:
                total_dist = 0
                continue
            total_dist += self._calc_dist_between_nodes(graph, before_node, node)
            before_node = node
        
        return total_dist
    
    def _calc_dist_between_nodes(self, graph: Union[nx.Graph, nx.DiGraph], node1: str, node2: str) -> float:
        x1, y1, _ = graph.nodes[node1]["pose"]
        x2, y2, _ = graph.nodes[node2]["pose"]
        dist: float = math.hypot(x2-x1, y2-y1)

        return dist

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

    def _write_localize_error_from_bag(self, bag: Bag, bag_id: int, output_path: str) -> None:

        rospy.loginfo(f"start writing localize error of bag id: {bag_id}")
        rospy.loginfo(f"output path is {output_path}")
        gt_pose: Optional[Pose] = None

        for topic, msg, _ in bag.read_messages(
                topics=[self._param.pose_topic_name]):
            if topic == self._param.pose_topic_name:
                gt_pose = msg_to_pose(msg, self._param.pose_topic_type)
            if gt_pose is None: continue
            
                # pose_list = get_array_2d_from_msg(pose)
            pose_stamped = PoseStamped()
            pose_stamped.pose = gt_pose

            gt_pose = None
        rospy.loginfo(f"bag id: {bag_id} is finished")

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

    def _search_nearest_node(self, coord: Tuple[float, float], graph: Union[nx.DiGraph, nx.Graph]) -> str:
        src_x, src_y = coord
        min_dist = float('inf')
        nearest_node: str = list(graph.nodes)[0]

        for tgt_node, tgt_pose in dict(graph.nodes.data('pose')).items():
            tgt_x, tgt_y, _ = cast(List[float], tgt_pose)
            dist: float = math.hypot(tgt_x - src_x, tgt_y - src_y)
            if dist < min_dist:
                min_dist = dist
                nearest_node = tgt_node

        return nearest_node

    def _get_coord_of_node(self, graph: Union[nx.Graph, nx.DiGraph], node: str) -> Tuple[float, float]:
        return graph.nodes[node]["pose"][0], graph.nodes[node]["pose"][1]

    def _node_transform(self, src_node: str, src_graph: Union[nx.Graph, nx.DiGraph],
                                             tgt_graph: Union[nx.Graph, nx.DiGraph]) -> str:
        src_x, src_y = self._get_coord_of_node(src_graph, src_node)
        tgt_node: str = self._search_nearest_node((src_x, src_y), tgt_graph)

        return tgt_node

    def process(self) -> None:

        rospy.loginfo(f"node num: {self._graph.number_of_nodes()}")
        rospy.loginfo(f"edge num: {self._graph.number_of_edges()}")
        # print(f"edges: {dict(self._graph.edges)}")
        rate = rospy.Rate(self._param.hz)

        marker_of_nodes_sphere, marker_of_nodes_text = self._generate_marker_of_nodes()
        marker_of_edges: Marker = self._generate_marker_of_edges()
        # self._connect_check()

        start_node, goal_node = "1_128", "1_185" # 1
        # start_node, goal_node = "2_20", "2_75" # 2 old
        # start_node, goal_node = "3_140", "3_210" # 2_re
        # start_node, goal_node = "2_85", "2_170" # 3
        # start_node, goal_node = "0_145", "2_270" # 4
        # start_node, goal_node = "1_80", "3_90" # 5

        # new
        # start_node, goal_node = "1_110", "4_81" # 1
        # start_node, goal_node = "0_5", "0_42" # 2 
        # start_node, goal_node = "3_55", "0_10" # 3 old
        # start_node, goal_node = "3_18", "3_59" # 3 re
        # start_node, goal_node = "0_60", "3_146" # 4 old
        # start_node, goal_node = "1_120", "2_50" # 4 re
        # start_node, goal_node = "3_80", "2_150" # 5

        # straight
        # start_node, goal_node = "2_30", "2_70" # 3_re_re
        # start_node, goal_node = "0_110", "0_12" # 4_re_re
        # start_node, goal_node = "3_110", "2_150" # 5 re
        # start_node, goal_node = "2_90", "2_145" # 6

        gt_start_node = self._node_transform(start_node, self._graph, self._gt_graph)
        gt_goal_node = self._node_transform(goal_node, self._graph, self._gt_graph)

        start_node_marker: Marker = self._create_nearest_marker_node(start_node)
        goal_node_marker: Marker = self._create_goal_marker_node(goal_node)
        shortest_path: Optional[List[str]] = self._calc_shortest_path(self._graph, start_node, goal_node)
        gt_shortest_path: Optional[List[str]] = self._calc_shortest_path(self._gt_graph, gt_start_node, gt_goal_node)
        # shortest_path: Optional[List[str]] = ["2_" + str(i) for i in range(90, 146)]

        dist_of_path: float = self._calc_dist_of_path(self._graph, shortest_path)
        gt_dist_of_path: float = self._calc_dist_of_path(self._gt_graph, gt_shortest_path)

        print(f"dist of path: {dist_of_path}")
        shortest_path_marker: Marker = self._generate_marker_of_path(shortest_path)

        print(f"gt dist of path: {gt_dist_of_path}")

        gt_pose_trajs: List[Path] = self._calc_gt_pose_trajs_from_bag_dir(self._param.bagfiles_dir)

        while not rospy.is_shutdown():
            self._visualize_nodes_sphere(marker_of_nodes_sphere)
            self._visualize_edges(marker_of_edges)
            self._nodes_text_pub.publish(marker_of_nodes_text)

            self._nearest_node_marker_pub.publish(start_node_marker)
            self._goal_node_marker_pub.publish(goal_node_marker)

            self._publish_gt_pose_trajs(self._gt_pose_trajs_pub, gt_pose_trajs)



            self._visualize_path(shortest_path_marker)

            # if not self._nearest_node_id is None:
            #     rospy.loginfo(f"dist to goal: {self._calc_dist_between_nodes(self._graph, goal_node, self._nearest_node_id)}")
            if not self._gt_pose is None:
                goal_x, goal_y = self._get_coord_of_node(self._graph, goal_node)
                dist_to_goal = math.hypot(
                        goal_x - self._gt_pose.pose.pose.position.x,
                        goal_y - self._gt_pose.pose.pose.position.y,)
                rospy.loginfo(f"dist to goal: {dist_to_goal}")

            rate.sleep()
