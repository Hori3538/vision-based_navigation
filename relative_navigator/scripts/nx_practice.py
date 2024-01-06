import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.vitality import partial
import torch

G = nx.DiGraph()

G.add_node(1, img=torch.Tensor([1, 2, 3]), gt_pose=[0.0, 0.0, 0.0])
# G.add_node(1, img=torch.Tensor([2, 2, 3]), gt_pose=[0.0, 0.0, 0.0])
G.add_node(2, img=torch.Tensor([1, 3, 3]), gt_pose=[0.0, 0.0, 1.0])
G.add_node(3, img=torch.Tensor([3, 1, 3]), gt_pose=[0.0, 0.0, 0.0])
G.add_node(4, img=torch.Tensor([3, 1, 3]), gt_pose=[0.0, 0.0, 0.0])

G.add_edge(1, 3, bin=1, conf=0.7, weight=1)
G.add_edge(1, 2, bin=0, conf=0.6, weight=1)
G.add_edge(1, 4, bin=0, conf=0.8, weight=1)
#
G.add_edge(2, 3, bin=4, conf=0.7, weight=0)


# print(list(nx.neighbors(G, 1)))
# print(nx.get_node_attributes(G, 'img'))
# print(G.nodes[1]['img'])
# print((G.out_edges(1)))
# for edge in dict(G.out_edges(1)):
# for edge in G.out_edges(1):
# print(G.succ[1])
# print(G.edges[1, 2]['bin'])
# print(dict(G.succ[1]))
# print(sorted(dict(G.succ[1]), key=lambda x: G.succ[1][x]['conf']))
# print(dict(sorted(dict(G.succ[1]), key=lambda x: G.succ[1][x]['conf'])))
# print((G.out_edges(1)))

print(type(nx.shortest_path(G, source=1, target=3, weight="weigth")))
def add_node(graph):
    graph.add_node(4, img=torch.Tensor([3, 1, 3]), gt_pose=[0.0, 0.0, 0.0])

# print(nx.get_node_attributes(G, ))
for edge in dict(G.nodes).items():
    print(edge)
    # print(edge['img'])
for node_idx, img in dict(G.nodes.data('img')).items():
    print(node_idx)
    print(img)


# add_node(G)
nx.draw_networkx(G)
plt.show()
