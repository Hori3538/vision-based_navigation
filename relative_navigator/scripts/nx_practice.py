import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.vitality import partial
import torch

G = nx.DiGraph()
print(type(G))

G.add_node(1, img=torch.Tensor([1, 2, 3]), gt_pose=[0.0, 0.0, 0.0])
G.add_node(2, img=torch.Tensor([1, 3, 3]), gt_pose=[0.0, 0.0, 1.0])
G.add_node(3, img=torch.Tensor([3, 1, 3]), gt_pose=[0.0, 0.0, 0.0])

G.add_edge(1, 3, bin=1, conf=0.7, weight=1)
G.add_edge(1, 2, bin=0, conf=0.7, weight=1)
G.add_edge(2, 3, bin=4, conf=0.7, weight=0)

print(dict(G.nodes))
print(list(G.nodes))
# print(nx.get_node_attributes(G, ))
print(dict(G.edges))
for edge in dict(G.nodes).items():
    print(edge)
# for edge in G.nodes:
#     print(edge)
# print(dict(G.edges[1]))
# print(type(G.nodes.data('img')))
for node_idx, img in dict(G.nodes.data('img')).items():
    print(node_idx)
    print(img)

print(nx.shortest_path(G, source=1, target=3, weight="weigth"))

nx.draw_networkx(G)
plt.show()
