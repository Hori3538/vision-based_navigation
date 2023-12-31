import os
import pickle
import numpy as np
import cv2
# from dataclasses import dataclass

import networkx as nx

from .utils import tensor_to_cv_image

# @dataclass(frozen=True)
# class TopologicalMapIO:
#     @staticmethod
#     def save(path: str, topological_map: nx.DiGraph) -> None:
#
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         with open(path, "wb") as f:
#             pickle.dump(topological_map, f)
#
#     @staticmethod
#     def load(path: str) -> nx.DiGraph:
#
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"File {path} does not exist.")
#         with open(path, "rb") as f:
#             return pickle.load(f)
def save_topological_map(path: str, topological_map: nx.DiGraph) -> None:

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(topological_map, f)

def load_topological_map(path: str) -> nx.DiGraph:

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    with open(path, "rb") as f:
        return pickle.load(f)

def save_nodes_as_img(graph: nx.DiGraph, dir: str):
    for node_id, img_tensor in dict(graph.nodes.data('img')).items():
        img_np: np.ndarray = tensor_to_cv_image(img_tensor)
        image_name = node_id + ".jpg"
        image_dir = os.path.join(dir, image_name)
        cv2.imwrite(image_dir, img_np)
        # print(f"bool {cv2.imwrite(image_dir, img_np)}")
