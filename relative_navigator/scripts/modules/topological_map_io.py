import os
import pickle
# from dataclasses import dataclass

import networkx as nx

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
