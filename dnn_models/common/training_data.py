from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class TrainingData:
    src_image: torch.Tensor
    dst_image: torch.Tensor
    direction_label: torch.Tensor
    orientation_label: torch.Tensor
    # labels: torch.Tensor
    relative_pose: torch.Tensor
