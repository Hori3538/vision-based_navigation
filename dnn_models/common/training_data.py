from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class TrainingData:
    src_image: torch.Tensor
    dst_image: torch.Tensor
    label: torch.Tensor
