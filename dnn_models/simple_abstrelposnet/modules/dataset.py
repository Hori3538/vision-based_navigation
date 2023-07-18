from torch.utils.data import Dataset
from glob import iglob
import os
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import transforms

from training_data import TrainingData

class DatasetForSimpleAbstRelPosNet(Dataset):
    def __init__(self, dataset_dir: str, use_transform: bool = False) -> None:
        data_path = []
        for data in iglob(os.path.join(dataset_dir, "*", "*")):
            data_path.append(data)
        self._data_path = data_path
        self._use_transform = use_transform
        self._transform = nn.Sequential(
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # type: ignore
                transforms.RandomGrayscale(0.2),
                transforms.RandomApply([transforms.GaussianBlur(3)], 0.2),
                transforms.RandomErasing(0.2, scale=(0.05, 0.1), ratio=(0.33, 1.67)),
            )

    def __len__(self) -> int:
        return len(self._data_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data: TrainingData = torch.load(self._data_path[idx])
        src, dst, label = (
                    (data.src_image/255).permute(2, 0, 1),
                    (data.dst_image/255).permute(2, 0, 1),
                    data.label.clone().detach()
                )
        if self._use_transform:
            src, dst = self._transform(src), self._transform(dst)

        return src, dst, label

def test() -> None:
    from argparse import ArgumentParser
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=str)
    args = parser.parse_args()

    dataset = DatasetForSimpleAbstRelPosNet(args.dataset_dir, use_transform=False)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True, drop_last=False)
    dataloader2 = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    data_len = dataset.__len__()
    print(f"data len: {data_len}")
    label_counts = torch.tensor([0, 0, 0, 0])

    for batch in dataloader:
        data = TrainingData(*batch)
        label = data.label
        label_counts[0] += torch.sum(label == 0)
        label_counts[1] += torch.sum(label == 1)
        label_counts[2] += torch.sum(label == 2)
        label_counts[3] += torch.sum(label == 3)

    print(f"label ratio: {label_counts / sum(label_counts)}")

    for batch in dataloader2:
        data = TrainingData(*batch)
        image_tensor = torch.cat((data.src_image[0], data.dst_image[0]), dim=2).squeeze()
        image = (image_tensor*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imshow("images", image)
        print(f"label: {data.label[0]}")
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("c"):
            break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
