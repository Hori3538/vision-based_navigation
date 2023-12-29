from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from glob import iglob
import os
from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
import random
import time

from training_data import TrainingData

class DatasetForDirectionNet(Dataset):
    def __init__(self, dataset_dirs: List[str], use_transform: bool = False) -> None:
        data_path = []
        for dataset_dir in dataset_dirs:
            for data in iglob(os.path.join(dataset_dir, "*")):
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
            # ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def __len__(self) -> int:
        return len(self._data_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data: TrainingData = torch.load(self._data_path[idx])
        src, dst, direction_label, orientation_label, relative_pose = (
                    (data.src_image/255).permute(2, 0, 1),
                    (data.dst_image/255).permute(2, 0, 1),
                    data.direction_label.clone().detach(),
                    data.orientation_label.clone().detach(),
                    data.relative_pose
                )
        if self._use_transform:
            src, dst = self._transform(src), self._transform(dst)

        return src, dst, direction_label, orientation_label, relative_pose

    @staticmethod
    def equalize_label_counts(dataset) -> None:
        
        direction_label_counts = DatasetForDirectionNet.count_data_for_each_label(dataset); 
        surplus_label_coutns = direction_label_counts - \
                direction_label_counts[direction_label_counts != 0].min()
        print(f"surplus_label_coutns: {surplus_label_coutns}")

        while torch.any(surplus_label_coutns > 0):
            data_idx = random.randint(0, len(dataset)-1) 
            _, _, direction_labels, _, _ = dataset[data_idx] 
            direction_label = torch.where(direction_labels == 1)
            if surplus_label_coutns[direction_label] > 0:
                del dataset._data_path[data_idx]
                surplus_label_coutns[direction_label] -= 1
    
    @staticmethod
    # def count_data_for_each_label(dataset) -> Tuple[torch.Tensor, ...]:
    def count_data_for_each_label(dataset) -> torch.Tensor:
        start = time.time()
        label_num: int = len(dataset[0][2])

        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
        direction_label_counts: torch.Tensor = torch.tensor([0] * label_num, dtype=torch.float)

        for batch in dataloader:
            data = TrainingData(*batch)
            direction_label = data.direction_label

            direction_label_counts += torch.sum(direction_label, 0)

        # print(f"time: {time.time() - start}")
        return direction_label_counts

def test() -> None:
    from argparse import ArgumentParser
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--dataset-dirs", type=str, nargs='*')
    args = parser.parse_args()

    dataset = DatasetForDirectionNet(args.dataset_dirs, use_transform=False)
    # データ確認用loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    data_len: int = dataset.__len__()
    print(f"data len: {data_len}")
    direction_label_counts = DatasetForDirectionNet.count_data_for_each_label(dataset)
    print(f"direction_label_counts: {direction_label_counts}")

    DatasetForDirectionNet.equalize_label_counts(dataset)
    data_len = dataset.__len__()
    print(f"data len: {data_len}")
    direction_label_counts = DatasetForDirectionNet.count_data_for_each_label(dataset)
    print(f"direction_label_counts: {direction_label_counts}")

    # direction_label_counts, orientation_label_counts = DatasetForDirectionNet.count_data_for_each_label(dataset)

    direction_label_ratio: torch.Tensor = direction_label_counts / torch.sum(direction_label_counts)

    print(f"direction_label_ratio: {direction_label_ratio}")
    print()

    for batch in dataloader:
        data = TrainingData(*batch)
        image_tensor = torch.cat((data.src_image[0], data.dst_image[0]), dim=2).squeeze()
        image = (image_tensor*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imshow("images", image)
        print(f"direction_label: {data.direction_label}")
        # print(f"orientation_label: {data.orientation_label}")
        print(f"relative_pose: {data.relative_pose}")
        print()
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("c"):
            break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
