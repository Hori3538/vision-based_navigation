from torch.utils.data import Dataset
from glob import iglob
import os
from typing import List, Tuple
import torch
import torch.nn as nn

import sys
sys.path.append("../../common")
sys.path.append("../common")
from training_data import TrainingData
# sys.path.append("../..")
# from common.training_data import TrainingData

class DatasetForAbstRelPosNet(Dataset):
    def __init__(self, dataset_dir: str, use_transform: bool = True) -> None:
        data_path = []
        for data in iglob(os.path.join(dataset_dir, "*", "*")):
            data_path.append(data)
        self._data_path = data_path
        self._use_transform = use_transform
        self._transform = nn.Sequential(

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

    dataset = DatasetForAbstRelPosNet(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True, drop_last=False)
    dataloader2 = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    data_len = dataset.__len__()
    print(f"data len: {data_len}")
    label_count_minus = torch.tensor([0] * 3)
    label_count_same = torch.tensor([0] * 3)
    label_count_plus = torch.tensor([0] * 3)

    for batch in dataloader:
        data = TrainingData(*batch)
        label = data.label

        label_count_minus += torch.sum(label == -1, 0)[:3]
        label_count_same += torch.sum(label == 0, 0)[:3]
        label_count_plus += torch.sum(label == 1, 0)[:3]

    print(f"x_label_rate -1: {label_count_minus[0] / data_len * 100:.3f}, 0: {label_count_same[0] / data_len * 100:.3f}, 1: {label_count_plus[0] / data_len * 100:.3f}")
    print(f"y_label_rate -1: {label_count_minus[1] / data_len * 100:.3f}, 0: {label_count_same[1] / data_len * 100:.3f}, 1: {label_count_plus[1] / data_len * 100:.3f}")
    print(f"yaw_label_rate -1: {label_count_minus[2] / data_len * 100:.3f}, 0: {label_count_same[2] / data_len * 100:.3f}, 1: {label_count_plus[2] / data_len * 100:.3f}")

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
