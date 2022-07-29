from torch.utils.data import Dataset
from glob import iglob
import os
from typing import List
import torch
import torch.nn as nn

import sys
sys.path.append("../../common")
from training_data import TrainingData

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

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        data: TrainingData = torch.load(self._data_path[idx])
        src, dst, label = (
                    (data.src_image/255).permute(2, 0, 1),
                    (data.dst_image/255).permute(2, 0, 1),
                    data.label.clone().detach()
                )
        if self._use_transform:
            src, dst = self._transform(src), self._transform(dst)

        return [src, dst, label]

def test() -> None:
    from argparse import ArgumentParser
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=str)
    args = parser.parse_args()

    dataset = DatasetForAbstRelPosNet(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    for batch in dataloader:
        print(f"type of batch: {type(batch)}")
        data = TrainingData(*batch)
        print(f"type of data: {type(data)}")
        image_tensor = torch.cat((data.src_image, data.dst_image), dim=2).squeeze()
        image = (image_tensor*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        print(f"label: {data.label}")
        cv2.imshow("images", image)
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("c"):
            break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
