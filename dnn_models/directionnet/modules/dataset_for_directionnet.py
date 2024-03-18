from torch.utils.data import Dataset, DataLoader
from glob import iglob
import os
from typing import List, Tuple
import torch
import random
import time
import copy
from dnn_utils import image_tensor_cat_and_show
import dnn_utils

from training_data import TrainingData

class DatasetForDirectionNet(Dataset):
    def __init__(self, dataset_dirs: List[str]) -> None:
        data_path = []
        for dataset_dir in dataset_dirs:
            for data in iglob(os.path.join(dataset_dir, "*")):
                data_path.append(data)

        random.shuffle(data_path)
        self._data_path = data_path

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

        return src, dst, direction_label, orientation_label, relative_pose

    @staticmethod
    # def equalize_label_counts(dataset, max_gap_times: int=1) -> None:
    def equalize_label_counts(dataset, max_gap_times: int=1) -> torch.Tensor:
        
        label_counts = DatasetForDirectionNet.count_data_for_each_label(dataset); 
        surplus_label_coutns = label_counts - label_counts[label_counts!=0].min() * max_gap_times

        start = time.time()
        data_idx = 0
        while torch.any(surplus_label_coutns > 0):
            # data_idx = random.randint(0, len(dataset)-1) 
            _, _, labels, _, _ = dataset[data_idx] 
            label = torch.where(labels == 1)
            if surplus_label_coutns[label] > 0:
                del dataset._data_path[data_idx]
                surplus_label_coutns[label] -= 1
                label_counts[label] -= 1
            else: data_idx += 1
        end = time.time()
        print(f"equalize data time: {end-start}")

        return label_counts
    
    @staticmethod
    def count_data_for_each_label(dataset) -> torch.Tensor:
        start = time.time()
        label_num: int = len(dataset[0][2])

        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False,
                                num_workers=8, pin_memory=True)
        direction_label_counts: torch.Tensor = torch.tensor([0] * label_num, dtype=torch.float)

        for batch in dataloader:
            data = TrainingData(*batch)
            direction_label = data.direction_label

            direction_label_counts += torch.sum(direction_label, 0)
        end = time.time()
        print(f"counting data time: {end-start}")

        return direction_label_counts

def test() -> None:
    from argparse import ArgumentParser
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--dataset-dirs", type=str, nargs='*')
    args = parser.parse_args()

    dataset = DatasetForDirectionNet(args.dataset_dirs)
    # データ確認用loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True,
                            num_workers=os.cpu_count(), pin_memory=True)

    transform = dnn_utils.test_transform
    DatasetForDirectionNet.equalize_label_counts(dataset)
    data_len = dataset.__len__()
    print(f"data len: {data_len}")
    direction_label_counts = DatasetForDirectionNet.count_data_for_each_label(dataset)
    print(f"direction_label_counts: {direction_label_counts}")

    direction_label_ratio: torch.Tensor = direction_label_counts / torch.sum(direction_label_counts)

    print(f"direction_label_ratio: {direction_label_ratio}")
    print()

    for batch in dataloader:
        data = TrainingData(*batch)
        image_tensor = torch.cat((data.src_image[0], data.dst_image[0]), dim=2).squeeze()
        # image_tensor = torch.cat((transform(data.src_image[0]), transform(data.dst_image[0])), dim=2).squeeze()
        image = (image_tensor*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # cv2.imshow("images", image)
        print(f"direction_label: {data.direction_label}")
        print(f"relative_pose: {data.relative_pose}")
        print()
        # key = cv2.waitKey(0)
        # if key == ord("q") or key == ord("c"):
        #     break
        # cv2.destroyAllWindows()
        image_tensor_cat_and_show(data.src_image[0],transform(data.src_image[0]), "/home/amsl/Pictures")

if __name__ == "__main__":
    test()
