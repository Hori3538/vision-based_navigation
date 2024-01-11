from torch.utils.data import Dataset, DataLoader
from glob import iglob
import os
from typing import List, Tuple
import torch
import random
import time

from training_data import TrainingData

class DatasetForOrientationNet(Dataset):
    def __init__(self, dataset_dirs: List[str]) -> None:
        data_path = []
        for dataset_dir in dataset_dirs:
            for data in iglob(os.path.join(dataset_dir, "*")):
                data_path.append(data)

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
    def equalize_label_counts(dataset, max_gap_times: int=1) -> None:
        orientation_label_counts = DatasetForOrientationNet.count_data_for_each_label(dataset); 
        surplus_label_coutns = orientation_label_counts - \
                orientation_label_counts[orientation_label_counts != 0].min()*max_gap_times

        while torch.any(surplus_label_coutns > 0):
            data_idx = random.randint(0, len(dataset)-1) 
            _, _, _, orientation_labels, _ = dataset[data_idx] 
            orientation_label = torch.where(orientation_labels == 1)
            if surplus_label_coutns[orientation_label] > 0:
                del dataset._data_path[data_idx]
                surplus_label_coutns[orientation_label] -= 1
    
    @staticmethod
    def count_data_for_each_label(dataset) -> torch.Tensor:
        start = time.time()
        label_num: int = len(dataset[0][3])

        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False,
                                # num_workers=os.cpu_count(), pin_memory=True)
                                num_workers=8, pin_memory=True)
        orientation_label_counts: torch.Tensor = torch.tensor([0] * label_num, dtype=torch.float)

        load_start = time.time()
        for batch in dataloader:
            load_end = time.time()
            print(f"load_time: {load_end-load_start:.4f}")
            data = TrainingData(*batch)
            orientation_label = data.orientation_label

            orientation_label_counts += torch.sum(orientation_label, 0)
            load_start = time.time()

        end = time.time()
        print(f"counting data time: {end-start}")

        return orientation_label_counts

def test() -> None:
    from argparse import ArgumentParser
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader

    parser = ArgumentParser()
    parser.add_argument("--dataset-dirs", type=str, nargs='*')
    args = parser.parse_args()

    dataset = DatasetForOrientationNet(args.dataset_dirs)
    # データ確認用loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    # DatasetForDirectionNet.equalize_label_counts(dataset)

    DatasetForOrientationNet.equalize_label_counts(dataset)
    data_len = dataset.__len__()
    print(f"data len: {data_len}")
    orientation_label_counts = DatasetForOrientationNet.count_data_for_each_label(dataset)
    print(f"orientation_label_counts: {orientation_label_counts}")

    print()

    for batch in dataloader:
        data = TrainingData(*batch)
        image_tensor = torch.cat((data.src_image[0], data.dst_image[0]), dim=2).squeeze()
        image = (image_tensor*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imshow("images", image)

        print(f"orientation_label: {data.orientation_label}")
        print(f"relative_pose: {data.relative_pose}")
        print()
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("c"):
            break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
