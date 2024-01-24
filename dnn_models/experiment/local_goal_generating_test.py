import argparse
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from my_models import DirectionNet, OrientationNet, RelPosNet
from dataset_for_directionnet import DatasetForDirectionNet
from dnn_utils import fix_seed, image_tensor_cat_and_show

def calc_weighted_mean_angle(angle_for_each_label: List[float], orientation_net_output: torch.Tensor) -> float:
    orientation_net_probs: List[float] = F.softmax(orientation_net_output, dim=0).tolist()
    # print(f"orientation_net_probs: {orientation_net_probs}")
    weighted_mean_angle: float = 0
    for i, angle in enumerate(angle_for_each_label):
        weighted_mean_angle += angle * orientation_net_probs[i]

    return weighted_mean_angle

def calc_weighted_mean_pose(angle_for_each_label: List[float], direction_net_output: torch.Tensor, dist_to_local_goal: float=1.5) -> Tuple[float, float]:
    direction_net_positive_probs: List[float] = F.softmax(direction_net_output[:4], dim=0).tolist() 
    # print(f"direction_net_positive_probs: {direction_net_positive_probs}")
    weighted_mean_x: float = 0
    weighted_mean_y: float = 0
    for i, angle in enumerate(angle_for_each_label):
        weighted_mean_x += math.cos(angle) * dist_to_local_goal * direction_net_positive_probs[i]
        weighted_mean_y += math.sin(angle) * dist_to_local_goal * direction_net_positive_probs[i]
    # print(f"weighted_mean_x: {weighted_mean_x}, weighted_mean_y: {weighted_mean_y}")
    return weighted_mean_x, weighted_mean_y

def is_far(relative_pose: torch.Tensor) -> bool:
    if abs(relative_pose[2]) > math.radians(25): return True
    if math.hypot(relative_pose[0], relative_pose[1]) > 1.5: return True
    return False

def main():
    print("=== test start ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-dirs", type=str, nargs='*')
    parser.add_argument("--direction-net-path", type=str, default="")
    parser.add_argument("--orientation-net-path", type=str, default="")
    parser.add_argument("--relpos-net-path", type=str, default="")
    parser.add_argument("--image-dir", type=str, default="/home/amsl/Pictures")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device ="cpu"

    fix_seed()

    direction_net = torch.jit.load(args.direction_net_path).eval().to(device)
    orientation_net = torch.jit.load(args.orientation_net_path).eval().to(device)
    relpos_net = torch.jit.load(args.relpos_net_path).eval().to(device)

    test_dataset = DatasetForDirectionNet(args.dataset_dirs)
    DatasetForDirectionNet.equalize_label_counts(test_dataset, max_gap_times=3)
    # test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    
    # loss_func = torch.nn.MSELoss()
    loss_func = torch.nn.MSELoss(reduction='none')

    data_num = test_dataset.__len__()
    print(f"data num: {data_num}")

    proposed_mses: Optional[torch.Tensor] = None
    rival_mses: Optional[torch.Tensor] = None

    angle_for_each_label: List[float] = [math.radians(25), 0, -math.radians(25)]
    with torch.no_grad():
        for data in test_loader:
            src_image, dst_image, _, _, relative_pose = data
            relative_pose = relative_pose.squeeze()
            if is_far(relative_pose): continue

            direction_net_output = direction_net(src_image.to(device), dst_image.to(device)).squeeze()
            orientation_net_output = orientation_net(src_image.to(device), dst_image.to(device)).squeeze()
            relpos_net_output = relpos_net(src_image.to(device), dst_image.to(device)).squeeze().to("cpu")
            weighted_mean_angle = calc_weighted_mean_angle(angle_for_each_label, orientation_net_output)
            weighted_mean_x, weighted_mean_y = calc_weighted_mean_pose(angle_for_each_label, direction_net_output)
            proposed_relative_pose = torch.Tensor([weighted_mean_x, weighted_mean_y, weighted_mean_angle])
            
            proposed_mse = loss_func(proposed_relative_pose, relative_pose)
            rival_mse = loss_func(relpos_net_output, relative_pose)

            if proposed_mses == None:
                proposed_mses = torch.unsqueeze(proposed_mse, 0)
            else:
                proposed_mses = torch.cat([proposed_mses, torch.unsqueeze(proposed_mse, 0)], dim=0)

            if rival_mses == None:
                rival_mses = torch.unsqueeze(rival_mse, 0)
            else:
                rival_mses = torch.cat([rival_mses, torch.unsqueeze(rival_mse, 0)], dim=0)

        print(f"proposed rmse: {torch.sqrt(torch.mean(proposed_mses, dim=0))}")
        print(f"rival rmse: {torch.sqrt(torch.mean(rival_mses, dim=0))}")


if __name__ == "__main__":
    main()
