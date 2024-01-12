import argparse
import math

import torch
from torch.utils.data import DataLoader

from my_models import RelPosNet
from dataset_for_directionnet import DatasetForDirectionNet
from dnn_utils import fix_seed, image_tensor_cat_and_show

def main():
    print("=== test start ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dirs", type=str, nargs='*')
    parser.add_argument("--weight-path", type=str)
    parser.add_argument("--image-dir", type=str, default="/home/amsl/Pictures")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device ="cpu"

    fix_seed()

    model = RelPosNet().to(device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    test_dataset = DatasetForDirectionNet(args.dataset_dirs)
    DatasetForDirectionNet.equalize_label_counts(test_dataset, max_gap_times=3)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, drop_last=False)
    
    loss_func = torch.nn.MSELoss()

    data_num = test_dataset.__len__()
    print(f"data num: {data_num}")

    with torch.no_grad():
        mse_loss_sum = 0
        for data in test_loader:
            src_image, dst_image, _, _, relative_pose = data

            test_output = model(src_image.to(device), dst_image.to(device))
            test_loss = loss_func(test_output, relative_pose)
            mse_loss_sum += test_loss

        print(f"rmse avg {math.sqrt(mse_loss_sum / len(test_loader))}")
        print()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    with torch.no_grad():
        for data in test_loader:
            src_image, dst_image, _, _, relative_pose = data
            
            test_output = model(src_image.to(device), dst_image.to(device))
            print(f"model_output: {test_output}")
            print(f"relative_pose: {relative_pose}")
            print()

            continue_flag= image_tensor_cat_and_show(src_image[0], dst_image[0], args.image_dir)
            if(not continue_flag): break

if __name__ == "__main__":
    main()
