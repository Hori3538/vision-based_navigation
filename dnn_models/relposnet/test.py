import argparse
from datetime import datetime
import os
import random
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np

from relposnet import RelPosNet
from dataset_for_directionnet import DatasetForDirectionNet

def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("=== test start ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dirs", type=str, nargs='*')
    parser.add_argument("--weight-path", type=str)
    parser.add_argument("--image-dir", type=str, default="/home/amsl/Pictures")
    args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device ="cpu"

    seed = 42
    fix_seed(seed)

    model = RelPosNet().to(device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    test_dataset = DatasetForDirectionNet(args.dataset_dirs)
    # DatasetForDirectionNet.equalize_label_counts(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, drop_last=False)
    
    loss_func = torch.nn.MSELoss()

    # direction_label_counts = DatasetForDirectionNet.count_data_for_each_label(test_dataset)
    # print(f"direction_label_counts: {direction_label_counts}")

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

            image_tensor = torch.cat((src_image[0], dst_image[0]), dim=2).squeeze()
            image = (image_tensor*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            cv2.imshow("images", image)
            key = cv2.waitKey(0)
            if key == ord("q") or key == ord("c"):
                break
            if key == ord("r"):
                image_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                image_dir = os.path.join(args.image_dir, image_name)
                print(f"image_dir: {image_dir}")
                cv2.imwrite(image_dir, image)

                print("saving image\n")
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
