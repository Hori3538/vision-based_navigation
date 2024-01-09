import argparse
from datetime import datetime
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np

from orientationnet import OrientationNet
from dataset_for_orientationnet import DatasetForOrientationNet

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

    model = OrientationNet().to(device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    test_dataset = DatasetForOrientationNet(args.dataset_dirs)
    DatasetForOrientationNet.equalize_label_counts(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, drop_last=False)

    orientation_label_counts = DatasetForOrientationNet.count_data_for_each_label(test_dataset)
    print(f"orientation_label_counts: {orientation_label_counts}")

    data_num = test_dataset.__len__()
    print(f"data num: {data_num}")

    bin_num: int = len(orientation_label_counts)
    with torch.no_grad():
        orientation_label_correct_count = torch.tensor([0] * bin_num)
        for data in test_loader:
            src_image, dst_image, _, orientation_label, relative_pose = data

            test_output = model(src_image.to(device), dst_image.to(device))
            onehot_output_orientation = F.one_hot(test_output.max(1).indices,
                                                num_classes=bin_num)
            orientation_judge_tensor = torch.logical_and(
                    onehot_output_orientation == orientation_label.to(device),
                    onehot_output_orientation == 1)

            orientation_label_correct_count += torch.sum(orientation_judge_tensor.to("cpu"), 0)

        orientation_label_accuracy = orientation_label_correct_count / orientation_label_counts
        orientation_total_accuracy = orientation_label_correct_count.sum() / orientation_label_counts.sum()

        print(f"orientation label accuracy {orientation_label_accuracy}")
        print(f"orientation_total_accuracy: {orientation_total_accuracy}")
        print()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    with torch.no_grad():
        for data in test_loader:
            src_image, dst_image, _, orientation_label, relative_pose = data
            
            test_output = model(src_image.to(device), dst_image.to(device))
            print(f"relative_pose: {relative_pose}")
            print(f"orientation_label: {orientation_label}")
            print(f"orientation_output_prob: {F.softmax(test_output, dim=1)}")
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