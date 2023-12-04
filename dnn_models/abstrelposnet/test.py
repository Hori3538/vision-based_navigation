import argparse
from datetime import datetime
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np

from model import AbstRelPosNet
from dataset import DatasetForAbstRelPosNet

def main():
    print("=== test start ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--weight-path", type=str)
    parser.add_argument("--image-dir", type=str, default="/home/amsl/Pictures")
    args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device ="cpu"

    model = AbstRelPosNet().to(device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    test_dataset = DatasetForAbstRelPosNet(args.dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, drop_last=False)

    direction_label_counts, orientation_label_counts = DatasetForAbstRelPosNet.count_data_for_each_label(test_dataset)
    print(f"direction_label_counts: {direction_label_counts}")
    print(f"orientation_label_counts: {orientation_label_counts}")

    data_num = test_dataset.__len__()
    print(f"data num: {data_num}")

    bin_num: int = len(orientation_label_counts)
    with torch.no_grad():
        direction_label_correct_count = torch.tensor([0] * (bin_num+1))
        orientation_label_correct_count = torch.tensor([0] * bin_num)
        for data in test_loader:
            src_image, dst_image, direction_label, orientation_label, relative_pose = data

            test_output = model(src_image.to(device), dst_image.to(device))
            onehot_output_direction = F.one_hot(test_output[:, :bin_num+1].max(1).indices,
                                                num_classes=bin_num+1)
            onehot_output_orientation = F.one_hot(test_output[:, bin_num+1:].max(1).indices,
                                                num_classes=bin_num)
            direction_judge_tensor = torch.logical_and(
                    onehot_output_direction == direction_label.to(device),
                    onehot_output_direction == 1)
            orientation_judge_tensor = torch.logical_and(
                    onehot_output_orientation == orientation_label.to(device),
                    onehot_output_orientation == 1)

            direction_label_correct_count += torch.sum(direction_judge_tensor.to("cpu"), 0)
            orientation_label_correct_count += torch.sum(orientation_judge_tensor.to("cpu"), 0)

        direction_label_accuracy = direction_label_correct_count / direction_label_counts
        orientation_label_accuracy = orientation_label_correct_count / orientation_label_counts
        direction_total_accuracy = direction_label_correct_count.sum() / direction_label_counts.sum()
        orientation_total_accuracy = orientation_label_correct_count.sum() / orientation_label_counts.sum()

        print(f"direction label accuracy {direction_label_accuracy}")
        print(f"orientation label accuracy {orientation_label_accuracy}")
        print(f"direction_total_accuracy: {direction_total_accuracy}")
        print(f"orientation_total_accuracy: {orientation_total_accuracy}")
        print()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    with torch.no_grad():
        for data in test_loader:
            src_image, dst_image, direction_label, orientation_label, relative_pose = data
            
            test_output = model(src_image.to(device), dst_image.to(device))
            print(f"relative_pose: {relative_pose}")
            print(f"drirection_output_prob: {F.softmax(test_output[:, :bin_num+1], dim=1)}")
            print(f"orientation_output_prob: {F.softmax(test_output[:, bin_num+1:], dim=1)}")
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
