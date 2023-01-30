import argparse
from datetime import datetime
import os

import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np

from model import AbstRelPosNet
from modules.dataset import DatasetForAbstRelPosNet
from modules.onehot_conversion import onehot_decoding, onehot_encoding, create_onehot_from_output

def main():
    print("=== test start ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--weight-path", type=str)
    parser.add_argument("--image-dir", type=str, default="/home/amsl/Pictures")
    args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model = AbstRelPosNet().to(device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    test_dataset = DatasetForAbstRelPosNet(args.dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, drop_last=False)

    data_num = test_dataset.__len__()
    print(f"data num: {data_num}")
    label_count_minus = torch.tensor([0] * 3)
    label_count_same = torch.tensor([0] * 3)
    label_count_plus = torch.tensor([0] * 3)

    with torch.no_grad():
        correct_count = torch.tensor([0]*3)
        complete_correct_count = 0
        for data in test_loader:
            src_image, dst_image, label = data
            abst_pose = label[:, :3]
            encoded_abst_pose = onehot_encoding(abst_pose)
            concrete_pose = label[:, 3:]

            label_count_minus += torch.sum(label == -1, 0)[:3]
            label_count_same += torch.sum(label == 0, 0)[:3]
            label_count_plus += torch.sum(label == 1, 0)[:3]

            test_output = model(src_image.to(device), dst_image.to(device))
            onehot_output = create_onehot_from_output(test_output)
            decoded_output = onehot_decoding(onehot_output)

            judge_tensor = abst_pose == decoded_output
            correct_count += torch.sum(judge_tensor, 0) 
            complete_correct_count += torch.sum(torch.sum(judge_tensor, 1) == 3) 

        print(f"x_label_rate -1: {label_count_minus[0] / data_num * 100:.3f}, 0: {label_count_same[0] / data_num * 100:.3f}, 1: {label_count_plus[0] / data_num * 100:.3f}")
        print(f"y_label_rate -1: {label_count_minus[1] / data_num * 100:.3f}, 0: {label_count_same[1] / data_num * 100:.3f}, 1: {label_count_plus[1] / data_num * 100:.3f}")
        print(f"yaw_label_rate -1: {label_count_minus[2] / data_num * 100:.3f}, 0: {label_count_same[2] / data_num * 100:.3f}, 1: {label_count_plus[2] / data_num * 100:.3f}")

        label_accuracy = correct_count / data_num
        label_complete_accuracy = complete_correct_count / data_num
        print(f"label accuracy ... x: {label_accuracy[0]:.3f}, y: {label_accuracy[1]:.3f}, yaw: {label_accuracy[2]:.3f}")
        print(f"complete label accuracy: {label_complete_accuracy:.3f}")



    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    with torch.no_grad():
        for data in test_loader:
            src_image, dst_image, label = data
            abst_pose = label[:, :3]
            concrete_pose = label[:, 3:]

            test_output = model(src_image.to(device), dst_image.to(device))
            onehot_output = create_onehot_from_output(test_output)
            decoded_output = onehot_decoding(onehot_output)

            
            print(f"concrete_pose: {concrete_pose}")
            print(f"abst_pose: {abst_pose}")
            print(f"model's_output: {decoded_output[0]} \n")

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
