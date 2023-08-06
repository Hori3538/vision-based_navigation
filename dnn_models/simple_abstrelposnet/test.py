import argparse
from datetime import datetime
import os

import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np

from model import SimpleAbstRelPosNet
from dataset import DatasetForSimpleAbstRelPosNet
from onehot_conversion import onehot_decoding, create_onehot_from_output

def main():
    print("=== test start ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--weight-path", type=str)
    parser.add_argument("--image-dir", type=str, default="/home/amsl/Pictures")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleAbstRelPosNet().to(device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    test_dataset = DatasetForSimpleAbstRelPosNet(args.dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, drop_last=False)

    data_num = test_dataset.__len__()
    print(f"data num: {data_num}")

    # label: 0:stop, 1:forwad, 2:left_turn, 3:right_turn, -1:out_range
    label_counts = torch.tensor([0]*4)

    with torch.no_grad():
        correct_count = 0
        for data in test_loader:
            src_image, dst_image, label = data
            abst_pose = label[:, [0]]
            concrete_pose = label[:, 1:]

            label_counts[0] += torch.sum(label == 0)
            label_counts[1] += torch.sum(label == 1)
            label_counts[2] += torch.sum(label == 2)
            label_counts[3] += torch.sum(label == 3)

            test_output = model(src_image.to(device), dst_image.to(device))
            onehot_output = create_onehot_from_output(test_output)
            decoded_output = onehot_decoding(onehot_output)

            judge_tensor = abst_pose == decoded_output
            correct_count += torch.sum(judge_tensor) 

        label_rate = label_counts / data_num * 100
        print(f"label_rate stop: {label_rate[0]:.3f}, forward: {label_rate[1]:.3f}, left_turn: {label_rate[2]:.3f}, left_turn: {label_rate[3]:.3f}")

        label_accuracy = correct_count / data_num
        print(f"label accuracy ... {label_accuracy:.3f}")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    with torch.no_grad():
        for data in test_loader:
            src_image, dst_image, label = data
            abst_pose = label[:, [0]]
            concrete_pose = label[:, 1:]

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
