import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from orientationnet import OrientationNet
from dataset_for_orientationnet import DatasetForOrientationNet
from dnn_utils import fix_seed, image_tensor_cat_and_show

def main():
    print("=== test start ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dirs", type=str, nargs='*')
    parser.add_argument("--weight-path", type=str)
    parser.add_argument("--image-dir", type=str, default="/home/amsl/Pictures")
    args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device ="cpu"

    fix_seed()

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

            continue_flag= image_tensor_cat_and_show(src_image[0], dst_image[0], args.image_dir)
            if(not continue_flag): break

if __name__ == "__main__":
    main()
