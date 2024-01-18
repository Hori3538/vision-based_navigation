import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from my_models import DirectionNet
from dataset_for_directionnet import DatasetForDirectionNet
from dnn_utils import fix_seed, image_tensor_cat_and_show
from torchvision import transforms
import torchvision
from torchvision.transforms.functional import equalize

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

    model = DirectionNet().to(device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    test_dataset = DatasetForDirectionNet(args.dataset_dirs)
    direction_label_counts = DatasetForDirectionNet.equalize_label_counts(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, drop_last=False,
                             num_workers=8, pin_memory=True)

    # direction_label_counts = DatasetForDirectionNet.count_data_for_each_label(test_dataset)
    print(f"direction_label_counts: {direction_label_counts}")

    data_num = test_dataset.__len__()
    print(f"data num: {data_num}")

    label_num: int = len(direction_label_counts)
    transform = torchvision.transforms.RandomEqualize(1.0)
    with torch.no_grad():
        direction_label_correct_count = torch.tensor([0] * label_num)
        for data in test_loader:
            src_image, dst_image, direction_label, _, relative_pose = data
            # src_image, dst_image = equalize(src_image), equalize(dst_image)
            src_image, dst_image = transform((src_image*255).type(torch.uint8))/255, transform((dst_image*255).type(torch.uint8))/255

            test_output = model(src_image.to(device), dst_image.to(device))
            onehot_output_direction = F.one_hot(test_output.max(1).indices,
                                                num_classes=label_num)
            direction_judge_tensor = torch.logical_and(
                    onehot_output_direction == direction_label.to(device),
                    onehot_output_direction == 1)

            direction_label_correct_count += torch.sum(direction_judge_tensor.to("cpu"), 0)

        direction_label_accuracy = direction_label_correct_count / direction_label_counts
        direction_total_accuracy = direction_label_correct_count.sum() / direction_label_counts.sum()

        print(f"direction label accuracy {direction_label_accuracy}")
        print(f"direction_total_accuracy: {direction_total_accuracy}")
        print()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    with torch.no_grad():
        for data in test_loader:
            src_image, dst_image, direction_label, _, relative_pose = data
            
            test_output = model(src_image.to(device), dst_image.to(device))
            print(f"relative_pose: {relative_pose}")
            print(f"direction_label: {direction_label}")
            print(f"drirection_output_prob: {F.softmax(test_output, dim=1)}")
            print()

            continue_flag= image_tensor_cat_and_show(src_image[0], dst_image[0], args.image_dir)
            if(not continue_flag): break

if __name__ == "__main__":
    main()
