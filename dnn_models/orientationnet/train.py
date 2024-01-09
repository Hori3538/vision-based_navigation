import argparse
from datetime import datetime
import os

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from orientationnet import OrientationNet
from dataset_for_orientationnet import DatasetForOrientationNet
from focal_loss import Loss as FocalLoss

def main():
    print("== Training Script ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--train-dataset-dirs", type=str, nargs='*')
    parser.add_argument("-v", "--valid-dataset-dirs", type=str, default="", nargs='*')
    parser.add_argument("-p", "--pretrained-weights", type=str, default="")
    parser.add_argument("-n", "--num-data", type=int, default=50000)
    parser.add_argument("-l", "--lr-max", type=float, default=1e-3)
    parser.add_argument("-m", "--lr-min", type=float, default=1e-4)
    # parser.add_argument("-t", "--train-ratio", type=int, default=8)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-w", "--num-workers", type=int, default=0)
    parser.add_argument("-e", "--num-epochs", type=int, default=30)
    parser.add_argument("-i", "--weight-dir", type=str, default="./weights")
    parser.add_argument("-o", "--log-dir", type=str, default="./logs")
    parser.add_argument("-r", "--dirs-name", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu" 
    torch.backends.cudnn.bencmark = True
    torch.multiprocessing.set_start_method("spawn") if args.num_workers>0 else None

    dirs_name = args.dirs_name if args.dirs_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, dirs_name)
    weight_dir = os.path.join(args.weight_dir, dirs_name)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "args.txt"), mode="w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}, {value}\n")

    model = OrientationNet().to(device)
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights))

    # train_dataset = DatasetForOrientationNet(args.train_dataset_dirs, use_transform=True)
    train_dataset = DatasetForOrientationNet(args.train_dataset_dirs)
    DatasetForOrientationNet.equalize_label_counts(train_dataset)
    valid_dataset = DatasetForOrientationNet(args.valid_dataset_dirs)
    DatasetForOrientationNet.equalize_label_counts(valid_dataset)

    num_data: int = min(args.num_data, len(train_dataset))
    train_dataset, _ = random_split(train_dataset, [num_data, len(train_dataset) - num_data],
            generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=True, drop_last=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
            shuffle=True, drop_last=True, num_workers=args.num_workers)

    train_transform = nn.Sequential(
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # type: ignore
            transforms.RandomGrayscale(0.2),
            transforms.RandomApply([transforms.GaussianBlur(3)], 0.2),
            transforms.RandomErasing(0.2, scale=(0.05, 0.1), ratio=(0.33, 1.67)),
        )
    orientation_label_counts = DatasetForOrientationNet.count_data_for_each_label(train_dataset)
    print(f"orientation_label_counts: {orientation_label_counts}")
    criterion_for_orientation = FocalLoss(fl_gamma=2,
                                        samples_per_class=orientation_label_counts.tolist(),
                                        class_balanced=True, beta=0.99)
    optimizer = optim.RAdam(model.parameters(), lr=args.lr_max)
    step_size_up: int = 8 * len(train_dataset) / args.batch_size
    scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=args.lr_min, max_lr=args.lr_max, step_size_up=step_size_up, mode="triangular2", cycle_momentum=False)
    label_num: int = len(orientation_label_counts)

    writer = SummaryWriter(log_dir = log_dir)
    best_loss = float('inf')
    best_acc: float = 0
    for epoch in range(args.num_epochs):
        model.train()

        with tqdm(total=len(train_loader), ncols=100) as progbar:
            progbar.set_description(f"epoch {epoch}")

            epoch_train_orientation_loss = 0.0
            for data in train_loader:
                optimizer.zero_grad()

                src_image, dst_image, _, orientation_label, _ = data
                src_image, dst_image = train_transform(src_image), train_transform(dst_image)

                train_output = model(src_image.to(device), dst_image.to(device))
                orientation_loss = criterion_for_orientation(train_output,
                                                         orientation_label.to(device))
                epoch_train_orientation_loss += float(orientation_loss)
                orientation_loss.backward()
                optimizer.step()
                progbar.update(1)
        epoch_train_orientation_loss /= len(train_loader)
        writer.add_scalar("orientation_loss/train", epoch_train_orientation_loss, epoch)

        model.eval()
        with torch.no_grad():
            epoch_valid_orientation_loss = 0.0

            # direction_label_correct_count = torch.tensor([0] * label_num)
            orientation_label_correct_count = torch.tensor([0] * label_num)
            for data in valid_loader:
                src_image, dst_image, _, orientation_label, _ = data

                valid_output = model(src_image.to(device), dst_image.to(device))
                orientation_loss = criterion_for_orientation(valid_output,
                                                         orientation_label.to(device))

                onehot_output_orientation = F.one_hot(valid_output.max(1).indices,
                                                    num_classes=label_num)
                orientation_judge_tensor = torch.logical_and(
                        onehot_output_orientation == orientation_label.to(device),
                        onehot_output_orientation == 1)

                orientation_label_correct_count += torch.sum(orientation_judge_tensor.to("cpu"), 0)

                epoch_valid_orientation_loss += float(orientation_loss)

            orientation_label_counts_valid = DatasetForOrientationNet.count_data_for_each_label(valid_dataset)
            orientation_label_accuracy = orientation_label_correct_count / orientation_label_counts_valid
            orientation_total_accuracy = orientation_label_correct_count.sum() / orientation_label_counts_valid.sum()

            print(f"orientation label accuracy {orientation_label_accuracy}")
            print(f"orientation_total_accuracy: {orientation_total_accuracy}")

        epoch_valid_orientation_loss /= len(valid_loader)
        writer.add_scalar("orientation_loss/valid", epoch_valid_orientation_loss, epoch)
        writer.add_scalar("orientation_loss/orientation_total_accuracy", orientation_total_accuracy, epoch)

        scheduler.step()

        avg_acc: float = float(orientation_total_accuracy)
        if avg_acc > best_acc:
            model_path = os.path.join(weight_dir, f"best_acc.pt")
            torch.save(model.state_dict(), model_path)
            best_acc = avg_acc
            print(f"best_avg_acc: {best_acc:.3f}")

        if epoch_valid_orientation_loss < best_loss:
            model_path = os.path.join(weight_dir, f"best_loss.pt")
            torch.save(model.state_dict(), model_path)
            best_loss = epoch_valid_orientation_loss
            print(f"best_loss: {best_loss:.3f}")

        print(f"epoch {epoch} finished. train loss: {epoch_train_orientation_loss:.3f}, valid loss: {epoch_valid_orientation_loss:.3f}")
    
    writer.close()
    print("==Finished Training==")

if __name__ == "__main__":
    main()