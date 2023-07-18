import argparse
from datetime import datetime
from logging import critical
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from model import SimpleAbstRelPosNet
from dataset import DatasetForSimpleAbstRelPosNet
from onehot_conversion import onehot_decoding, onehot_encoding, create_onehot_from_output

def main():
    print("== Training Script ==")

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--dataset-dir", type=str)
    parser.add_argument("-p", "--pretrained-weights", type=str, default="")
    parser.add_argument("-n", "--num-data", type=int, default=10000)
    parser.add_argument("-l", "--lr-max", type=float, default=1e-4)
    parser.add_argument("-m", "--lr-min", type=float, default=1e-5)
    parser.add_argument("-t", "--train-ratio", type=int, default=8)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-w", "--num-workers", type=int, default=0)
    parser.add_argument("-e", "--num-epochs", type=int, default=60)
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

    model = SimpleAbstRelPosNet().to(device)
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights))
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = optim.RAdam(model.parameters(), lr=args.lr_max)
    scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=args.lr_min, max_lr=args.lr_max, step_size_up=10, mode="triangular", cycle_momentum=False
            )

    dataset = DatasetForSimpleAbstRelPosNet(args.dataset_dir)
    dataset, _ = random_split(dataset, [args.num_data, len(dataset) - args.num_data],
            generator=torch.Generator().manual_seed(args.seed))

    train_len = int(args.num_data * args.train_ratio / 10)
    train_dataset, valid_dataset = random_split(
                dataset, [train_len, args.num_data -train_len], generator=torch.Generator().manual_seed(args.seed)
            )

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

    writer = SummaryWriter(log_dir = log_dir)

    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()

        with tqdm(total=len(train_loader), ncols=100) as progbar:
            progbar.set_description(f"epoch {epoch}")

            epoch_train_loss = 0.0
            for data in train_loader:
                optimizer.zero_grad()

                src_image, dst_image, label = data
                src_image, dst_image = train_transform(src_image), train_transform(dst_image)

                abst_pose_label = label[:, 0]
                # print(f"abst_pose_label: {abst_pose_label}")
                encoded_abst_pose = onehot_encoding(abst_pose_label)
                # print(f"encoded_abst_pose_label: {encoded_abst_pose}")
                train_output = model(src_image.to(device), dst_image.to(device))
                # print(f"train_output: {train_output}")
                train_loss = criterion(train_output, encoded_abst_pose.to(device))
                epoch_train_loss += train_loss
                train_loss.backward()
                optimizer.step()
                progbar.update(1)
        epoch_train_loss /= len(train_loader)
        writer.add_scalar("loss/train", epoch_train_loss, epoch)

        model.eval()
        with torch.no_grad():
            epoch_valid_loss = 0.0
            correct_count = 0
            for data in valid_loader:
                src_image, dst_image, label = data
                abst_pose_label = label[:, 0]
                encoded_abst_pose = onehot_encoding(abst_pose_label)
                valid_output = model(src_image.to(device), dst_image.to(device))
                # print(f"valid_output: {valid_output}")
                onehot_output = create_onehot_from_output(valid_output)
                # print(f"onehot_output: {onehot_output}")
                decoded_output = onehot_decoding(onehot_output)
                print(f"abst_pose_label: {abst_pose_label}")
                print(f"decoded_output: {decoded_output}")

                judge_tensor = abst_pose_label == decoded_output
                correct_count += torch.sum(judge_tensor) 

                valid_loss = criterion(valid_output, encoded_abst_pose.to(device))
                epoch_valid_loss += valid_loss
            data_count = valid_dataset.__len__() // args.batch_size * args.batch_size
            label_accuracy = correct_count / data_count
            print(f"label accuracy ... {label_accuracy:.3f}")
        epoch_valid_loss /= len(valid_loader)
        writer.add_scalar("loss/valid", epoch_valid_loss, epoch)

        scheduler.step()

        if best_loss > epoch_valid_loss:
            model_path = os.path.join(weight_dir, f"best_loss.pt")
            torch.save(model.state_dict(), model_path)
            best_loss = epoch_valid_loss
            print(f"best_loss: {best_loss:.3f}")

        print(f"epoch {epoch} finished. train loss: {epoch_train_loss:.3f}, valid loss: {epoch_valid_loss:.3f}")
    
    writer.close()
    print("==Finished Training==")

if __name__ == "__main__":
    main()

