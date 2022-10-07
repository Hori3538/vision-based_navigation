import argparse
from datetime import datetime
import os

import torch
from torch import inf, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import AbstRelPosNet
from modules.dataset import DatasetForAbstRelPosNet
from loss_func import AbstPoseLoss

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
    parser.add_argument("-e", "--num-epochs", type=int, default=50)
    parser.add_argument("-a", "--beta", type=int, default=1.0)
    parser.add_argument("-i", "--weight-dir", type=str, default="./weights")
    parser.add_argument("-o", "--log-dir", type=str, default="./logs")
    parser.add_argument("-r", "--dirs-name", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.bencmark = True
    torch.multiprocessing.set_start_method("spawn") if args.num_workers>0 else None

    dirs_name = args.dirs_name if args.dirs_name else datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(args.log_dir, dirs_name)
    weight_dir = os.path.join(args.weight_dir, dirs_name)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "args.txt"), mode="w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}, {value}\n")

    model = AbstRelPosNet().to(device)
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights))
    criterion = AbstPoseLoss(args.beta, device)
    optimizer = optim.RAdam(model.parameters(), lr=args.lr_max)
    scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=args.lr_min, max_lr=args.lr_max, step_size_up=10, mode="triangular", cycle_momentum=False
            )

    dataset = DatasetForAbstRelPosNet(args.dataset_dir)
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
                abst_pose = label.clone().detach()[0, :3]
                train_output = model(src_image.to(device), dst_image.to(device))
                train_loss = criterion(train_output, abst_pose.to(device))
                epoch_train_loss += train_loss
                train_loss.backward()
                optimizer.step()
                progbar.update(1)
        epoch_train_loss /= len(train_loader)
        writer.add_scalar("loss/train", epoch_train_loss, epoch)

        model.eval()
        with torch.no_grad():
            epoch_valid_loss = 0.0
            for data in valid_loader:
                src_image, dst_image, label = data
                abst_pose = label.clone().detach()[0, :3]
                valid_output = model(src_image.to(device), dst_image.to(device))
                valid_loss = criterion(valid_output, abst_pose.to(device))
                epoch_valid_loss += valid_loss
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

