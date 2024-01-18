import random
from datetime import datetime
import os

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np

def fix_seed(seed=42):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def image_tensor_cat_and_show(reft_img: torch.Tensor, right_img: torch.Tensor, img_save_dir: str) -> bool:
        image_tensor = torch.cat((reft_img, right_img), dim=2).squeeze()
        image = (image_tensor*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imshow("images", image)
        key = cv2.waitKey(0)
        if key == ord("q") or key == ord("c"): return False
        if key == ord("r"):
            image_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            image_dir = os.path.join(img_save_dir, image_name)
            print(f"image_dir: {image_dir}")
            cv2.imwrite(image_dir, image)

            print("saving image\n")
        cv2.destroyAllWindows()
        return True

transform: nn.Sequential = nn.Sequential(
        transforms.ColorJitter(
            # brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # type: ignore
            # brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2),  # type: ignore
            brightness=[0.3, 2.0], contrast=0.7, saturation=0.7, hue=0.2),  # type: ignore
        # transforms.RandomGrayscale(0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], 0.2),
        # transforms.RandomErasing(0.2, scale=(0.05, 0.1), ratio=(0.33, 1.67)),
    )
