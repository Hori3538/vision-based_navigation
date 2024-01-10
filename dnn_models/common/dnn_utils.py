import random
from datetime import datetime

import torch
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
