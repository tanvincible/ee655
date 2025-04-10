import cv2
import numpy as np
import random
from pathlib import Path

def generate_concat_2x2(image_dir, mask_dir, output_image_dir, output_mask_dir, num_samples=1000):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_image_dir = Path(output_image_dir)
    output_mask_dir = Path(output_mask_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    classes = [str(i) for i in range(10)]
    image_paths = {cls: list((image_dir / cls).glob("*.jpg")) for cls in classes}
    mask_paths = {cls: list((mask_dir / cls).glob("*.jpg")) for cls in classes}

    for i in range(num_samples):
        chosen_classes = random.choices(classes, k=4)
        imgs, masks = [], []

        for cls in chosen_classes:
            idx = random.randint(0, len(image_paths[cls]) - 1)
            img = cv2.imread(str(image_paths[cls][idx]), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(mask_paths[cls][idx]), cv2.IMREAD_GRAYSCALE)
            imgs.append(img)
            masks.append(msk)

        top_img = np.hstack((imgs[0], imgs[1]))
        bottom_img = np.hstack((imgs[2], imgs[3]))
        full_img = np.vstack((top_img, bottom_img))

        top_mask = np.hstack((masks[0], masks[1]))
        bottom_mask = np.hstack((masks[2], masks[3]))
        full_mask = np.vstack((top_mask, bottom_mask))

        cv2.imwrite(str(output_image_dir / f"{i:05d}.jpg"), full_img)
        cv2.imwrite(str(output_mask_dir / f"{i:05d}_mask.jpg"), full_mask)

# Usage:
generate_concat_2x2(
    image_dir=Path("MNIST_Dataset_JPG_format/MNIST_JPG_training"),
    mask_dir=Path("MNIST_Masks_Training"),
    output_image_dir=Path("MNIST_Concat2x2_Images"),
    output_mask_dir=Path("MNIST_Concat2x2_Masks"),
    num_samples=1000
)
