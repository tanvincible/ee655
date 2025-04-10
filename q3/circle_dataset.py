# q3/circle_dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class CircleMNISTDataset(Dataset):
    def __init__(self, root_img, root_mask, transform=None):
        self.image_paths = []
        self.circle_labels = []
        self.classes = []
        self.transform = transform

        for cls in sorted(os.listdir(root_img)):
            img_dir = os.path.join(root_img, cls)
            mask_dir = os.path.join(root_mask, cls)

            for img_name in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, img_name)

                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.classes.append(int(cls))
                    # extract circle from mask
                    mask = cv2.imread(mask_path, 0)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        (x, y), r = cv2.minEnclosingCircle(contours[0])
                        self.circle_labels.append([x / 28, y / 28, r / 28])  # Normalize
                    else:
                        self.circle_labels.append([0, 0, 0])  # Fallback

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        class_label = self.classes[idx]
        circle_params = self.circle_labels[idx]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(class_label), torch.tensor(circle_params)
