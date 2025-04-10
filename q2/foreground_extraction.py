import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

# ------------------------------
# Dataset
# ------------------------------
class MNISTSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = []
        self.mask_paths = []
        for label in os.listdir(img_dir):
            img_subdir = os.path.join(img_dir, label)
            mask_subdir = os.path.join(mask_dir, label)
            for fname in os.listdir(img_subdir):
                self.img_paths.append(os.path.join(img_subdir, fname))
                self.mask_paths.append(os.path.join(mask_subdir, fname))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# ------------------------------
# U-Net-like CNN (minimal)
# ------------------------------
class SimpleSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ------------------------------
# IoU Metric
# ------------------------------
def compute_iou(preds, masks):
    preds_bin = (preds > 0.5).float()
    ious = []
    for p, m in zip(preds_bin, masks):
        p = p.view(-1).cpu().numpy()
        m = m.view(-1).cpu().numpy()
        if np.sum(m) == 0 and np.sum(p) == 0:
            ious.append(1.0)
        elif np.sum(m) == 0 or np.sum(p) == 0:
            ious.append(0.0)
        else:
            ious.append(jaccard_score(m, p))
    return np.mean(ious)

# ------------------------------
# Training and Evaluation
# ------------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set = MNISTSegmentationDataset(
        img_dir="../MNIST_Dataset_JPG_format/MNIST_JPG_training",
        mask_dir="../q1/q1a/MNIST_Masks_Training"
    )
    test_set = MNISTSegmentationDataset(
        img_dir="../MNIST_Dataset_JPG_format/MNIST_JPG_testing",
        mask_dir="../q1/q1a/MNIST_Masks_Testing"
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    model = SimpleSegNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu()
            all_preds.append(preds)
            all_masks.append(masks)
    preds_cat = torch.cat(all_preds)
    masks_cat = torch.cat(all_masks)

    iou = compute_iou(preds_cat, masks_cat)
    print(f"Test IoU: {iou:.4f}")

if __name__ == "__main__":
    train_model()
