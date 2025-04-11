import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Constants ---
IMAGE_DIR = 'q1/q1c/MNIST_Concat2x2_Images'
MASK_DIR = 'q1/q1c/MNIST_Concat2x2_Masks'
IMG_SIZE = 112  # since 28x2 = 56, and 2x = 112
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class MNISTSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.int64)

        mask = mask // 255

        if self.transform:
            img = self.transform(img)

        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask)

# --- Model: Simple U-Net ---
class UNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UNet, self).__init__()
        def CBR(in_c, out_c): return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.enc1 = CBR(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = CBR(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

# --- Dice Coefficient Metric ---
def dice_coef(pred, target, smooth=1.0):
    pred = F.one_hot(pred, NUM_CLASSES).permute(0, 3, 1, 2).float()
    target = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

# --- Load Data ---
img_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)])
mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR)])

train_imgs, test_imgs, train_masks, test_masks = train_test_split(img_files, mask_files, test_size=0.2, random_state=42)

train_ds = MNISTSegmentationDataset(train_imgs, train_masks)
test_ds = MNISTSegmentationDataset(test_imgs, test_masks)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1)

# --- Training ---
model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Train Loss: {total_loss / len(train_loader):.4f}")

# --- Evaluation ---
model.eval()
dice_scores = []
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        preds = torch.argmax(preds, dim=1)
        dice_scores.append(dice_coef(preds.cpu(), masks.cpu()))

print(f"\n✅ Test Dice Coefficient: {np.mean(dice_scores):.4f}")
