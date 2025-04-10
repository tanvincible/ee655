# q3/q3.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from circle_dataset import CircleMNISTDataset
from utils import circle_iou
import os

class CircleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc_class = nn.Sequential(
            nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.fc_circle = nn.Sequential(
            nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
            nn.Linear(64, 3)  # x, y, r
        )

    def forward(self, x):
        feat = self.features(x)
        cls = self.fc_class(feat)
        circle = self.fc_circle(feat)
        return cls, circle

# Paths
train_img = "../q1/q1b/MNIST_CircleMasks_Training"
test_img = "../q1/q1b/MNIST_CircleMasks_Testing"
root_img = "../MNIST_Dataset_JPG_format/MNIST_JPG_training"
root_test_img = "../MNIST_Dataset_JPG_format/MNIST_JPG_testing"

# Dataset & Dataloader
train_ds = CircleMNISTDataset(root_img, train_img)
test_ds = CircleMNISTDataset(root_test_img, test_img)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CircleNet().to(device)
criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y_cls, y_circle in train_loader:
        x, y_cls, y_circle = x.to(device), y_cls.to(device), y_circle.to(device)
        optimizer.zero_grad()
        out_cls, out_circle = model(x)
        loss_cls = criterion_cls(out_cls, y_cls)
        loss_circle = criterion_reg(out_circle, y_circle)
        loss = loss_cls + loss_circle
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/model.pt")

# Evaluation with IoU
model.eval()
total_iou = 0
n = 0

with torch.no_grad():
    for x, y_cls, y_circle in test_loader:
        x = x.to(device)
        y_cls = y_cls.to(device)
        y_circle = y_circle.to(device)
        out_cls, out_circle = model(x)
        preds = torch.argmax(out_cls, dim=1)

        for i in range(x.size(0)):
            true_cls = y_cls[i].item()
            pred_cls = preds[i].item()
            if true_cls != pred_cls:
                iou = 0.0
            else:
                iou = circle_iou(y_circle[i], out_circle[i])
            total_iou += iou
            n += 1

print(f"\nMean IoU over test set: {total_iou / n:.4f}")
