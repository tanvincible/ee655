import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import math

# ==== Step 1: Custom Dataset ====
class CircleMNISTDataset(Dataset):
    def __init__(self, image_root, circle_mask_root, transform=None):
        self.image_paths = []
        self.labels = []
        self.circles = []
        self.transform = transform

        for digit in range(10):
            digit_path = os.path.join(image_root, str(digit))
            mask_path = os.path.join(circle_mask_root, str(digit))
            for filename in os.listdir(digit_path):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(digit_path, filename)
                    mask_img = Image.open(os.path.join(mask_path, filename)).convert('L')
                    np_mask = np.array(mask_img)

                    # Extract circle from white pixels
                    y, x = np.where(np_mask > 0)
                    if len(x) == 0 or len(y) == 0:
                        continue
                    xc = (x.min() + x.max()) / 2 / 28
                    yc = (y.min() + y.max()) / 2 / 28
                    r = max((x.max() - x.min()), (y.max() - y.min())) / 2 / 28

                    self.image_paths.append(img_path)
                    self.labels.append(digit)
                    self.circles.append([xc, yc, r])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        label = self.labels[idx]
        circle = torch.tensor(self.circles[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        return img, label, circle

# ==== Step 2: Network ====
class CircleNet(nn.Module):
    def __init__(self):
        super(CircleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 128), nn.ReLU(), nn.Linear(128, 10)
        )
        self.circle_regressor = nn.Sequential(
            nn.Linear(64*7*7, 128), nn.ReLU(), nn.Linear(128, 3), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x_flat = self.flatten(x)
        class_logits = self.classifier(x_flat)
        circle_pred = self.circle_regressor(x_flat)
        return class_logits, circle_pred

# ==== Step 3: IoU Function ====
def circle_iou(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return (min(r1, r2)**2) / (max(r1, r2)**2)
    r1_sq = r1 ** 2
    r2_sq = r2 ** 2
    part1 = r1_sq * math.acos((d**2 + r1_sq - r2_sq) / (2 * d * r1))
    part2 = r2_sq * math.acos((d**2 + r2_sq - r1_sq) / (2 * d * r2))
    part3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    inter = part1 + part2 - part3
    union = math.pi * (r1_sq + r2_sq) - inter
    return inter / union

# ==== Step 4: Evaluation ====
def evaluate(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels, circles in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            circles = circles.to(device)

            class_logits, circle_preds = model(images)
            pred_classes = torch.argmax(class_logits, dim=1)

            for i in range(images.size(0)):
                pred_class = pred_classes[i].item()
                true_class = labels[i].item()
                pred_circle = circle_preds[i].cpu().numpy()
                true_circle = circles[i].cpu().numpy()

                if pred_class == true_class:
                    iou = circle_iou(pred_circle, true_circle)
                else:
                    iou = 0.0
                total_iou += iou
                total_samples += 1
    mean_iou = total_iou / total_samples
    print(f"\n📊 Mean IoU on Test Set: {mean_iou:.4f}")
    return mean_iou

# ==== Step 5: Train & Run ====
def train_and_eval():
    train_root = "MNIST_Dataset_JPG_format/MNIST_JPG_training"
    test_root = "MNIST_Dataset_JPG_format/MNIST_JPG_testing"
    circle_train = "q1/q1b/MNIST_CircleMasks_Training"
    circle_test = "q1/q1b/MNIST_CircleMasks_Testing"

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    train_set = CircleMNISTDataset(train_root, circle_train, transform)
    test_set = CircleMNISTDataset(test_root, circle_test, transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CircleNet().to(device)

    criterion_class = nn.CrossEntropyLoss()
    criterion_circle = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for images, labels, circles in train_loader:
            images, labels, circles = images.to(device), labels.to(device), circles.to(device)
            optimizer.zero_grad()
            class_logits, circle_preds = model(images)
            loss_class = criterion_class(class_logits, labels)
            loss_circle = criterion_circle(circle_preds, circles)
            loss = loss_class + loss_circle
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    evaluate(model, test_loader, device)

if __name__ == "__main__":
    train_and_eval()
