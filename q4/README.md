# Q4: Semantic Segmentation on MNIST 2x2 Composite Dataset

This task involves training a deep learning model from scratch to perform **semantic segmentation** on the new dataset generated in **Q1(c)** — where MNIST digits have been spatially concatenated in a 2x2 grid. The goal is to classify each pixel into one of the digit classes (0–9).

---

## 📁 Dataset Structure

The dataset used here was generated in Q1(c). It consists of:

```
├── MNIST_Concat2x2_Images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── MNIST_Concat2x2_Masks/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
```

- Each image is a 2x2 composite of MNIST digits.
- Each mask contains **class labels from 0 to 9** representing the digit occupying that region.

---

## 🏗️ Model Architecture

A simple U-Net-like Convolutional Neural Network was used for semantic segmentation. It consists of:

- Encoder with multiple convolutional + ReLU + MaxPool layers.
- Decoder with transposed convolutions for upsampling.
- Final output layer with `10` output channels (one for each digit class).

---

## 🧪 Loss & Evaluation

- **Loss Function:** `CrossEntropyLoss` (applied pixel-wise).
- **Metric:** Dice Coefficient averaged across classes.

---

## 🚀 How to Run

### 1. Install Requirements

Activate your Python environment and install dependencies:

```bash
pip install torch torchvision opencv-python numpy
```

### 2. Train the Model

```bash
python q4/train_segmentation.py
```

### 3. Dice Score Output

The script prints the **mean Dice coefficient** on the test set after each epoch and saves the model.

---

## 📝 Notes

- The dataset must be pre-generated using `q1c.py`.
- Images and masks should be the same size (e.g., 128x128).
- Mask values must be in the range `[0, 9]` (no 255s or background-only masks).
- All masks are assumed to be **single-channel grayscale** with class indices as pixel values.

---

## 📊 Example Output

```
Epoch 1/10
Train Loss: 0.885
Val Dice Score: 0.712

Epoch 2/10
Train Loss: 0.643
Val Dice Score: 0.785

...
```

---

## 📂 File Overview

```
q4/
├── train_segmentation.py      # Main training script
└── README.md                  # You're here!
```

---

## ✏️ Author

Tanvi Pooranmal Meena