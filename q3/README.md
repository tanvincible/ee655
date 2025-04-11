# Q3 - Classification with Circlization using Deep Learning

## 📌 Objective

Train a deep learning network **from scratch** to:
- **Classify** digits (0–9) from the `MNIST_CircleMasks_Training` dataset.
- **Predict tight bounding circles** around the digits for **localization** (circlization).

> The model is evaluated using **IoU (Intersection over Union)** on the predicted vs ground truth circles.

### ⚠️ Evaluation Rule:
If the **classification is incorrect**, the **IoU score is considered zero**.

---

## 📁 Dataset Structure

Uses the dataset from Q1(b), structured as:

```
q1/q1b/
├── MNIST_CircleMasks_Training/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
├── MNIST_CircleMasks_Testing/
│   ├── 0/
│   ├── 1/
│   └── ...
└── q1b.py
```

Each image file is a `28x28` grayscale image with a **digit** centered inside a **white circular mask** over black background.

---

## 🧠 Model Architecture

- A **CNN** with:
  - 3 convolutional layers with ReLU and MaxPooling
  - 2 separate heads:
    - `Classification Head`: Predicts the digit class (0–9).
    - `Regression Head`: Predicts 3 values `(x_center, y_center, radius)` for the tight circle.

---

## 🚀 How to Run

1. Activate your virtual environment:
   ```bash
   source ee655/bin/activate
   ```

2. Run the training script:
   ```bash
   python q3/train_and_evaluate.py
   ```

---

## 🧪 Evaluation: IoU Metric

After training, the model is evaluated on the test set using the **IoU metric** between:
- Predicted circle `(x, y, r)`
- Ground truth circle

If the classification is wrong, the IoU is automatically set to `0.0`.

Output:
```bash
Epoch 1: Loss = 162.4205
Epoch 2: Loss = 48.5278
Epoch 3: Loss = 33.3905
Epoch 4: Loss = 23.5051
Epoch 5: Loss = 20.5727
Epoch 6: Loss = 14.8037
Epoch 7: Loss = 12.4969
Epoch 8: Loss = 9.8469
Epoch 9: Loss = 9.3188
Epoch 10: Loss = 6.9169

📊 Mean IoU on Test Set: 0.9698
```

---

## 📊 Dependencies

Install dependencies (inside virtual env):
```bash
pip install opencv-python numpy torch torchvision matplotlib
```

---

## 📂 Files

- `train_and_evaluate.py`: Main script for training and evaluation.
- `README.md`: You are here.

---

## ✅ Notes

- Ground truth circles were generated using `cv2.minEnclosingCircle` in Q1(b).
- All classification and localization is done using a **single network** with two outputs.
- Designed for reproducibility and compliance with assignment instructions.
