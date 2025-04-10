# Q3 – Classification with Circlization

## 📝 Task
Train a Deep Learning network **from scratch** to perform:
1. **Classification** of digits (0–9) from images.
2. **Circlization**: Predict the **tightest enclosing circle** (center `x, y`, and `radius`) around the digit.

Evaluate the model using the **IoU (Intersection over Union)** metric.

> 🔔 If the classification is **incorrect**, assign **IoU = 0**.

---

## 📁 Folder Structure

```
q3/
├── q3.py                  # Main training + evaluation script
├── circle_dataset.py      # Custom Dataset loader for images + circle labels
├── utils.py               # Circle IoU computation
├── saved_models/          # Folder to store trained models
│   └── model.pt
```

---

## 📂 Dataset Dependencies

This script depends on datasets generated in **Q1(b)**:
- `../q1/q1b/MNIST_CircleMasks_Training/`
- `../q1/q1b/MNIST_CircleMasks_Testing/`

As well as raw images from:
- `../MNIST_Dataset_JPG_format/MNIST_JPG_training/`
- `../MNIST_Dataset_JPG_format/MNIST_JPG_testing/`

> ✅ Make sure this directory structure is correct before running.

---

## 📦 Requirements

Install dependencies:
```bash
pip install torch torchvision opencv-python numpy
```

---

## 🚀 How to Run

```bash
cd q3
python q3.py
```

This will:
- Train a CNN model from scratch on the dataset.
- Predict class and circle parameters.
- Evaluate using **IoU**, considering IoU = 0 for incorrect classifications.
- Print the final **mean IoU score**.

---

## 🧠 Model Details

- CNN with two `Conv2D` + `ReLU` + `MaxPool` layers.
- Two heads:
  - **Classification head**: Outputs digit class (0–9).
  - **Regression head**: Predicts normalized `(x_center, y_center, radius)` for the circle.

---

## 📊 Output Example

Epoch 10: Loss = 84.7689  
Mean IoU over test set: 0.0778

### Full Output:

```bash
Epoch 1: Loss = 334.4422
Epoch 2: Loss = 142.1294
Epoch 3: Loss = 120.3470
Epoch 4: Loss = 110.5330
Epoch 5: Loss = 102.8760
Epoch 6: Loss = 97.6974
Epoch 7: Loss = 93.9095
Epoch 8: Loss = 89.5419
Epoch 9: Loss = 85.9460
Epoch 10: Loss = 84.7689

Mean IoU over test set: 0.0778
```

---

## 🧪 Evaluation Criteria

- **Classification Accuracy** is not reported directly.
- **IoU Metric** is computed only if classification is correct.
- Otherwise, **IoU is 0** (as per question instructions).

---

## 🧰 Extra Utilities

- `utils.py`: Contains `circle_iou()` to compute IoU between two circles.
- `saved_models/model.pt`: Stores the best model trained.

---

## 📌 Note

All circle coordinates and radii are **normalized** to `[0, 1]` during training for stability. They are automatically scaled back for IoU computation.

---

## 👩‍💻 Author

Assignment solution for Q3 – Deep Learning Classification with Circlization (2 marks)
By [Tanvi Pooranmal Meena](https://github.com/tanvincible/)