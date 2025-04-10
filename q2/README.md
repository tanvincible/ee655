# Q2 - Foreground Extraction using Deep Learning

This task trains a deep learning model **from scratch** to perform **foreground segmentation** on the new dataset generated in **Q1(a)**. The model learns to extract digit foregrounds from MNIST images using the Otsu-based binary masks as ground truth.

---

## 🗂️ Dataset Structure

```
q1/q1a/
├── MNIST_Masks_Training/       # Ground truth segmentation masks for training
├── MNIST_Masks_Testing/        # Ground truth segmentation masks for testing
MNIST_Dataset_JPG_format/
├── MNIST_JPG_training/         # Original training images
├── MNIST_JPG_testing/          # Original testing images
```

Each image in the dataset has a corresponding binary mask indicating the digit foreground, generated using Otsu thresholding in Q1(a).

---

## 🧠 Model

- A simple **CNN-based encoder-decoder** (U-Net style) architecture is trained **from scratch**.
- Loss function: **Binary Cross-Entropy (BCE)**
- Evaluation metric: **IoU (Intersection over Union)**

---

## 🚀 How to Run

1. **Activate your virtual environment**

```bash
source ee655/bin/activate
```

2. **Install required dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the model**

```bash
python q2/foreground_extraction.py
```

4. **Outputs**

## TL;DR:
- Trained model: `q2/model.pth`
- IoU score printed after evaluation on the test set: **0.5439**

### Full output:
```bash
100%|█████████████████████████████████████████████████████████████| 938/938 [00:26<00:00, 36.05it/s]
Epoch 1: Loss = 0.3643
100%|█████████████████████████████████████████████████████████████| 938/938 [00:32<00:00, 29.01it/s]
Epoch 2: Loss = 0.3025
100%|█████████████████████████████████████████████████████████████| 938/938 [00:32<00:00, 29.04it/s]
Epoch 3: Loss = 0.2985
100%|█████████████████████████████████████████████████████████████| 938/938 [00:32<00:00, 28.48it/s]
Epoch 4: Loss = 0.2968
100%|█████████████████████████████████████████████████████████████| 938/938 [00:31<00:00, 29.47it/s]
Epoch 5: Loss = 0.2958
100%|█████████████████████████████████████████████████████████████| 938/938 [00:30<00:00, 30.27it/s]
Epoch 6: Loss = 0.2952
100%|█████████████████████████████████████████████████████████████| 938/938 [00:31<00:00, 29.87it/s]
Epoch 7: Loss = 0.2947
100%|█████████████████████████████████████████████████████████████| 938/938 [00:32<00:00, 29.03it/s]
Epoch 8: Loss = 0.2943
100%|█████████████████████████████████████████████████████████████| 938/938 [00:30<00:00, 30.49it/s]
Epoch 9: Loss = 0.2940
100%|█████████████████████████████████████████████████████████████| 938/938 [00:30<00:00, 30.52it/s]
Epoch 10: Loss = 0.2938
100%|█████████████████████████████████████████████████████████████| 938/938 [00:30<00:00, 30.56it/s]
Epoch 11: Loss = 0.2935
100%|█████████████████████████████████████████████████████████████| 938/938 [00:30<00:00, 30.28it/s]
Epoch 12: Loss = 0.2933
100%|█████████████████████████████████████████████████████████████| 938/938 [00:29<00:00, 31.30it/s]
Epoch 13: Loss = 0.2931
100%|█████████████████████████████████████████████████████████████| 938/938 [00:30<00:00, 30.60it/s]
Epoch 14: Loss = 0.2930
100%|█████████████████████████████████████████████████████████████| 938/938 [00:31<00:00, 29.60it/s]
Epoch 15: Loss = 0.2928
100%|█████████████████████████████████████████████████████████████| 938/938 [00:31<00:00, 29.57it/s]
Epoch 16: Loss = 0.2927
100%|█████████████████████████████████████████████████████████████| 938/938 [00:30<00:00, 30.66it/s]
Epoch 17: Loss = 0.2926
100%|█████████████████████████████████████████████████████████████| 938/938 [00:30<00:00, 31.25it/s]
Epoch 18: Loss = 0.2925
100%|█████████████████████████████████████████████████████████████| 938/938 [00:30<00:00, 31.22it/s]
Epoch 19: Loss = 0.2924
100%|█████████████████████████████████████████████████████████████| 938/938 [00:29<00:00, 31.37it/s]
Epoch 20: Loss = 0.2923
Test IoU: 0.5439
```

---

## 📊 Test Performance

| Metric | Value |
|--------|-------|
| IoU    | 0.5439  |

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- tqdm
- matplotlib

You can install them with:

```bash
pip install torch torchvision opencv-python numpy scikit-learn tqdm matplotlib
```

---

## 📁 Files

- `foreground_extraction.py`: Main script to train and evaluate the segmentation model
- `model.pth`: Saved model after training
- `README.md`: You are here.

---

## ✍️ Notes

- The masks are rough ground truths based on Otsu thresholding — performance is capped by their quality.
- The architecture can be swapped with a more advanced U-Net or DeepLab for better accuracy.
