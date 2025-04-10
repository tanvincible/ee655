import cv2
import numpy as np
from pathlib import Path

def generate_otsu_masks(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for class_dir in input_dir.glob("*"):
        if class_dir.is_dir():
            output_class_dir = output_dir / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            for image_path in class_dir.glob("*.jpg"):
                # Read grayscale image
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                # Apply Gaussian blur
                blur = cv2.GaussianBlur(img, (5, 5), 0)
                # Otsu's thresholding
                _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Save the mask
                output_path = output_class_dir / image_path.name
                cv2.imwrite(str(output_path), mask)

# Usage:
generate_otsu_masks(
    Path("MNIST_Dataset_JPG_format/MNIST_JPG_training"),
    Path("MNIST_Masks_Training")
)

generate_otsu_masks(
    Path("MNIST_Dataset_JPG_format/MNIST_JPG_testing"),
    Path("MNIST_Masks_Testing")
)
