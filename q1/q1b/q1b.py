import cv2
import numpy as np
from pathlib import Path

def generate_circle_masks(mask_dir, output_dir):
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    for class_dir in mask_dir.glob("*"):
        if class_dir.is_dir():
            output_class_dir = output_dir / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            for mask_path in class_dir.glob("*.jpg"):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                circle_mask = np.zeros_like(mask)
                if contours:
                    # Use largest contour
                    largest = max(contours, key=cv2.contourArea)
                    (x, y), radius = cv2.minEnclosingCircle(largest)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(circle_mask, center, radius, 255, -1)

                output_path = output_class_dir / mask_path.name
                cv2.imwrite(str(output_path), circle_mask)

# Usage:
generate_circle_masks(
    Path("MNIST_Masks_Training"),
    Path("MNIST_CircleMasks_Training")
)

generate_circle_masks(
    Path("MNIST_Masks_Testing"),
    Path("MNIST_CircleMasks_Testing")
)
