import cv2
import numpy as np
from tqdm import tqdm

# Paths
video_path = 'q5/denis_walk.avi'
background_image_path = 'q5/bg.png'
output_path = 'q5/output_with_new_bg.avi'

# Load and resize background image
bg_image = cv2.imread(background_image_path)

# Load video
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

bg_image = cv2.resize(bg_image, (frame_w, frame_h))

# Estimate static background using average of initial N frames
N = 30
avg_bg = np.zeros((frame_h, frame_w, 3), dtype=np.float32)

for _ in range(N):
    ret, frame = cap.read()
    if not ret:
        break
    avg_bg += frame.astype(np.float32)

avg_bg /= N
avg_bg = avg_bg.astype(np.uint8)

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

# Morphology kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Process video
for _ in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Absolute difference from average background
    diff = cv2.absdiff(frame, avg_bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Step 2: Otsu's threshold
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Clean + THICKEN mask (this solves shredding)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    # Convert to 3 channel
    mask_3ch = cv2.merge([mask] * 3)
    inv_mask_3ch = cv2.merge([cv2.bitwise_not(mask)] * 3)

    # Foreground and background parts
    fg = cv2.bitwise_and(frame, mask_3ch)
    bg = cv2.bitwise_and(bg_image, inv_mask_3ch)
    final = cv2.add(fg, bg)

    out.write(final)

cap.release()
out.release()
print("✅ Output saved to:", output_path)
