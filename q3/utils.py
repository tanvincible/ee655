# q3/utils.py
import math

def circle_iou(true, pred):
    x1, y1, r1 = true.cpu().numpy()
    x2, y2, r2 = pred.cpu().numpy()

    d = math.hypot(x1 - x2, y1 - y2)
    r1 *= 28
    r2 *= 28

    if d >= r1 + r2:
        return 0.0  # No overlap

    if d <= abs(r1 - r2):
        return (min(r1, r2)**2) / (max(r1, r2)**2)

    part1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2*d*r1))
    part2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2*d*r2))
    part3 = 0.5 * math.sqrt((-d + r1 + r2)*(d + r1 - r2)*(d - r1 + r2)*(d + r1 + r2))

    inter = part1 + part2 - part3
    union = math.pi * r1**2 + math.pi * r2**2 - inter
    return inter / union
