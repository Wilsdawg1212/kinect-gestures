import freenect
from src.capture import get_depth
import numpy as np
import cv2

T = 100

def get_background_room(num_frames: int = 40):
    frames = []
    for _ in range(num_frames):
        depth = get_depth()
        if depth is None:
            continue

        # total_elements = depth.size
        # non_zero_count = np.count_nonzero(depth)
        # zero_count = total_elements - non_zero_count
        # print(f"{zero_count} zeros out of {total_elements}")
        frames.append(depth)

    if not frames:
        raise RuntimeError("Failed to capture any depth frames")

    return np.median(frames, axis=0)


def get_foreground_mask(depth, bg_depth):
    validB = bg_depth > 0
    validD = (depth > 0) & (depth <= 5000)
    valid = validB & validD
    diff = abs(bg_depth - depth)
    
    raw_fg = (diff > T) & valid
    raw_fg_u8 = raw_fg.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    forg_mask = cv2.morphologyEx(raw_fg_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    forg_mask = cv2.morphologyEx(forg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return forg_mask

