import freenect
from src.capture import get_depth
import numpy as np

T = 50

def get_background_room(num_frames: int = 30):
    frames = []
    for _ in range(num_frames):
        depth = get_depth()
        if depth is None:
            continue
        frames.append(depth)

    if not frames:
        raise RuntimeError("Failed to capture any depth frames")

    return np.mean(frames, axis=0)


def get_foreground_mask(depth, bg_depth):
    validB = bg_depth > 0
    validD = depth > 0
    valid = validB & validD
    diff = depth - bg_depth
    raw_fg = (diff > T) & valid

    return raw_fg

