import freenect
import numpy as np
import cv2

from src.segment import get_background_room, get_foreground_mask

def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Normalize a depth-like array into an 8-bit image for cv2.imshow."""
    if img is None:
        return np.zeros((480, 640), dtype=np.uint8)
    img = img.astype(np.float32)
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    out = (255.0 * (img - mn) / (mx - mn)).astype(np.uint8)
    return out

def main():
    background = None

    cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    print("Controls: [w] capture background, [q] or [esc] quit")

    while True:
        # Grab current depth frame from Kinect
        depth, _ = freenect.sync_get_depth()
        if depth is None:
            continue

        # Display depth for sanity (normalized)
        depth_vis = normalize_for_display(depth)
        cv2.imshow("depth", depth_vis)

        # Compute + display mask if background has been captured
        if background is not None:
            mask = get_foreground_mask(depth, background)

            # Ensure mask is displayable (0/255 uint8)
            if mask.dtype != np.uint8:
                mask_vis = (mask.astype(np.uint8) * 255)
            else:
                # If it's already uint8 but 0/1, scale up
                mask_vis = mask * (255 if mask.max() <= 1 else 1)

            cv2.imshow("mask", mask_vis)
        else:
            # Show an empty mask window until background is captured
            cv2.imshow("mask", np.zeros_like(depth_vis))

        key = cv2.waitKey(1) & 0xFF

        if key == ord("w"):
            print("Capturing background...")
            background = get_background_room()
            print("Background captured.")

        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    freenect.sync_stop()


if __name__ == "__main__":
    main()


    