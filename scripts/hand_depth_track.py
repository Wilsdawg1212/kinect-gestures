import freenect
import numpy as np
import cv2

def get_depth_mm():
    depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_MM)
    return depth  # uint16 millimeters

def main():
    cv2.namedWindow("depth_vis", cv2.WINDOW_NORMAL)
    cv2.namedWindow("hand_mask", cv2.WINDOW_NORMAL)

    # Tune these
    valid_min_mm = 2000
    valid_max_mm = 10000
    band_mm = 220          # thickness of "hand band" beyond closest depth
    min_blob_area = 3000   # reject tiny blobs

    # smoothing for centroid (0=no smoothing, closer to 1 = heavier smoothing)
    alpha = 0.6
    smoothed = None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    while True:
        depth = get_depth_mm()

        # ---- Visualization image ----
        dclip = np.clip(depth, 0, valid_max_mm).astype(np.uint16)
        d8 = (dclip / valid_max_mm * 255).astype(np.uint8)
        d8_vis = cv2.applyColorMap(d8, cv2.COLORMAP_BONE)  # easier to read than gray

        # ---- Find closest depth in a valid range ----
        valid = (depth >= valid_min_mm) & (depth <= valid_max_mm)
        if not np.any(valid):
            cv2.imshow("depth_vis", d8_vis)
            cv2.imshow("hand_mask", np.zeros_like(d8))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            continue

        closest = int(depth[valid].min())

        # ---- Segment pixels near the closest surface ----
        hand_mask = (depth >= closest) & (depth <= closest + band_mm) & valid
        hand_mask = (hand_mask.astype(np.uint8) * 255)

        # ---- Clean up mask ----
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ---- Largest contour = hand candidate ----
        contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area >= min_blob_area:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Smooth the centroid so it doesn't jitter
                    if smoothed is None:
                        smoothed = (cx, cy)
                    else:
                        sx, sy = smoothed
                        sx = int(alpha * sx + (1 - alpha) * cx)
                        sy = int(alpha * sy + (1 - alpha) * cy)
                        smoothed = (sx, sy)

                    # Draw results
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(d8_vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.circle(d8_vis, (cx, cy), 6, (255, 255, 255), -1)
                    cv2.circle(d8_vis, smoothed, 8, (0, 0, 0), -1)

                    cv2.putText(
                        d8_vis,
                        f"closest={closest}mm area={int(area)}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        cv2.imshow("depth_vis", d8_vis)
        cv2.imshow("hand_mask", hand_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
