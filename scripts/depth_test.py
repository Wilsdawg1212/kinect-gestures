import freenect
import numpy as np
import cv2

def get_depth():
    depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_MM)
    return depth

def get_ir():
    ir, _ = freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)
    return ir

while True:
    depth = get_depth()
    # visualize depth (clip for display)
    d = np.clip(depth, 0, 2000).astype(np.uint16)
    d8 = (d / 2000.0 * 255).astype(np.uint8)

    cv2.imshow("depth", d8)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
