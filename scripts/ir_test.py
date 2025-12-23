# ir_test.py
import freenect
import numpy as np
import cv2

def get_ir_8bit():
    """
    Returns an 8-bit IR image (H x W) from the Kinect v1.
    """
    ir, _ = freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)
    return ir

def main():
    cv2.namedWindow("kinect_ir", cv2.WINDOW_NORMAL)

    while True:
        ir = get_ir_8bit()

        # Ensure uint8 for display (should already be uint8)
        if ir.dtype != np.uint8:
            ir = ir.astype(np.uint8)

        # Optional: slightly boost contrast for easier viewing
        ir_vis = cv2.equalizeHist(ir)

        cv2.imshow("kinect_ir", ir_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q or ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
