import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# ---- config ----
MODEL_PATH = Path(os.environ["GESTURE_MODEL_PATH"]).expanduser()

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

CAM_INDEX = 0

# ---- globals used by callback ----
latest_text = "Starting..."
latest_score = 0.0
latest_ts_ms = 0

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def result_callback(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_text, latest_score, latest_ts_ms
    latest_ts_ms = timestamp_ms

    if result.gestures and len(result.gestures[0]) > 0:
        top = result.gestures[0][0]
        latest_text = str(f"{top.category_name} {len(result.gestures)}")
        latest_score = float(top.score)
    else:
        latest_text = "No gesture"
        latest_score = 0.0

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try CAM_INDEX 0/1/2 and check macOS Camera permissions.")

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        num_hands=2,
    )

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(time.time() * 1000)
            recognizer.recognize_async(mp_image, timestamp_ms)

            cv2.putText(frame_bgr, f"{latest_text} ({latest_score:.2f})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Gesture Recognizer", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
