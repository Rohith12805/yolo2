import os
import time
import cv2
from ultralytics import YOLO

# 1. Configuration
VIDEO_PATH = "TestVideo/WhatsApp Video 2026-10-05 at 14.18.30.mp4"  # Place your dashcam MP4 file here
MODEL_PATH = "runs/detect/road_damage_phase2/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "weights/best.pt"

print(f"Loading YOLO model from: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(
        f"Error: Could not open video file '{VIDEO_PATH}'. Check the filename."
    )
    exit()

WINDOW_NAME = "ADAS Bumper Vision System - Real-Time Evaluation"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1024, 600)

paused = False
ai_enabled = True
prev_time = time.time()
annotated_frame = None

print("\n--- Controls ---")
print(" [Spacebar] : Pause / Resume Playback")
print(" [A]        : Toggle AI Detections On/Off")
print(" [R]        : Restart Video from Beginning")
print(" [Q]        : Quit Player\n")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            # Loop video automatically when it reaches the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        detected_count = 0
        pothole_warning = False

        if ai_enabled:
            # On-the-fly frame inference
            results = model.predict(
                frame, imgsz=512, conf=0.18, iou=0.45, verbose=False
            )
            annotated_frame = results[0].plot(line_width=2)
            detected_count = len(results[0].boxes)

            # Check if any detected box belongs to pothole class (class 3 / D40)
            for cls_idx in results[0].boxes.cls.tolist():
                if int(cls_idx) == 3:  # D40 = Pothole
                    pothole_warning = True
        else:
            annotated_frame = frame.copy()

        # Telemetry Banner Overlay
        h, w, _ = annotated_frame.shape
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

        status_text = "AI ACTIVE" if ai_enabled else "RAW SENSOR"
        status_color = (0, 255, 0) if ai_enabled else (180, 180, 180)
        cv2.putText(
            annotated_frame,
            f"Mode: {status_text}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (240, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            f"Defects in Frame: {detected_count}",
            (400, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if pothole_warning:
            cv2.putText(
                annotated_frame,
                "WARNING: POTHOLE DETECTED",
                (w - 380, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.imshow(WINDOW_NAME, annotated_frame)

    # 1 ms delay during playback; wait indefinitely when paused
    key = cv2.waitKey(0 if paused else 1) & 0xFF

    if key == ord(" "):  # Spacebar toggles pause
        paused = not paused
    elif key == ord("a") or key == ord("A"):  # Toggle AI
        ai_enabled = not ai_enabled
    elif key == ord("r") or key == ord("R"):  # Restart video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif key == ord("q") or key == ord("Q"):  # Exit
        break

cap.release()
cv2.destroyAllWindows()
