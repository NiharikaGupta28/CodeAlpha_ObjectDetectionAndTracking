from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.2

# Set to None to show all classes, or a list like ["person", "car"] to filter
CLASSES_TO_DETECT = None

source = 0  # change to "test_video.mp4" to test on the video file instead
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"Error: Could not open source: {source}")
    exit()

cv2.namedWindow("YOLOv8 + ByteTrack", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 + ByteTrack", 640, 480)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # ---- KEY CHANGE: model.track() instead of model() ----
    # persist=True tells ByteTrack to remember tracked objects across frames
    results = model.track(frame, verbose=False, imgsz=320, persist=True)
    result = results[0]

    frame_height, frame_width = frame.shape[:2]
    scale_factor = frame_width / 640

    for box in result.boxes:
        confidence = float(box.conf[0])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if CLASSES_TO_DETECT is not None and class_name not in CLASSES_TO_DETECT:
            continue

        # Get the track ID (None if not yet assigned/confirmed by the tracker)
        track_id = int(box.id[0]) if box.id is not None else -1

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_thickness = max(1, int(2 * scale_factor))
        font_scale = 0.6 * scale_factor
        text_thickness = max(1, int(2 * scale_factor))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

        # Label now includes the track ID alongside class name and confidence
        label = f"ID:{track_id} {class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 + ByteTrack", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()