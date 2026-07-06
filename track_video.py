from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict, deque
import cv2
import time

def draw_stats_panel(frame, fps, current_frame_counts, class_counts, frame_width):
    """
    Draws a semi-transparent stats panel in the top-right corner showing
    live FPS, objects currently in frame, and cumulative unique objects seen.
    """
    panel_x = frame_width - 250
    panel_y = 10
    line_height = 25

    # Build the lines of text we want to show
    lines = [f"FPS: {int(fps)}"]
    lines.append("-- In Frame Now --")
    for class_name, count in current_frame_counts.items():
        lines.append(f"{class_name}: {count}")
    lines.append("-- Total Seen --")
    for class_name, count in class_counts.items():
        lines.append(f"{class_name}: {count}")

    panel_height = line_height * len(lines) + 10

    # Draw a semi-transparent black rectangle as the panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (frame_width - 10, panel_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw each line of text on top of the panel
    for i, line in enumerate(lines):
        y = panel_y + 20 + (i * line_height)
        cv2.putText(frame, line, (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

model = YOLO("yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.5
SAVE_OUTPUT = False  # set to True to save the annotated video

# Set to None to show all classes, or a list like ["person", "car"] to filter
CLASSES_TO_DETECT = None

# Keeps track of every unique (class_name, track_id) pair we've ever seen
seen_track_ids = set()

show_trails = True  # current state - can be toggled live with 't' key

# Stores recent center-point history for each track ID (max 30 points per object)
track_history = defaultdict(lambda: deque(maxlen=30))

# Counts unique objects per class
class_counts = {}

source = 0 # change to "test_video.mp4" to test on the video file instead
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"Error: Could not open source: {source}")
    exit()

writer = None
output_filename = None
frame_buffer = []  # temporarily hold frames in memory until we know our actual FPS

if SAVE_OUTPUT:
    frame_width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"tracked_output_{timestamp}.mp4"

    
cv2.namedWindow("YOLOv8 + ByteTrack", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 + ByteTrack", 640, 480)

prev_time = 0
session_start_time = time.time()
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
    current_frame_counts = {}
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


        # Only count if this track ID hasn't been seen before (avoids double-counting same object across frames)
        if track_id != -1:
            unique_key = (class_name, track_id)
            if unique_key not in seen_track_ids:
                seen_track_ids.add(unique_key)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        # Count this object as currently visible in this frame (resets every frame)
        current_frame_counts[class_name] = current_frame_counts.get(class_name, 0) + 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Calculate the center point of this bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Add this position to the object's trail history (only if it has a valid track ID)
        if track_id != -1:
            track_history[track_id].append((center_x, center_y))

        box_thickness = max(1, int(2 * scale_factor))
        font_scale = 0.6 * scale_factor
        text_thickness = max(1, int(2 * scale_factor))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

        # Label now includes the track ID alongside class name and confidence
        label = f"ID:{track_id} {class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)

        # Draw the motion trail for this object (only if trails are currently enabled)
        if show_trails and track_id != -1 and len(track_history[track_id]) > 1:
            points = list(track_history[track_id])
            for i in range(1, len(points)):
                thickness = max(1, int((i / len(points)) * 4 * scale_factor))
                cv2.line(frame, points[i - 1], points[i], (0, 200, 255), thickness)

    frame = draw_stats_panel(frame, fps, current_frame_counts, class_counts, frame_width)

    if SAVE_OUTPUT:
        frame_buffer.append(frame.copy())
    
    hint_text = "Press 'T' to toggle trails | 'Q' to quit"
    cv2.putText(frame, hint_text, (10, frame_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("YOLOv8 + ByteTrack", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        show_trails = not show_trails
        print(f"Motion trails: {'ON' if show_trails else 'OFF'}")

cap.release()

print("\n--- Final Object Counts ---")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

if SAVE_OUTPUT and len(frame_buffer) > 0:
    elapsed_total = time.time() - session_start_time
    actual_fps = len(frame_buffer) / elapsed_total if elapsed_total > 0 else 20

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, actual_fps,
                              (frame_width_orig, frame_height_orig))
    for f in frame_buffer:
        writer.write(f)
    writer.release()
    print(f"Saved annotated output as: {output_filename} (at measured {actual_fps:.2f} FPS)")

cv2.destroyAllWindows()