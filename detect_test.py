from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Run detection on a single image first (easier to verify than jumping straight to video)
# We'll use one frame captured from your webcam for this test
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not capture frame from webcam.")
else:
    # Run YOLO detection on this single frame
    results = model(frame)

    # results is a list (one entry per image we passed in - we only passed 1)
    result = results[0]

    print(f"Number of objects detected: {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls[0])          # numeric class ID
        class_name = model.names[class_id]  # convert to human-readable name like "person"
        confidence = float(box.conf[0])     # how confident YOLO is (0 to 1)
        print(f"Detected: {class_name}, Confidence: {confidence:.2f}")