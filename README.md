# CodeAlpha - Object Detection and Tracking

## Objective
Real-time object detection and tracking system using YOLOv8 for detection 
and ByteTrack for multi-object tracking, built with OpenCV for video I/O.

## Tech Stack
- Python 3.14
- OpenCV
- Ultralytics YOLOv8
- ByteTrack (via Ultralytics)
- FilterPy + SciPy (for bonus custom SORT implementation)

## How to Run
1. Clone this repo
2. Create virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python test_webcam.py` (webcam verification - Day 1)
6. Run: `python video_pipeline.py` (video pipeline with FPS counter - Day 2)
7. Run: `python detect_video.py` (YOLOv8 object detection with bounding boxes - Day 3). Edit the `source` variable to switch between webcam (`0`) and video file (`"test_video.mp4"`).

## Notes
- Tested on CPU-only hardware (Intel i5-4210U, no dedicated GPU); achieves ~7-8 FPS with YOLOv8n at reduced input resolution. Performance would be significantly higher on modern CPUs or with GPU acceleration.
- Detection is limited to the 80 COCO object classes; objects outside this set (e.g., earbuds) or very small/distant objects may not be reliably detected.

## Progress Log
- Day 1: Environment setup, dependencies installed, webcam access verified, repo initialized
- Day 2: Built reusable video pipeline (webcam + video file support), added FPS counter, added optional output saving with unique timestamped filenames
- Day 3: Integrated YOLOv8n for object detection, added bounding boxes with labels/confidence, added confidence threshold filtering, made box/text size scale with frame resolution, added FPS counter, optimized inference speed (~4-5 to ~7-8 FPS) using reduced input resolution (imgsz=320) - tested on CPU-only hardware (Intel i5-4210U, no dedicated GPU)