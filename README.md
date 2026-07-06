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
- Tested on CPU-only hardware (Intel i5-4210U, no dedicated GPU); achieves ~7-8 FPS with YOLOv8n at reduced input resolution.
- Detection limited to 80 COCO classes; objects outside this set (e.g., earbuds) not detectable.
- Robustness testing observations: dim lighting reduces detection reliability for smaller/detailed objects (e.g., plants); small objects (e.g., phone) have a distance "sweet spot" for reliable detection; fast movement causes brief bounding box flicker due to per-frame-only detection (motivates adding tracking); partial occlusion of a person was handled well.
- Tracking uses ByteTrack (motion-based matching via IoU + Kalman filter prediction). IDs remain stable through continuous visibility and brief occlusion, but reassign new IDs when an object fully exits and re-enters frame, since ByteTrack does not use appearance-based re-identification (a limitation Deep SORT addresses, at higher computational cost - not used here due to CPU-only hardware constraints).
- Object counts represent unique tracked appearances (based on track ID), not unique real-world individuals - the same person leaving and re-entering frame is counted as a new object, consistent with the tracking limitation noted above.

## Progress Log
- Day 1: Environment setup, dependencies installed, webcam access verified, repo initialized
- Day 2: Built reusable video pipeline (webcam + video file support), added FPS counter, added optional output saving with unique timestamped filenames
- Day 3: Integrated YOLOv8n for object detection, added bounding boxes with labels/confidence, added confidence threshold filtering, made box/text size scale with frame resolution, added FPS counter, optimized inference speed (~4-5 to ~7-8 FPS) using reduced input resolution (imgsz=320) - tested on CPU-only hardware (Intel i5-4210U, no dedicated GPU)
- Day 4: Added optional class filtering, conducted robustness testing (multiple objects, lighting, distance, movement, occlusion) - documented findings on model limitations
- Day 5: Integrated ByteTrack for multi-object tracking via model.track(), added persistent track IDs to bounding box labels. Confirmed IDs remain stable during continuous visibility and normal movement. Identified and investigated an ID-switching limitation: full frame exit/re-entry causes ID reassignment since ByteTrack uses motion-only (non-appearance-based) matching - a known trade-off versus Deep SORT, which was avoided here for CPU performance reasons. Experimented with increasing tracker buffer window; confirmed it does not resolve full re-entry cases, reverted to default settings.
- Day 6: Added creative enhancements - (1) unique object counting via track ID, (2) live on-screen stats panel showing FPS/current/total counts, (3) annotated output video saving with accurate playback speed (measured actual FPS rather than source FPS, avoiding sped-up output), (4) motion trail visualization with fading effect, toggleable live via 'T' key.
- Day 7: Conducted full robustness/regression testing across core functionality, feature toggles, stats accuracy, edge cases (missing file, empty frame, clean exit), and cross-testing on high-resolution nighttime traffic footage. Confirmed error handling works correctly for invalid sources. Investigated apparent "no detection" on dark traffic video - determined objects were actually being detected even at reduced resolution, but bounding boxes/labels were small and hard to visually distinguish against a busy, low-contrast night scene (a rendering/visibility issue, not a detection failure).