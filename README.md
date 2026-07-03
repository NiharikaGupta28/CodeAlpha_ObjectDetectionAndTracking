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

## Progress Log
- Day 1: Environment setup, dependencies installed, webcam access verified, repo initialized