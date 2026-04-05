# Multi-Object Detection and Tracking

## 📌 Project Overview
This project detects and tracks multiple players in a sports video using computer vision.

## 🔧 Technologies Used
- Python
- YOLOv8 (Object Detection)
- DeepSORT (Tracking)
- OpenCV

## 🎯 Features
- Detects multiple people in video
- Assigns unique IDs to each person
- Tracks them across frames

## ▶️ How to Run
1. Install dependencies:
pip install ultralytics opencv-python numpy deep_sort_realtime

2. Run the code:
python main.py

## 📹 Input Video
https://youtu.be/EI7RWiVsr3w?si=cBsVov1LxGbiOw0o

## 📤 Output
Annotated video with bounding boxes and IDs

## ⚠️ Limitations
- ID switching during occlusion
- Depends on video quality