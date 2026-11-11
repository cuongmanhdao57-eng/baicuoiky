from ultralytics import YOLO
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = YOLO(os.path.join(APP_DIR, "models", "yolov8n.pt"))
