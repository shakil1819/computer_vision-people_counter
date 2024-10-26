"""
Configuration settings for the Pedestrian Detection Application.
"""

VIDEO_SCALE_PERCENT = 100
CONFIDENCE_LEVEL = 0.25
THRESHOLD_CENTERS = 30
FRAME_MAX = 10
PATIENCE = 100
ALPHA = 0.3
VIDEO_CODEC = "MP4V"

YOLO_MODEL_PATH = "yolov8x.pt"
YOLO_CLASSES_OF_INTEREST = [0]

GUI_TITLE = "People Counter with Computer Vision"
GUI_WIDTH = 600
GUI_HEIGHT = 400