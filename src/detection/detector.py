from ultralytics import YOLO
import numpy as np
import pandas as pd
from src.config.config import YOLO_MODEL_PATH, CONFIDENCE_LEVEL, YOLO_CLASSES_OF_INTEREST

class PedestrianDetector:
    """
    Detects pedestrians in video frames using the YOLO model.
    """
    
    def __init__(self):
        """
        Initializes the PedestrianDetector with the YOLO model.
        """
        self.model = YOLO(YOLO_MODEL_PATH)
        self.model.classes = YOLO_CLASSES_OF_INTEREST
        self.classes = self.model.model.names
    
    def predict(self, frame: np.ndarray) -> pd.DataFrame:
        """
        Performs pedestrian detection on the given frame.

        Args:
            frame (numpy.ndarray): The input video frame.

        Returns:
            pandas.DataFrame: DataFrame containing detection results.
        """
        results = self.model.predict(frame, conf=CONFIDENCE_LEVEL, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        if len(boxes) == 0:
            return pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
        
        df = pd.DataFrame(
            np.concatenate([boxes, conf.reshape(-1, 1), classes.reshape(-1, 1)], axis=1),
            columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class']
        )
        return df
    
    def get_labels(self, classes: np.ndarray) -> list:
        """
        Retrieves class labels based on class indices.

        Args:
            classes (numpy.ndarray): Array of class indices.

        Returns:
            list: List of class labels.
        """
        return [self.classes[int(cls)] for cls in classes]