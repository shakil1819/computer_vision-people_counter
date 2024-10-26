import cv2
from typing import List, Dict
import supervision as sv
import numpy as np
from src.video.video_processor import VideoProcessor

class ROIManager:
    """
    Handles extraction and processing of Regions of Interest (ROIs) from video frames.
    """
    
    def __init__(self, video_path: str, regions: List[str]):
        """
        Initializes the ROIManager with the video path and region names.

        Args:
            video_path (str): Path to the video file.
            regions (list): List of region names.
        """
        self.video_path = video_path
        self.regions = regions
        self.ROIs: List[Dict] = []
        self.video_info = sv.VideoInfo.from_video_path(video_path)
        self.generator = sv.get_video_frames_generator(video_path)
        self.iterator = iter(self.generator)
        self.frame = next(self.iterator)
    
    def select_rois(self) -> List[Dict]:
        """
        Allows user to select ROIs in the video frames.

        Returns:
            list: List of ROI dictionaries with names, polygons, and ranges.
        """
        for region_name in self.regions:
            points = self._select_region(region_name)
            roi = self._define_roi(points, region_name)
            self.ROIs.append(roi)
        return self.ROIs
    
    def _select_region(self, region_name: str) -> List[tuple]:
        """
        Captures user-selected points for a single ROI.

        Args:
            region_name (str): Name of the ROI.

        Returns:
            list: List of four (x, y) tuples defining the ROI polygon.
        """
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(region_name, img_display)

        img_display = self.frame.copy()
        cv2.namedWindow(region_name)
        cv2.setMouseCallback(region_name, mouse_callback)
        
        points = []
        while True:
            cv2.imshow(region_name, img_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or len(points) == 4:
                break
        
        cv2.destroyAllWindows()
        for _ in range(2):
            cv2.waitKey(1)
        
        return points
    
    def _define_roi(self, points: List[tuple], region_name: str) -> Dict:
        """
        Defines the ROI dictionary based on selected points.

        Args:
            points (list): List of (x, y) tuples defining the ROI.
            region_name (str): Name of the ROI.

        Returns:
            dict: ROI dictionary with name, polygon, and range.
        """
        roi_x = min(p[0] for p in points)
        roi_y = min(p[1] for p in points)
        roi_width = max(p[0] for p in points) - roi_x
        roi_height = max(p[1] for p in points) - roi_y
        
        x_range = [max(roi_x, 0), min(roi_x + roi_width, self.video_info.width - 1)]
        y_range = [max(roi_y, 0), min(roi_y + roi_height, self.video_info.height - 1)]
        
        roi = {
            "name": region_name,
            "polygon": points,
            "range": [x_range, y_range]
        }
        return roi