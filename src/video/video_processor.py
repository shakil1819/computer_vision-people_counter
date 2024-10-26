import cv2
from typing import Tuple
from src.config.config import VIDEO_SCALE_PERCENT, VIDEO_CODEC

class VideoProcessor:
    """
    Handles video loading, resizing, and saving operations.
    """
    
    def __init__(self, video_path: str):
        """
        Initializes the VideoProcessor with the given video path.

        Args:
            video_path (str): Path to the input video file.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.apply_scaling()
    
    def apply_scaling(self):
        """
        Applies scaling to video dimensions based on configuration.
        """
        if VIDEO_SCALE_PERCENT != 100:
            self.width = int(self.width * VIDEO_SCALE_PERCENT / 100)
            self.height = int(self.height * VIDEO_SCALE_PERCENT / 100)
    
    def get_frame(self):
        """
        Retrieves the next frame from the video.

        Returns:
            tuple: A tuple containing a boolean indicating success and the frame.
        """
        return self.cap.read()
    
    def release(self):
        """
        Releases the video capture object.
        """
        self.cap.release()
    
    def get_video_writer(self, output_path: str):
        """
        Initializes a VideoWriter object for saving annotated videos.

        Args:
            output_path (str): Path to save the annotated video.

        Returns:
            cv2.VideoWriter: The VideoWriter object.
        """
        return cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*VIDEO_CODEC),
            self.fps,
            (self.width, self.height)
        )