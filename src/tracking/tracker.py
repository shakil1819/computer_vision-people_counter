import numpy as np
from typing import Dict, Tuple

class Tracker:
    """
    Tracks pedestrians across video frames.
    """
    
    def __init__(self, threshold_centers: int, frame_max: int, patience: int):
        """
        Initializes the Tracker with tracking parameters.

        Args:
            threshold_centers (int): Distance threshold for tracking.
            frame_max (int): Maximum frame difference for tracking.
            patience (int): Number of frames to keep tracking history.
        """
        self.threshold_centers = threshold_centers
        self.frame_max = frame_max
        self.patience = patience
        self.centers_old: Dict[str, Dict[int, Tuple[int, int]]] = {}
        self.last_key = ''
    
    def update_tracking(
        self,
        obj_center: Tuple[int, int],
        current_frame: int
    ) -> Tuple[Dict[str, Dict[int, Tuple[int, int]]], str, bool, str]:
        """
        Updates tracking information with the new object center.

        Args:
            obj_center (tuple): (x, y) coordinates of the detected object center.
            current_frame (int): Current frame number.

        Returns:
            tuple: Updated tracking dictionary, object ID, is_new flag, and last_key.
        """
        is_new = False
        lastpos = [
            (k, list(center.keys())[-1], list(center.values())[-1])
            for k, center in self.centers_old.items()
        ]
        lastpos = [
            (k, pos) for k, frame, pos in lastpos
            if abs(frame - current_frame) <= self.frame_max
        ]
        previous_pos = [
            (k, pos) for k, pos in lastpos
            if np.linalg.norm(np.array(pos) - np.array(obj_center)) < self.threshold_centers
        ]
        
        if previous_pos:
            obj_id = previous_pos[0][0]
            self.centers_old[obj_id][current_frame] = obj_center
        else:
            if self.last_key:
                last_num = int(self.last_key.replace('ID', ''))
                obj_id = f'ID{last_num + 1}'
            else:
                obj_id = 'ID0'
            is_new = True
            self.centers_old[obj_id] = {current_frame: obj_center}
            self.last_key = obj_id
        
        return self.centers_old, obj_id, is_new, self.last_key
    
    def filter_tracks(self) -> Dict[str, Dict[int, Tuple[int, int]]]:
        """
        Filters out old tracking information based on patience.

        Returns:
            dict: Filtered tracking dictionary.
        """
        filtered = {}
        for k, frames in self.centers_old.items():
            filtered[k] = dict(list(frames.items())[-self.patience:])
        return filtered