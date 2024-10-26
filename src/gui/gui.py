import tkinter as tk
from tkinter import filedialog, messagebox
from src.video.video_processor import VideoProcessor
from src.roi.roi_manager import ROIManager
from src.detection.detector import PedestrianDetector
from src.tracking.tracker import Tracker
from src.utils.utils import resize_frame
from src.config.config import (
    GUI_TITLE, GUI_WIDTH, GUI_HEIGHT,
    VIDEO_SCALE_PERCENT, CONFIDENCE_LEVEL,
    THRESHOLD_CENTERS, FRAME_MAX, PATIENCE, ALPHA
)
import os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

class PedestrianDetectionApp:
    """
    GUI Application for Pedestrian Detection.
    """
    
    def __init__(self, root):
        """
        Initializes the GUI components.

        Args:
            root (Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title(GUI_TITLE)
        
        # Video Path
        tk.Label(root, text="Video Path:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)
        self.entry_video_path = tk.Entry(root, width=50)
        self.entry_video_path.grid(row=0, column=1, padx=10, pady=10, columnspan=2)
        tk.Button(root, text="Browse", command=self.browse_video_path).grid(row=0, column=3, padx=10, pady=10)
        
        # Target Directory
        tk.Label(root, text="Target Directory:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.E)
        self.entry_target_dir = tk.Entry(root, width=50)
        self.entry_target_dir.grid(row=1, column=1, padx=10, pady=10, columnspan=2)
        tk.Button(root, text="Browse", command=self.browse_target_dir).grid(row=1, column=3, padx=10, pady=10)
        
        # Number of Regions
        tk.Label(root, text="Number of Regions:").grid(row=2, column=0, padx=10, pady=10, sticky=tk.E)
        self.entry_num_regions = tk.Entry(root, width=50)
        self.entry_num_regions.grid(row=2, column=1, padx=10, pady=10)
        
        # Region Names
        tk.Label(root, text="Region Names (comma-separated):").grid(row=3, column=0, padx=10, pady=10, sticky=tk.E)
        self.entry_region_names = tk.Entry(root, width=50)
        self.entry_region_names.grid(row=3, column=1, padx=10, pady=10, columnspan=2)
        
        # Start Button
        tk.Button(root, text="Start Detection", command=self.start_detection).grid(row=5, column=0, columnspan=4, pady=10)
        
        # Result Text Box
        self.result_text = tk.Text(root, height=10, width=60, state=tk.DISABLED, relief="groove", wrap=tk.WORD)
        self.result_text.grid(row=6, column=0, columnspan=4, pady=10)
        
        # Clear Button
        tk.Button(root, text="Clear", command=self.clear_values).grid(row=4, column=0, columnspan=4, pady=10)
    
    def browse_video_path(self):
        """
        Opens a file dialog to select a video file.
        """
        file_path = filedialog.askopenfilename(title="Select Video File")
        self.entry_video_path.delete(0, tk.END)
        self.entry_video_path.insert(0, file_path)
    
    def browse_target_dir(self):
        """
        Opens a directory dialog to select the target directory.
        """
        dir_path = filedialog.askdirectory(title="Select Target Directory")
        self.entry_target_dir.delete(0, tk.END)
        self.entry_target_dir.insert(0, dir_path)
    
    def clear_values(self):
        """
        Clears all input fields and the result text box.
        """
        self.entry_video_path.delete(0, tk.END)
        self.entry_target_dir.delete(0, tk.END)
        self.entry_num_regions.delete(0, tk.END)
        self.entry_region_names.delete(0, tk.END)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
    
    def start_detection(self):
        """
        Starts the pedestrian detection process.
        """
        video_path = self.entry_video_path.get()
        target_dir = self.entry_target_dir.get()
        region_names = self.entry_region_names.get()
        num_regions = self.entry_num_regions.get()
        
        if not video_path or not target_dir or not region_names or not num_regions.isdigit():
            messagebox.showerror("Error", "Please provide valid inputs.")
            return
        
        num_regions = int(num_regions)
        regions = [name.strip() for name in region_names.split(",")]
        
        if num_regions != len(regions):
            messagebox.showerror("Error", "Number of regions entered does not match the specified number.")
            return
        
        try:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Detection in progress...\n")
            self.result_text.config(state=tk.DISABLED)
            
            # Initialize components
            video_processor = VideoProcessor(video_path)
            roi_manager = ROIManager(video_path, regions)
            rois = roi_manager.select_rois()
            detector = PedestrianDetector()
            tracker = Tracker(
                threshold_centers=THRESHOLD_CENTERS,
                frame_max=FRAME_MAX,
                patience=PATIENCE
            )
            output_video_path = os.path.join(
                target_dir,
                f"Annotated_{os.path.basename(video_path).split('.')[0]}.mp4"
            )
            video_writer = video_processor.get_video_writer(output_video_path)
            roi_counts = {roi['name']: 0 for roi in rois}
            
            for frame_idx in tqdm(range(video_processor.frame_count), desc="Processing Frames"):
                is_frame, frame = video_processor.get_frame()
                if not is_frame:
                    break
                
                # Resize frame if necessary
                if VIDEO_SCALE_PERCENT != 100:
                    frame = resize_frame(frame, VIDEO_SCALE_PERCENT)
                
                for roi in rois:
                    x_range, y_range = roi['range']
                    roi_frame = frame[y_range[0]:y_range[1], x_range[0]:x_range[1]]
                    detections = detector.predict(roi_frame)
                    labels = detector.get_labels(detections['class'].values)
                    
                    for _, detection in detections.iterrows():
                        xmin, ymin, xmax, ymax, conf, cls = detection.astype(int)
                        center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
                        tracker.centers_old, obj_id, is_new, tracker.last_key = tracker.update_tracking(
                            obj_center=(center_x, center_y),
                            current_frame=frame_idx
                        )
                        if is_new:
                            roi_counts[roi['name']] += 1
                        cv2.rectangle(roi_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        cv2.circle(roi_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.putText(
                            roi_frame,
                            f"{obj_id}: {conf:.2f}",
                            (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.8,
                            (0, 0, 255),
                            1
                        )
                
                # Update tracking
                tracker.centers_old = tracker.filter_tracks()
                
                # Overlay ROI polygons
                overlay = frame.copy()
                for roi in rois:
                    pts = np.array(roi['polygon'], dtype=np.int32)
                    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                    cv2.fillPoly(overlay, [pts], (255, 0, 0))
                frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)
                
                # Display counts
                y_coordinate = 40
                for region, count in roi_counts.items():
                    cv2.putText(
                        frame,
                        f'People in {region}: {count}',
                        (30, y_coordinate),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        1
                    )
                    y_coordinate += 50
                
                # Write annotated frame to output video
                video_writer.write(frame)
            
            video_processor.release()
            video_writer.release()
            
            # Display results
            self.result_text.config(state=tk.NORMAL)
            self.result_text.insert(tk.END, "Detection completed.\nResults:\n")
            for region, count in roi_counts.items():
                self.result_text.insert(tk.END, f"People in {region}: {count}\n")
            self.result_text.insert(tk.END, f"\nAnnotated video saved at: {output_video_path}")
            self.result_text.config(state=tk.DISABLED)
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during detection: {str(e)}")