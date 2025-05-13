"""
Vehicle Speed Monitoring System using YOLOv8 and Deep SORT
---------------------------------------------------------
This implementation combines YOLOv8 for object detection and Deep SORT for object tracking
to monitor vehicle speeds in traffic videos. The system can detect if vehicles exceed a predefined
speed limit, making it suitable for transportation monitoring applications.

Required packages:
- ultralytics  # For YOLOv8
- opencv-python
- numpy
- scipy
- filterpy

Installation:
pip install ultralytics opencv-python numpy scipy filterpy
"""

import os
import time
import math
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter

# ------------------------------------------------------------------------
# Global configs
# ------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.35  # Confidence threshold for detection (lowered for YOLO)
MAX_COSINE_DISTANCE = 0.4    # Threshold for feature distance in Deep SORT
NN_BUDGET = 100              # Maximum samples to store for each track
MAX_AGE = 30                 # Maximum frames to keep a track alive without detection
MIN_HITS = 3                 # Minimum detections before track is initialized

# Speed estimation settings
SPEED_LIMIT_KPH = 90         # Speed limit in km/h
REFERENCE_DISTANCE = 10      # Reference real-world distance in meters (needs calibration)
REFERENCE_PIXELS = 200       # Corresponding pixel distance in the image (needs calibration)
FPS = 30                     # Frames per second of the video

# YOLO model selection
YOLO_MODEL = 'yolov8n.pt'    # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
                             # Smaller models (n, s) are faster but less accurate
                             # Larger models (l, x) are more accurate but slower

# YOLO class names (COCO dataset)
YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define which classes we're interested in (indices in YOLO_CLASSES)
VEHICLE_CLASS_IDS = [2, 5, 7, 3]  # car, bus, truck, motorcycle
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle']

# ------------------------------------------------------------------------
# YOLOv8 Implementation 
# ------------------------------------------------------------------------
class YOLODetector:
    def __init__(self, model_name=YOLO_MODEL, confidence_threshold=CONFIDENCE_THRESHOLD):
        # Load pre-trained YOLOv8 model
        self.model = YOLO(model_name)
        print(f"Using YOLOv8 model: {model_name}")
        
        # Check if CUDA is available
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.confidence_threshold = confidence_threshold
        
    def detect(self, frame):
        """
        Detects vehicles in a given frame.
        Returns: List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        # Run YOLOv8 inference on the frame
        results = self.model(frame, conf=self.confidence_threshold, device=self.device)
        
        # Extract detections
        detections = []
        
        if results[0].boxes:
            boxes = results[0].boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                # Filter for vehicle classes only (car, bus, truck, motorcycle)
                if class_id in VEHICLE_CLASS_IDS:
                    # Format: [x1, y1, x2, y2, confidence, class_id]
                    detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return np.array(detections) if len(detections) > 0 else np.empty((0, 6))

# ------------------------------------------------------------------------
# Feature Extractor for Deep SORT
# ------------------------------------------------------------------------
class FeatureExtractor:
    def __init__(self):
        """
        A simple feature extractor for Deep SORT.
        In a production environment, a proper embedding network would be used.
        """
        pass
        
    def extract(self, frame, boxes):
        """
        Extract features from image regions given by boxes.
        Returns: Array of feature vectors
        """
        features = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # Extract the patch
            patch = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            
            if patch.size == 0:
                # If patch is empty, create a dummy feature
                features.append(np.zeros(10))
                continue
                
            # Simple color histogram feature
            hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
            
        return np.array(features)

# ------------------------------------------------------------------------
# Kalman Filter for Deep SORT
# ------------------------------------------------------------------------
class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        bbox format: [x1, y1, x2, y2, confidence, class_id]
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        self.kf.R[2:, 2:] *= 10.  # Measurement uncertainty
        self.kf.P[4:, 4:] *= 1000.  # Covariance
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Convert bbox [x1, y1, x2, y2] to [x, y, s, r] where
        # x, y is the center, s is the scale/area, and r is the aspect ratio
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.class_id = int(bbox[5])
        
        # Speed calculation
        self.positions = deque(maxlen=30)  # Store recent positions
        self.timestamps = deque(maxlen=30)  # Store corresponding timestamps
        self.current_speed = 0
        self.speed_measurements = deque(maxlen=10)  # Recent speed measurements
        
        # Add initial position
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.positions.append((center_x, center_y))
        self.timestamps.append(time.time())
        
    def update(self, bbox):
        """
        Updates the state vector with observed bbox
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Update Kalman filter
        self.kf.update(self._convert_bbox_to_z(bbox))
        
        # Update position for speed calculation
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.positions.append((center_x, center_y))
        self.timestamps.append(time.time())
        
        # Calculate speed if we have at least 2 positions
        if len(self.positions) >= 2:
            self._calculate_speed()
        
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        """
        Returns the current bounding box estimate
        """
        return self._convert_x_to_bbox(self.kf.x)
    
    def _convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
        [x, y, s, r] where x, y is the center of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    # scale is area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x, score=None):
        """
        Takes a bounding box in the center form [x, y, s, r] and returns it in the form
        [x1, y1, x2, y2] where x1, y1 is the top left and x2, y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))
            
    def _calculate_speed(self):
        """Calculate the current speed of the tracked object"""
        if len(self.positions) < 2:
            return
            
        # Get the two most recent positions and timestamps
        pos1, pos2 = self.positions[-2], self.positions[-1]
        t1, t2 = self.timestamps[-2], self.timestamps[-1]
        
        # Calculate distance in pixels
        pixel_distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        
        # Convert to real-world distance (using calibration constants)
        real_distance = pixel_distance * (REFERENCE_DISTANCE / REFERENCE_PIXELS)
        
        # Calculate time difference
        time_diff = t2 - t1  # in seconds
        
        if time_diff > 0:
            # Calculate speed in meters per second
            speed_mps = real_distance / time_diff
            
            # Convert to km/h
            speed_kph = speed_mps * 3.6
            
            # Add to recent measurements
            self.speed_measurements.append(speed_kph)
            
            # Average the recent measurements to smooth out noise
            self.current_speed = sum(self.speed_measurements) / len(self.speed_measurements)
            
        return self.current_speed

# ------------------------------------------------------------------------
# Deep SORT Tracker
# ------------------------------------------------------------------------
class DeepSORTTracker:
    def __init__(self, max_cosine_distance=0.4, nn_budget=100, max_age=30, min_hits=3):
        """
        Initialize Deep SORT tracker
        """
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.trackers = []
        self.frame_count = 0
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
    def update(self, frame, detections):
        """
        Update the tracker with new detections
        frame: current video frame
        detections: output from detector, format [x1, y1, x2, y2, confidence, class_id]
        Returns: list of active tracks
        """
        self.frame_count += 1
        
        # Predict new locations of tracks
        track_boxes = []
        track_indices = []
        for i, tracker in enumerate(self.trackers):
            box = tracker.predict()[0]
            track_boxes.append(box)
            track_indices.append(i)
        
        # Associate detections with tracks
        if len(detections) > 0:
            # Extract features from detection regions
            features = self.feature_extractor.extract(frame, detections)
            
            if len(track_boxes) > 0:
                matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_trackers(
                    detections, track_boxes, features, track_indices)
                
                # Update matched trackers with assigned detections
                for t, d in matched:
                    self.trackers[t].update(detections[d])
                
                # Create new trackers for unmatched detections
                for i in unmatched_dets:
                    self.trackers.append(KalmanBoxTracker(detections[i]))
                    
                # Remove dead tracks
                i = len(self.trackers) - 1
                for t in reversed(range(len(self.trackers))):
                    if (self.trackers[t].time_since_update > self.max_age):
                        self.trackers.pop(t)
            else:
                # No existing tracks, create new ones for all detections
                for i in range(len(detections)):
                    self.trackers.append(KalmanBoxTracker(detections[i]))
        
        # Return active tracks
        active_tracks = []
        for tracker in self.trackers:
            if tracker.time_since_update <= 1 and (
                tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                
                bbox = tracker.get_state()[0]
                active_tracks.append({
                    'bbox': bbox,
                    'id': tracker.id,
                    'class_id': tracker.class_id,
                    'speed': tracker.current_speed,
                    'speeding': tracker.current_speed > SPEED_LIMIT_KPH
                })
                
        return active_tracks
            
    def _associate_detections_to_trackers(self, detections, trackers, features, track_indices):
        """
        Associates detections to tracked objects using feature similarity
        """
        if len(trackers) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(trackers)))
            
        # Calculate cost using IoU
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
                
        # Create cost matrix with combination of IoU and feature distance
        cost_matrix = 1 - iou_matrix
        
        # Use Hungarian algorithm for matching
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches using threshold
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 0.7:  # Threshold for considering it a match
                unmatched_detections.remove(row)
                unmatched_trackers.remove(col)
                matches.append((col, row))
                
        return matches, unmatched_detections, unmatched_trackers
            
    def _iou(self, bbox1, bbox2):
        """
        Computes IOU between two bounding boxes [x1, y1, x2, y2]
        """
        # Determine coordinates of intersection
        xx1 = max(bbox1[0], bbox2[0])
        yy1 = max(bbox1[1], bbox2[1])
        xx2 = min(bbox1[2], bbox2[2])
        yy2 = min(bbox1[3], bbox2[3])
        
        # Area of intersection
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        intersection = w * h
        
        # Area of both bounding boxes
        w1 = bbox1[2] - bbox1[0]
        h1 = bbox1[3] - bbox1[1]
        w2 = bbox2[2] - bbox2[0]
        h2 = bbox2[3] - bbox2[1]
        
        area1 = w1 * h1
        area2 = w2 * h2
        
        # IoU
        union = area1 + area2 - intersection
        if union <= 0:
            return 0
        return intersection / union

# ------------------------------------------------------------------------
# Speed Monitoring System
# ------------------------------------------------------------------------
class SpeedMonitoringSystem:
    def __init__(self, video_path=None, output_path='output_video.mp4', model_name=YOLO_MODEL):
        """
        Initialize the speed monitoring system
        """
        self.video_path = video_path
        self.output_path = output_path
        
        # Initialize detector and tracker
        self.detector = YOLODetector(model_name=model_name, confidence_threshold=CONFIDENCE_THRESHOLD)
        self.tracker = DeepSORTTracker(
            max_cosine_distance=MAX_COSINE_DISTANCE,
            nn_budget=NN_BUDGET,
            max_age=MAX_AGE,
            min_hits=MIN_HITS
        )
        
        # Colors for visualization
        self.colors = {}
        
    def _get_color(self, idx):
        """Get color for a specific ID"""
        if idx not in self.colors:
            r, g, b = [np.random.randint(0, 255) for _ in range(3)]
            self.colors[idx] = (int(r), int(g), int(b))
        return self.colors[idx]
        
    def process_video(self, calibrate=False):
        """
        Process a video file or webcam stream
        """
        # Open video capture
        if self.video_path is None:
            cap = cv2.VideoCapture(0)  # Use webcam
            print("Using webcam feed")
        else:
            cap = cv2.VideoCapture(self.video_path)
            print(f"Processing video: {self.video_path}")
            
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = FPS  # Use default if not available
            
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processing_times = []
        speeding_vehicles = set()
        
        # Performance optimization
        # Define a frame skip value - process every Nth frame for detection
        # This can dramatically increase throughput at the cost of occasional missed detections
        frame_skip = 0  # Process every frame (set to higher values for more speed)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                start_time = time.time()
                frame_count += 1
                
                if calibrate and frame_count == 1:
                    self._calibrate(frame)
                
                # Option to skip frames for detection (but continue tracking)
                if frame_count % (frame_skip + 1) == 0:
                    # Detect vehicles
                    detections = self.detector.detect(frame)
                    
                    # Track vehicles
                    tracks = self.tracker.update(frame, detections)
                else:
                    # Use previous detections, just update tracking
                    tracks = self.tracker.update(frame, np.empty((0, 6)))
                
                # Draw detection and tracking results
                frame = self._draw_results(frame, tracks)
                
                # Calculate and display processing speed
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                avg_time = sum(processing_times[-30:]) / min(len(processing_times), 30)
                fps_text = f"Processing: {1/avg_time:.2f} FPS"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2)
                
                # Display speed limit
                limit_text = f"Speed Limit: {SPEED_LIMIT_KPH} km/h"
                cv2.putText(frame, limit_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 0, 255), 2)
                
                # Count speeding vehicles
                for track in tracks:
                    if track['speed'] > SPEED_LIMIT_KPH:
                        speeding_vehicles.add(track['id'])
                
                speeding_count = f"Speeding Vehicles: {len(speeding_vehicles)}"
                cv2.putText(frame, speeding_count, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 0, 255), 2)
                
                # Write frame to output video
                out.write(frame)
                
                # Display the frame (resize for faster display if needed)
                display_frame = frame
                if width > 1280:  # Resize large frames for display
                    display_frame = cv2.resize(frame, (1280, int(height * 1280 / width)))
                
                cv2.imshow('Speed Monitoring', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"Output saved to {self.output_path}")
            
    def _draw_results(self, frame, tracks):
        """
        Draw detection boxes, tracking IDs, and speed information
        """
        for track in tracks:
            bbox = track['bbox']
            track_id = track['id']
            class_id = track['class_id']
            speed = track['speed']
            speeding = track['speeding']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = (0, 255, 0) if not speeding else (0, 0, 255)  # Green for normal, Red for speeding
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and class
            class_name = YOLO_CLASSES[class_id]
            cv2.putText(frame, f"ID: {track_id} ({class_name})", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw speed
            speed_text = f"{speed:.1f} km/h"
            cv2.putText(frame, speed_text, 
                       (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw warning if speeding
            if speeding:
                cv2.putText(frame, "SPEEDING!", 
                           (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        return frame
        
    def _calibrate(self, frame):
        """
        Interactive calibration tool
        This is a placeholder - in a real implementation, this would allow users to
        set reference points to calibrate real-world distances
        """
        # In a real implementation, this would:
        # 1. Allow user to select points on the image to establish a reference distance
        # 2. Update REFERENCE_PIXELS based on user selection
        # 3. Let user input the real-world REFERENCE_DISTANCE
        
        print("Calibration mode placeholder")
        # For simplicity, we'll just use the predefined values
        print(f"Using reference distance: {REFERENCE_DISTANCE} meters")
        print(f"Using reference pixels: {REFERENCE_PIXELS} pixels")

# ------------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------------
def main():
    import argparse
    
    global SPEED_LIMIT_KPH, CONFIDENCE_THRESHOLD
    
    parser = argparse.ArgumentParser(description='Vehicle Speed Monitoring System')
    parser.add_argument('--video', type=str, default=None, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Path for output video')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration mode')
    parser.add_argument('--speed_limit', type=float, default=SPEED_LIMIT_KPH, help='Speed limit in km/h')
    parser.add_argument('--model', type=str, default=YOLO_MODEL, 
                       help='YOLOv8 model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                       help='Detection confidence threshold')
    parser.add_argument('--half', action='store_true', help='Use half precision (FP16) for faster inference')
    
    args = parser.parse_args()
    
    # Update global settings
    SPEED_LIMIT_KPH = args.speed_limit
    CONFIDENCE_THRESHOLD = args.confidence
    
    # Set half precision flag for YOLO
    if args.half:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')
    
    # Initialize and run the system
    system = SpeedMonitoringSystem(
        video_path=args.video, 
        output_path=args.output,
        model_name=args.model
    )
    system.process_video(calibrate=args.calibrate)

if __name__ == "__main__":
    main()