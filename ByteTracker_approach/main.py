# Main implementation: Player Re-Identification using Body Part Appearances
# Based on research paper: 2310.14469v1.pdf
# Integrates YOLOv11 detection with two-stream ReID network

import cv2
import torch
import numpy as np
import yaml
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import os
import sys

# Import our custom modules
from models import TwoStreamReIDNetwork, load_pretrained_reid_model
from utils import (
    ImagePreprocessor, PlayerGallery, VideoProcessor, Visualizer,
    extract_player_crop, get_player_color, ByteTrackConfig
)
from config import Config

class PlayerReIDSystem:
    """
    Complete Player Re-Identification system implementing the research paper:
    "Player Re-Identification Using Body Part Appearances"
    
    Features:
    - YOLOv11 detection + ByteTrack tracking
    - Two-stream appearance + pose feature extraction
    - Compact bilinear pooling for feature fusion
    - Gallery-based re-identification with timeout mechanism
    """
    
    def __init__(self, yolo_model_path=None, reid_model_path=None):
        print("Initializing Player Re-ID System...")
        
        # Initialize device
        self.device = torch.device(Config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Initialize YOLO detector
        self.detector = self._load_yolo_detector(yolo_model_path or Config.YOLO_MODEL_PATH)
        
        # Initialize ReID network
        self.reid_network = self._load_reid_network(reid_model_path)
        
        # Initialize utilities
        self.preprocessor = ImagePreprocessor()
        self.gallery = PlayerGallery()
        self.visualizer = Visualizer()
        
        # Create tracker config
        self.tracker_config_path = self._create_tracker_config()
        
        print("System initialization complete!")
    
    def _load_yolo_detector(self, model_path):
        """Load and configure YOLOv11 detector."""
        try:
            detector = YOLO(model_path)
            print(f"Loaded YOLOv11 model from: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to default YOLOv11n model...")
            detector = YOLO('yolo11n.pt')
        
        return detector
    
    def _load_reid_network(self, model_path):
        """Load ReID network."""
        reid_net = load_pretrained_reid_model(model_path)
        print("Two-stream ReID network loaded successfully")
        return reid_net
    
    def _create_tracker_config(self):
        """Create ByteTrack configuration file."""
        config_path = 'bytetrack_custom.yaml'
        return ByteTrackConfig.save_tracker_config(config_path)
    
    def extract_reid_features(self, player_crops):
        """
        Extract ReID features from player crops using two-stream network.
        
        Args:
            player_crops: list of numpy arrays (player images)
            
        Returns:
            torch.Tensor: normalized feature vectors (N, feature_dim)
        """
        if not player_crops:
            return torch.empty(0, Config.FUSED_DIM)
        
        # Preprocess crops
        processed_crops = []
        for crop in player_crops:
            try:
                processed = self.preprocessor(crop)
                processed_crops.append(processed)
            except Exception as e:
                print(f"Error preprocessing crop: {e}")
                # Create dummy tensor if preprocessing fails
                dummy = torch.zeros(3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
                processed_crops.append(dummy)
        
        if not processed_crops:
            return torch.empty(0, Config.FUSED_DIM)
        
        # Stack into batch
        batch = torch.stack(processed_crops).to(self.device)
        
        # Extract features using two-stream network
        with torch.no_grad():
            features = self.reid_network(batch)
        
        return features
    
    def process_detections(self, frame, detections, frame_idx):
        """
        Process YOLO detections and perform re-identification.
        
        Args:
            frame: video frame (numpy array)
            detections: YOLO detection results
            frame_idx: current frame index
            
        Returns:
            list: assigned player IDs for each detection
        """
        if not detections or len(detections) == 0:
            return []
        
        # Extract bounding boxes and crops
        bboxes = []
        crops = []
        confidences = []
        
        for detection in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
            confidence = detection.conf[0].cpu().numpy()
            
            # Extract player crop
            crop = extract_player_crop(frame, (x1, y1, x2, y2))
            
            bboxes.append((x1, y1, x2, y2))
            crops.append(crop)
            confidences.append(confidence)
        
        # Extract ReID features
        features = self.extract_reid_features(crops)
        
        # Assign player IDs using gallery matching
        assigned_ids = []
        for i, feature_vector in enumerate(features):
            # Try to find match in gallery
            matched_id = self.gallery.find_match(feature_vector, frame_idx)
            
            if matched_id is not None:
                # Update existing player
                player_id = self.gallery.add_or_update(feature_vector, frame_idx, matched_id)
            else:
                # Create new player
                player_id = self.gallery.add_or_update(feature_vector, frame_idx)
            
            assigned_ids.append(player_id)
        
        return assigned_ids, bboxes, confidences
    
    def process_video(self, video_path, output_path=None, show_progress=True):
        """
        Process entire video with player re-identification.
        
        Args:
            video_path: path to input video
            output_path: path to save output video (optional)
            show_progress: whether to show progress bar
            
        Returns:
            dict: processing statistics
        """
        print(f"Processing video: {video_path}")
        
        stats = {
            'total_frames': 0,
            'total_detections': 0,
            'unique_players': 0,
            'processing_time': 0
        }
        
        import time
        start_time = time.time()
        
        with VideoProcessor(video_path, output_path) as video:
            total_frames = video.get_total_frames()
            
            if show_progress:
                pbar = tqdm(total=total_frames, desc="Processing frames")
            
            frame_idx = 0
            
            while True:
                ret, frame = video.read_frame()
                if not ret:
                    break
                
                try:
                    # Run YOLO detection + tracking
                    results = self.detector.track(
                        frame, 
                        persist=True, 
                        tracker=self.tracker_config_path,
                        verbose=False
                    )
                    
                    # Check if we have detections
                    if results and results[0].boxes is not None:
                        detections = results[0].boxes
                        
                        # Process detections with ReID
                        assigned_ids, bboxes, confidences = self.process_detections(
                            frame, detections, frame_idx
                        )
                        
                        # Update statistics
                        stats['total_detections'] += len(detections)
                        
                        # Visualize results
                        for i, (bbox, player_id, conf) in enumerate(zip(bboxes, assigned_ids, confidences)):
                            color = get_player_color(player_id)
                            frame = self.visualizer.draw_detection(
                                frame, bbox, player_id, conf, color
                            )
                    
                    # Draw statistics
                    gallery_stats = self.gallery.get_stats()
                    frame = self.visualizer.draw_stats(frame, gallery_stats, frame_idx)
                    
                    # Write frame to output
                    video.write_frame(frame)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    # Write original frame if processing fails
                    video.write_frame(frame)
                
                frame_idx += 1
                
                if show_progress:
                    pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        # Update final statistics
        stats['total_frames'] = frame_idx
        stats['unique_players'] = self.gallery.get_stats()['total_players']
        stats['processing_time'] = time.time() - start_time
        
        print(f"Processing complete!")
        print(f"Processed {stats['total_frames']} frames")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Unique players identified: {stats['unique_players']}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        
        if output_path:
            print(f"Output saved to: {output_path}")
        
        return stats
    
    def process_single_frame(self, frame):
        """
        Process a single frame (for real-time applications).
        
        Args:
            frame: input frame (numpy array)
            
        Returns:
            numpy array: frame with visualized detections and IDs
        """
        frame_idx = getattr(self, '_frame_counter', 0)
        self._frame_counter = frame_idx + 1
        
        try:
            # Run detection
            results = self.detector.track(
                frame, 
                persist=True, 
                tracker=self.tracker_config_path,
                verbose=False
            )
            
            # Process detections
            if results and results[0].boxes is not None:
                detections = results[0].boxes
                assigned_ids, bboxes, confidences = self.process_detections(
                    frame, detections, frame_idx
                )
                
                # Visualize
                for bbox, player_id, conf in zip(bboxes, assigned_ids, confidences):
                    color = get_player_color(player_id)
                    frame = self.visualizer.draw_detection(frame, bbox, player_id, conf, color)
            
            # Draw stats
            gallery_stats = self.gallery.get_stats()
            frame = self.visualizer.draw_stats(frame, gallery_stats, frame_idx)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return frame

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Player Re-Identification System')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--yolo-model', type=str, help='YOLOv11 model path')
    parser.add_argument('--reid-model', type=str, help='ReID model path')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Check input video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        return
    
    # Set default output path if not provided
    if not args.output:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{video_name}_reid_output.mp4"
    
    try:
        # Initialize system
        system = PlayerReIDSystem(
            yolo_model_path=args.yolo_model,
            reid_model_path=args.reid_model
        )
        
        # Process video
        stats = system.process_video(args.video, args.output)
        
        print("\\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def demo():
    """
    Demo function showing how to use the system programmatically.
    """
    print("Running Player Re-ID Demo...")
    
    # Initialize system
    system = PlayerReIDSystem()
    
    # Process demo video (replace with your video path)
    video_path = Config.VIDEO_PATH
    output_path = Config.OUTPUT_PATH
    
    if os.path.exists(video_path):
        stats = system.process_video(video_path, output_path)
        print(f"Demo completed! Check output: {output_path}")
    else:
        print(f"Demo video not found: {video_path}")
        print("Please update Config.VIDEO_PATH with your video file")

if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        main()
    else:
        # Run demo if no arguments
        demo()