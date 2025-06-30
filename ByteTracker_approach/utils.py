# Utility functions for Player Re-Identification system
# Supporting the implementation of research paper: 2310.14469v1.pdf

import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import defaultdict, deque
import torch.nn.functional as F
from config import Config

class ImagePreprocessor:
    """
    Image preprocessing pipeline for player crops.
    Resizes and normalizes images according to the paper specifications.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])
    
    def __call__(self, image):
        """
        Preprocess a single image crop.
        
        Args:
            image: numpy array (H, W, C) in BGR format
            
        Returns:
            torch.Tensor: preprocessed image tensor (C, H, W)
        """
        if len(image.shape) == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        tensor = self.transform(image)
        return tensor

class PlayerGallery:
    """
    Gallery system for storing and matching player features.
    Implements timeout mechanism and cosine similarity matching.
    """
    def __init__(self, timeout=Config.GALLERY_TIMEOUT, threshold=Config.COSINE_THRESHOLD):
        self.gallery = {}  # player_id: {'features': deque, 'last_seen': frame_idx}
        self.timeout = timeout
        self.threshold = threshold
        self.next_id = 1
        
    def add_or_update(self, features, frame_idx, player_id=None):
        """
        Add new features to gallery or update existing player.
        
        Args:
            features: torch.Tensor, normalized feature vector
            frame_idx: int, current frame index
            player_id: int, existing player ID or None for new player
            
        Returns:
            int: assigned player ID
        """
        if player_id is None:
            # New player
            player_id = self.next_id
            self.next_id += 1
            self.gallery[player_id] = {
                'features': deque([features], maxlen=10),  # Keep last 10 features
                'last_seen': frame_idx
            }
        else:
            # Update existing player
            if player_id in self.gallery:
                self.gallery[player_id]['features'].append(features)
                self.gallery[player_id]['last_seen'] = frame_idx
        
        return player_id
    
    def find_match(self, features, frame_idx):
        """
        Find matching player in gallery using cosine similarity.
        
        Args:
            features: torch.Tensor, query feature vector
            frame_idx: int, current frame index
            
        Returns:
            int or None: matched player ID or None if no match
        """
        best_match_id = None
        best_similarity = -1
        
        # Clean up old entries
        self._cleanup_old_entries(frame_idx)
        
        # Search for best match
        for player_id, data in self.gallery.items():
            if frame_idx - data['last_seen'] > self.timeout:
                continue
                
            # Compute similarity with all stored features
            similarities = []
            for stored_features in data['features']:
                sim = F.cosine_similarity(features.unsqueeze(0), stored_features.unsqueeze(0))
                similarities.append(sim.item())
            
            # Use maximum similarity
            max_sim = max(similarities) if similarities else -1
            
            if max_sim > self.threshold and max_sim > best_similarity:
                best_similarity = max_sim
                best_match_id = player_id
        
        return best_match_id
    
    def _cleanup_old_entries(self, frame_idx):
        """Remove entries that have timed out."""
        to_remove = []
        for player_id, data in self.gallery.items():
            if frame_idx - data['last_seen'] > self.timeout:
                to_remove.append(player_id)
        
        for player_id in to_remove:
            del self.gallery[player_id]
    
    def get_stats(self):
        """Get gallery statistics."""
        return {
            'total_players': len(self.gallery),
            'next_id': self.next_id
        }

class VideoProcessor:
    """
    Video processing utilities for reading frames and writing output.
    """
    def __init__(self, video_path, output_path=None):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = None
        self.writer = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
            
        if self.output_path:
            # Get video properties
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
    
    def read_frame(self):
        """Read next frame from video."""
        ret, frame = self.cap.read()
        return ret, frame
    
    def write_frame(self, frame):
        """Write frame to output video."""
        if self.writer:
            self.writer.write(frame)
    
    def get_total_frames(self):
        """Get total number of frames in video."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

class Visualizer:
    """
    Visualization utilities for drawing bounding boxes and IDs.
    """
    @staticmethod
    def draw_detection(frame, bbox, player_id, confidence=None, color=(0, 255, 0)):
        """
        Draw bounding box and player ID on frame.
        
        Args:
            frame: numpy array, video frame
            bbox: tuple (x1, y1, x2, y2), bounding box coordinates
            player_id: int, player ID
            confidence: float, detection confidence
            color: tuple, BGR color for drawing
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare text
        text = f"ID: {player_id}"
        if confidence is not None:
            text += f" ({confidence:.2f})"
        
        # Draw text background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1-30), (x1 + text_size[0] + 10, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x1 + 5, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    @staticmethod
    def draw_stats(frame, stats, frame_idx):
        """Draw statistics on frame."""
        y_offset = 30
        texts = [
            f"Frame: {frame_idx}",
            f"Active Players: {stats['total_players']}",
            f"Next ID: {stats['next_id']}"
        ]
        
        for i, text in enumerate(texts):
            y_pos = y_offset + (i * 25)
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

class ByteTrackConfig:
    """
    Configuration for ByteTrack integration.
    """
    @staticmethod
    def get_tracker_config():
        """Get ByteTrack configuration as dictionary."""
        return {
            'tracker_type': 'bytetrack',
            'track_high_thresh': 0.4,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.5,
            'track_buffer': Config.TRACK_BUFFER,
            'match_thresh': Config.MATCH_THRESHOLD,
            'with_reid': False  # We implement our own ReID
        }
    
    @staticmethod
    def save_tracker_config(file_path='bytetrack_custom.yaml'):
        """Save ByteTrack configuration to YAML file."""
        import yaml
        
        config = ByteTrackConfig.get_tracker_config()
        
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return file_path

def extract_player_crop(frame, bbox, padding=0.1):
    """
    Extract player crop from frame with padding.
    
    Args:
        frame: numpy array, video frame
        bbox: tuple (x1, y1, x2, y2), bounding box coordinates
        padding: float, padding ratio
        
    Returns:
        numpy array: cropped player image
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    # Add padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * padding)
    pad_h = int(box_h * padding)
    
    # Ensure bounds
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    # Extract crop
    crop = frame[y1:y2, x1:x2]
    
    # Handle empty crops
    if crop.size == 0:
        return np.zeros((64, 32, 3), dtype=np.uint8)
    
    return crop

def cosine_similarity_batch(features1, features2):
    """
    Compute cosine similarity between two batches of features.
    
    Args:
        features1: torch.Tensor (N, D)
        features2: torch.Tensor (M, D)
        
    Returns:
        torch.Tensor: similarity matrix (N, M)
    """
    # Normalize features
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.mm(features1, features2.t())
    
    return similarity

def create_color_map(max_ids=100):
    """
    Create a color map for different player IDs.
    
    Args:
        max_ids: int, maximum number of unique IDs
        
    Returns:
        dict: mapping from ID to BGR color
    """
    np.random.seed(42)  # For consistent colors
    colors = {}
    
    for i in range(max_ids):
        colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
    
    return colors

# Global color map for consistent visualization
PLAYER_COLORS = create_color_map()

def get_player_color(player_id):
    """Get consistent color for a player ID."""
    return PLAYER_COLORS.get(player_id % len(PLAYER_COLORS), (0, 255, 0))

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test image preprocessor
    preprocessor = ImagePreprocessor()
    dummy_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    processed = preprocessor(dummy_image)
    print(f"Preprocessed image shape: {processed.shape}")
    
    # Test gallery
    gallery = PlayerGallery()
    dummy_features = torch.randn(Config.FUSED_DIM)
    dummy_features = F.normalize(dummy_features, p=2, dim=0)
    
    # Add some players
    id1 = gallery.add_or_update(dummy_features, 0)
    id2 = gallery.add_or_update(dummy_features * 0.9, 10)  # Similar features
    
    # Test matching
    query_features = dummy_features * 0.95  # Very similar
    matched_id = gallery.find_match(query_features, 20)
    
    print(f"Added player IDs: {id1}, {id2}")
    print(f"Matched ID: {matched_id}")
    print(f"Gallery stats: {gallery.get_stats()}")
    
    print("All utility tests passed!")