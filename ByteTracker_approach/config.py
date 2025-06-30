# Configuration file for Player Re-Identification using Body Part Appearances
# Based on the research paper: 2310.14469v1.pdf

import torch

class Config:
    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    import gdown
    url = f"https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD"

    output = "best.pt"
    gdown.download(url, output, quiet=False)
    
    # Model paths
    YOLO_MODEL_PATH = 'best.pt'  # Your fine-tuned YOLOv11 model
    RESNET_WEIGHTS = 'IMAGENET1K_V2'  # Pre-trained ResNet50 weights
    
    # Image preprocessing
    IMG_HEIGHT = 256
    IMG_WIDTH = 128
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Feature dimensions
    APPEARANCE_DIM = 512  # ResNet50 feature dimension
    POSE_DIM = 128       # Compressed pose feature dimension
    FUSED_DIM = 8000     # Compact bilinear pooling output dimension
    
    # Re-ID parameters
    COSINE_THRESHOLD = 0.65  # Similarity threshold for matching
    GALLERY_TIMEOUT = 450    # Frames to keep features (15s at 30fps)
    
    # Training parameters (if needed)
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    TRIPLET_MARGIN = 0.3
    
    # Tracking parameters
    TRACK_BUFFER = 300
    MATCH_THRESHOLD = 0.7
    
    # Video parameters
    VIDEO_PATH = '15sec_input_720p.mp4'
    OUTPUT_PATH = 'output_with_reid.mp4'
    FPS = 30