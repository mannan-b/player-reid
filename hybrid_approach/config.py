# Configuration for Transformer-Based Player Re-Identification
# Optimized for 15-second football clips
import gdown
url = f"https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD"

output = "best.pt"
gdown.download(url, output, quiet=False)
# Model Configuration
MODEL_CONFIG = {
    
    # Your fine-tuned YOLOv11 model path
    'yolo_model_path': 'best.pt',
    
    # Transformer AutoEncoder settings
    'transformer': {
        'input_dim': 512,        # CNN feature dimension
        'hidden_dim': 256,       # Transformer hidden dimension
        'num_heads': 8,          # Multi-head attention heads
        'num_layers': 4,         # Transformer layers
        'dropout': 0.1,          # Dropout rate
        'sequence_length': 8,    # Frames in sequence (optimal for 15-sec clips)
    },
    
    # Detection settings
    'detection': {
        'conf_threshold': 0.3,   # Confidence threshold for detections
        'input_size': (128, 256), # Standard person crop size
    },
    
    # Re-identification settings
    'reid': {
        'similarity_threshold': 0.65,  # Threshold for matching players
        'timeout_frames': 450,         # 15 seconds at 30fps
        'transformer_weight': 0.7,     # Weight for transformer features
        'color_weight': 0.3,           # Weight for color features
    },
    
    # Video settings optimized for 15-second clips
    'video': {
        'max_frames': 450,       # Maximum frames in 15 seconds at 30fps
        'progress_interval': 30,  # Show progress every 30 frames (1 second)
    }
}

# File paths for your specific setup
PATHS = {
    'model': 'best.pt',                    # Your fine-tuned YOLOv11 model
    'input_video': '15sec_input_720p.mp4', # Your 15-second clip
    'output_video': 'output_transformer_reid.mp4', # Output with consistent IDs
    'weights_dir': 'weights/',             # Directory for model weights
    'temp_dir': 'temp/',                   # Temporary files directory
}

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Color histogram configuration
COLOR_CONFIG = {
    'bins': 8,           # Number of bins for color histogram
    'color_space': 'HSV', # Color space for better color representation
}