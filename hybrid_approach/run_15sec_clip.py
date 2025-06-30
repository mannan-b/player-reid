#!/usr/bin/env python3
"""
Enhanced runner script for your 15-second football clip processing
Usage: python run_15sec_clip.py
"""

import os
import sys
import torch

def check_files():
    """Check if required files exist"""
    required_files = {
        'best.pt': 'Your fine-tuned YOLOv11 model',
        '15sec_input_720p.mp4': 'Your 15-second football clip'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"❌ {file_path} - {description}")
        else:
            print(f"✅ {file_path} - Found")
    
    if missing_files:
        print("\nMissing files:")
        for missing in missing_files:
            print(missing)
        print("\nPlease ensure these files are in the current directory.")
        return False
    return True

def main():
    print("🏃‍♂️ Transformer-Based Player Re-Identification for 15-Second Clips")
    print("=" * 60)
    
    # Check dependencies
    try:
        import torch
        import cv2
        from ultralytics import YOLO
        print(f"✅ PyTorch {torch.__version__} - Ready")
        print(f"✅ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return
    
    # Check required files
    print("\nChecking required files...")
    if not check_files():
        return
    
    # Import and run
    try:
        from transformer_reid_15sec import PlayerTracker
        
        MODEL_PATH = "best.pt"
        VIDEO_PATH = "15sec_input_720p.mp4"
        OUTPUT_PATH = "output_transformer_reid.mp4"
        
        print(f"\nConfiguration:")
        print(f"📦 Model: {MODEL_PATH}")
        print(f"🎬 Input: {VIDEO_PATH}")
        print(f"💾 Output: {OUTPUT_PATH}")
        
        # Initialize tracker
        print(f"\n🚀 Initializing Transformer-based tracker...")
        tracker = PlayerTracker(
            model_path=MODEL_PATH,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            sequence_length=8
        )
        print("✅ Tracker initialized successfully")
        
        # Process video
        print(f"\n🎯 Processing your 15-second clip...")
        tracker.process_video(VIDEO_PATH, OUTPUT_PATH)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Processed video saved: {OUTPUT_PATH}")
        print(f"✅ Players maintain consistent IDs throughout the clip")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
