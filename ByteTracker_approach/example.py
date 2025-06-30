# Simple example script for Player Re-ID System
# Quick test to ensure everything works correctly

import torch
import cv2
import os
from main import PlayerReIDSystem

def test_system():
    """
    Simple test to verify the system works correctly.
    """
    print("🔥 Testing Player Re-ID System...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Initialize the system
        print("1. Initializing system...")
        system = PlayerReIDSystem()
        print("✅ System initialized successfully!")
        
        # Test with a dummy video if no real video is available
        print("2. Testing with sample data...")
        
        # Create a dummy frame for testing
        dummy_frame = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8).numpy()
        
        # Test single frame processing
        result_frame = system.process_single_frame(dummy_frame)
        print("✅ Single frame processing works!")
        
        # Check if real video exists
        test_video = "15sec_input_720p.mp4"
        if os.path.exists(test_video):
            print(f"3. Processing real video: {test_video}")
            stats = system.process_video(test_video, "test_output.mp4")
            print("✅ Video processing completed!")
            print(f"   - Processed {stats['total_frames']} frames")
            print(f"   - Found {stats['unique_players']} unique players")
            print(f"   - Processing time: {stats['processing_time']:.2f}s")
        else:
            print(f"3. No test video found ({test_video})")
            print("   Place your 15-second clip as '15sec_clip.mp4' to test with real data")
        
        print("\n🎉 All tests passed! The system is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_demo():
    """
    Quick demonstration of how to use the system.
    """
    print("\n📖 Quick Demo - How to use the system:")
    
    demo_code = '''
# Basic usage example:

from main import PlayerReIDSystem

# 1. Initialize the system with your YOLOv11 model
system = PlayerReIDSystem(yolo_model_path='best.pt')

# 2. Process your video
stats = system.process_video(
    video_path='your_15sec_clip.mp4',
    output_path='output_with_reid.mp4'
)

# 3. Check results
print(f"Identified {stats['unique_players']} unique players")
print(f"Processed {stats['total_frames']} frames")

# That's it! Players will maintain same IDs even if they 
# leave and re-enter the frame.
'''
    
    print(demo_code)

if __name__ == "__main__":
    # Run the test
    success = test_system()
    
    if success:
        # Show demo
        quick_demo()
        
        print("\n🚀 Ready to process your videos!")
        print("   Run: python main.py --video your_video.mp4")
    else:
        print("\n🔧 Please check the installation and try again.")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")