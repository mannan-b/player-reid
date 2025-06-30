#!/usr/bin/env python3
"""
Quick test script to validate your setup before processing the 15-second clip
Run this first to catch any issues early
"""

import torch
import cv2
import numpy as np
import os

def test_pytorch():
    """Test PyTorch installation and CUDA availability"""
    print("🔍 Testing PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
        
        # Test basic tensor operations
        x = torch.randn(10, 10)
        if torch.cuda.is_available():
            x = x.cuda()
            y = torch.mm(x, x.t())
            print("✅ Basic GPU operations working")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_ultralytics():
    """Test Ultralytics installation"""
    print("\n🔍 Testing Ultralytics...")
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
        
        # Test with a small pretrained model if available
        if os.path.exists('best.pt'):
            print("✅ Found your fine-tuned model: best.pt")
            try:
                model = YOLO('best.pt')
                print("✅ Your fine-tuned model loads successfully")
            except Exception as e:
                print(f"⚠️  Issue loading your model: {e}")
        else:
            print("ℹ️  Your fine-tuned model 'best.pt' not found in current directory")
        
        return True
    except Exception as e:
        print(f"❌ Ultralytics test failed: {e}")
        return False

def test_opencv():
    """Test OpenCV installation"""
    print("\n🔍 Testing OpenCV...")
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} installed")
        
        # Test video processing capabilities
        if os.path.exists('15sec_input_720p.mp4'):
            print("✅ Found your input video: 15sec_input_720p.mp4")
            try:
                cap = cv2.VideoCapture('15sec_input_720p.mp4')
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps
                    print(f"✅ Video info: {w}x{h}, {fps:.1f}fps, {duration:.1f}s, {total_frames} frames")
                    
                    if abs(duration - 15.0) < 1.0:
                        print("✅ Video duration is approximately 15 seconds")
                    else:
                        print(f"⚠️  Video is {duration:.1f}s, not 15s - but should still work")
                else:
                    print("❌ Could not read video frames")
                cap.release()
            except Exception as e:
                print(f"⚠️  Issue reading video: {e}")
        else:
            print("ℹ️  Your input video '15sec_input_720p.mp4' not found in current directory")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_transformer_model():
    """Test Transformer AutoEncoder implementation"""
    print("\n🔍 Testing Transformer AutoEncoder...")
    try:
        from transformer_reid_15sec import TransformerAutoEncoder, PositionalEncoding
        
        # Test positional encoding
        pos_enc = PositionalEncoding(256)
        test_input = torch.randn(8, 1, 256)  # seq_len, batch_size, hidden_dim
        encoded = pos_enc(test_input)
        print("✅ Positional encoding working")
        
        # Test transformer autoencoder
        transformer = TransformerAutoEncoder()
        if torch.cuda.is_available():
            transformer = transformer.cuda()
            test_input = test_input.cuda()
        
        with torch.no_grad():
            latent, reconstructed = transformer(test_input, return_latent=True)
        
        print(f"✅ Transformer AutoEncoder working")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Latent shape: {latent.shape}")
        print(f"   Output dim: {latent.shape[-1]} (128D as expected)")
        
        return True
    except Exception as e:
        print(f"❌ Transformer test failed: {e}")
        return False

def test_memory_requirements():
    """Test memory requirements"""
    print("\n🔍 Testing memory requirements...")
    try:
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Get initial memory
            initial_memory = torch.cuda.memory_allocated()
            
            # Simulate model loading
            from transformer_reid_15sec import PlayerTracker
            
            # Check memory after model loading
            current_memory = torch.cuda.memory_allocated()
            memory_used = (current_memory - initial_memory) / 1e9
            
            print(f"✅ Estimated GPU memory usage: {memory_used:.2f} GB")
            
            if memory_used < 2.0:
                print("✅ Memory usage is reasonable (<2GB)")
            else:
                print("⚠️  High memory usage - may need to reduce batch size")
        else:
            print("ℹ️  Using CPU - memory usage will be in RAM instead of GPU")
        
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Transformer-Based Player Re-ID Setup")
    print("=" * 50)
    
    all_tests = [
        test_pytorch,
        test_ultralytics,
        test_opencv,
        test_transformer_model,
        test_memory_requirements
    ]
    
    passed = 0
    for test in all_tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(all_tests)} passed")
    
    if passed == len(all_tests):
        print("\n🎉 All tests passed! You're ready to process your 15-second clip.")
        print("Run: python run_15sec_clip.py")
    else:
        print(f"\n⚠️  {len(all_tests) - passed} test(s) failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("• pip install -r requirements.txt")
        print("• Ensure best.pt and 15sec_input_720p.mp4 are in current directory")
        print("• Check CUDA installation if using GPU")

if __name__ == "__main__":
    main()