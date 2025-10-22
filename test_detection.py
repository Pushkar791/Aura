"""
Quick test script to verify GPU detection is working
"""
import cv2
import numpy as np
import time
from src.detection_engine import AURADetectionEngine

print("=" * 60)
print("AURA Detection Test - GPU Verification")
print("=" * 60)

# Initialize detection engine
print("\n1. Initializing detection engine...")
engine = AURADetectionEngine()

if not engine.initialize():
    print("❌ Failed to initialize!")
    exit(1)

print("✅ Engine initialized!")
print(f"Device: {engine.device}")

# Test with a real image from webcam
print("\n2. Testing with webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("⚠️  No webcam found, using blank image")
    # Use blank image
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Test Image", (200, 320),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
else:
    print("✅ Webcam opened")
    # Capture one frame
    ret, test_image = cap.read()
    if not ret:
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    cap.release()

# Run detection
print("\n3. Running detection test (10 frames)...")
print("-" * 60)

times = []
detection_counts = []

for i in range(10):
    start = time.time()
    detections = engine.detect(test_image)
    inference_time = (time.time() - start) * 1000
    
    times.append(inference_time)
    detection_counts.append(len(detections))
    
    print(f"Frame {i+1}: {inference_time:.2f}ms | {len(detections)} detections")
    
    # Print detected objects
    if detections:
        for det in detections:
            print(f"  → {det.class_name} ({det.confidence:.2f})")

print("-" * 60)

# Statistics
avg_time = np.mean(times)
fps = 1000 / avg_time
total_detections = sum(detection_counts)

print("\n4. Performance Summary:")
print(f"   Average Time: {avg_time:.2f} ms")
print(f"   FPS: {fps:.1f}")
print(f"   Total Detections: {total_detections}")
print(f"   Device: {engine.device}")

# Get engine stats
stats = engine.get_stats()
print(f"\n5. Engine Stats:")
print(f"   Total Frames: {stats['total_frames']}")
print(f"   Avg Inference: {stats['avg_inference_time']*1000:.2f} ms")
print(f"   FPS: {stats['fps']:.1f}")

# Shutdown
engine.shutdown()

print("\n✅ Test Complete!")
print("=" * 60)

# Performance check
if fps > 30:
    print("🚀 EXCELLENT! GPU is working perfectly (30+ FPS)")
elif fps > 20:
    print("✅ GOOD! GPU acceleration active (20+ FPS)")
elif fps > 10:
    print("⚠️  OK, but could be better (10+ FPS)")
else:
    print("❌ SLOW! Check GPU settings or use CPU")

print("\nNow run: python hackathon_demo.py")
print("=" * 60)
