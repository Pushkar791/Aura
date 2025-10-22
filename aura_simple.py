"""
Simple AURA Live Detection
Real-time camera detection with GPU acceleration
"""

import cv2
import time
from src.detection_engine import AURADetectionEngine

print("=" * 70)
print("🚀 AURA Live Detection")
print("=" * 70)

# Initialize detection engine
print("\n📥 Loading detection engine...")
engine = AURADetectionEngine()

if not engine.initialize():
    print("❌ Failed to initialize!")
    exit(1)

print(f"✅ Engine ready! Device: {engine.device}")

# Open camera
print("\n📷 Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✅ Camera opened!")
print("\n🎬 Starting live detection...")
print("Press 'q' to quit, 's' for screenshot")
print("=" * 70)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break
    
    # Run detection
    detections = engine.detect(frame)
    
    # Draw detections
    display_frame = engine.draw_detections(frame, detections)
    
    # Calculate FPS
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    # Add info overlay
    info_text = f"FPS: {fps:.1f} | Detections: {len(detections)} | Device: {engine.device}"
    cv2.putText(display_frame, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add detection labels
    y_offset = 60
    for detection in detections:
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        cv2.putText(display_frame, label, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
    
    # Show frame
    cv2.imshow('AURA Live Detection', display_frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"aura_screenshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, display_frame)
        print(f"📸 Screenshot saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
engine.shutdown()

print("\n" + "=" * 70)
print("👋 AURA stopped")
print(f"📊 Stats: {frame_count} frames, {fps:.1f} FPS average")
print("=" * 70)
