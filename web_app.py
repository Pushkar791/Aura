"""
AURA Web Application - Real-Time Detection Only
Clean version with just live camera detection
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from src.detection_engine import AURADetectionEngine

app = Flask(__name__)

# Global variables
engine = None
camera = None
camera_source = 'laptop'  # 'laptop' or 'mobile'
last_mobile_frame = None
last_mobile_frame_time = 0
detection_stats = {
    'fps': 0,
    'total_detections': 0,
    'recent_detections': [],
    'frame_count': 0
}

def initialize_engine():
    """Initialize AURA detection engine"""
    global engine
    print("=" * 70)
    print("🚀 AURA Web Application - Real-Time Detection")
    print("=" * 70)
    
    print("\n📥 Loading detection engine...")
    engine = AURADetectionEngine()
    
    if not engine.initialize():
        print("❌ Failed to initialize!")
        return False
    
    print(f"✅ Engine ready! Device: {engine.device}")
    return True


def generate_frames():
    """Generate frames for video streaming"""
    global camera, detection_stats, camera_source, last_mobile_frame
    
    # Initialize laptop camera if needed
    if camera is None and camera_source == 'laptop':
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        frame = None
        
        if camera_source == 'laptop':
            if camera is None:
                time.sleep(0.1)
                continue
            success, frame = camera.read()
            if not success:
                break
        else:  # mobile
            # Use last received frame from mobile
            if last_mobile_frame is not None:
                frame = last_mobile_frame.copy()
            else:
                # Wait for first frame
                time.sleep(0.1)
                continue
        
        if frame is None:
            continue
        
        # Run detection with YOUR engine
        detections = engine.detect(frame)
        
        # Draw detections with YOUR method
        display_frame = engine.draw_detections(frame, detections)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Add info overlay
        source_text = "📱 Mobile" if camera_source == 'mobile' else "💻 Laptop"
        info_text = f"FPS: {fps:.1f} | Detections: {len(detections)} | Device: {engine.device} | Source: {source_text}"
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add detection labels
        y_offset = 65
        for detection in detections:
            label = f"{detection.class_name}: {detection.confidence:.2%}"
            cv2.putText(display_frame, label, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
        
        # Update stats
        detection_stats['fps'] = fps
        detection_stats['total_detections'] = len(detections)
        detection_stats['recent_detections'] = [
            {'class': d.class_name, 'confidence': float(d.confidence)}
            for d in detections
        ]
        detection_stats['frame_count'] = frame_count
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Main page"""
    return render_template('aura.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    """Get live statistics"""
    return jsonify({
        'fps': round(detection_stats['fps'], 1),
        'total_detections': detection_stats['total_detections'],
        'detections': detection_stats['recent_detections'],
        'frames': detection_stats['frame_count'],
        'device': engine.device if engine else 'Unknown',
        'camera_source': camera_source
    })


@app.route('/set_camera_source', methods=['POST'])
def set_camera_source():
    """Switch camera source between laptop and mobile"""
    global camera_source, camera, last_mobile_frame
    
    data = request.json
    new_source = data.get('source', 'laptop')
    
    if new_source not in ['laptop', 'mobile']:
        return jsonify({'status': 'error', 'message': 'Invalid source'})
    
    # Release laptop camera if switching to mobile
    if new_source == 'mobile' and camera is not None:
        camera.release()
        camera = None
    
    # Clear mobile frame if switching to laptop
    if new_source == 'laptop':
        last_mobile_frame = None
    
    camera_source = new_source
    return jsonify({'status': 'success', 'source': camera_source})


@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Receive frame from mobile camera"""
    global last_mobile_frame, last_mobile_frame_time
    
    try:
        data = request.json
        image_data = data.get('frame', '')
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        last_mobile_frame = frame
        last_mobile_frame_time = time.time()
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/get_camera_source')
def get_camera_source():
    """Get current camera source"""
    return jsonify({'source': camera_source})


if __name__ == '__main__':
    if not initialize_engine():
        print("❌ Failed to start!")
        exit(1)
    
    print("\n✅ Server starting...")
    print("🌐 Open: http://localhost:5000")
    print("📹 Real-time detection ready!")
    print("=" * 70)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        if camera:
            camera.release()
        if engine:
            engine.shutdown()