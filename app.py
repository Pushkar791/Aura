"""
AURA Web Application
Professional frontend for safety equipment detection
Modes: 1) Real-time Detection  2) Dataset Evaluation
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import time
import threading
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Global variables
model = None
camera = None
evaluation_results = {}
is_evaluating = False
evaluation_progress = 0
camera_source = 'laptop'  # 'laptop' or 'mobile'
last_mobile_frame = None
last_mobile_frame_time = 0

# Configuration
MODEL_PATH = "runs/safety_equipment/weights/best.pt"
TEST_DATA_PATH = r"C:\Users\RADHA SOAMI JI\Downloads\test-20251015T135512Z-1-001\test"
CONFIDENCE_THRESHOLD = 0.20

CLASS_NAMES = {
    0: 'OxygenTank',
    1: 'NitrogenTank',
    2: 'FirstAidBox',
    3: 'FireAlarm',
    4: 'SafetySwitchPanel',
    5: 'EmergencyPhone',
    6: 'FireExtinguisher'
}

CLASS_COLORS = {
    'OxygenTank': (0, 255, 255),
    'NitrogenTank': (255, 0, 255),
    'FirstAidBox': (0, 255, 0),
    'FireAlarm': (255, 0, 0),
    'SafetySwitchPanel': (255, 255, 0),
    'EmergencyPhone': (0, 165, 255),
    'FireExtinguisher': (255, 128, 0)
}


def load_model():
    """Load YOLO model"""
    global model
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded on {device}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def draw_detections(frame, results, show_confidence=True):
    """Draw bounding boxes and labels on frame"""
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return frame, []
    
    detections = []
    
    for box in results[0].boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Get class and confidence
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = CLASS_NAMES[class_id]
        
        # Get color
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{class_name}"
        if show_confidence:
            label += f" {confidence:.2%}"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Store detection info
        detections.append({
            'class': class_name,
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2]
        })
    
    return frame, detections


def generate_frames():
    """Generate frames for real-time detection"""
    global camera, camera_source, last_mobile_frame, last_mobile_frame_time
    
    # Initialize laptop camera if needed
    if camera is None and camera_source == 'laptop':
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
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
        
        # Run detection
        start_time = time.time()
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, device=0, verbose=False)
        inference_time = (time.time() - start_time) * 1000
        
        # Draw detections
        frame, detections = draw_detections(frame, results)
        
        # Add info overlay
        fps = 1000 / inference_time if inference_time > 0 else 0
        source_text = "📱 Mobile" if camera_source == 'mobile' else "💻 Laptop"
        info_text = [
            f"Source: {source_text}",
            f"FPS: {fps:.1f}",
            f"Inference: {inference_time:.1f}ms",
            f"Detections: {len(detections)}",
            f"Confidence: {CONFIDENCE_THRESHOLD:.2f}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def evaluate_dataset():
    """Evaluate model on test dataset"""
    global evaluation_results, is_evaluating, evaluation_progress
    
    is_evaluating = True
    evaluation_progress = 0
    
    images_dir = Path(TEST_DATA_PATH) / 'images'
    labels_dir = Path(TEST_DATA_PATH) / 'labels'
    
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
    total_images = len(image_files)
    
    results = {
        'total_images': 0,
        'total_predictions': 0,
        'per_class_stats': defaultdict(lambda: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'ground_truth': 0,
            'predictions': 0
        }),
        'inference_times': [],
        'confidence_scores': [],
        'processed_images': []
    }
    
    for idx, img_path in enumerate(image_files[:50]):  # Limit to 50 for demo
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        label_path = labels_dir / (img_path.stem + '.txt')
        
        # Get ground truth
        ground_truth = parse_yolo_label(label_path)
        
        # Run inference
        start_time = time.time()
        model_results = model.predict(img, conf=CONFIDENCE_THRESHOLD, device=0, verbose=False)
        inference_time = (time.time() - start_time) * 1000
        results['inference_times'].append(inference_time)
        
        # Process predictions
        predictions = []
        if model_results[0].boxes is not None:
            for box in model_results[0].boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                predictions.append({
                    'class_id': class_id,
                    'confidence': conf
                })
                results['confidence_scores'].append(conf)
        
        # Match predictions with ground truth
        matched_gt = set()
        for pred in predictions:
            class_name = CLASS_NAMES[pred['class_id']]
            # Simple matching (for demo - you can enhance this)
            if any(gt['class_id'] == pred['class_id'] for gt in ground_truth):
                results['per_class_stats'][class_name]['true_positives'] += 1
                matched_gt.add(pred['class_id'])
            else:
                results['per_class_stats'][class_name]['false_positives'] += 1
        
        # Count false negatives
        for gt in ground_truth:
            class_name = CLASS_NAMES[gt['class_id']]
            results['per_class_stats'][class_name]['ground_truth'] += 1
            if gt['class_id'] not in matched_gt:
                results['per_class_stats'][class_name]['false_negatives'] += 1
        
        for pred in predictions:
            class_name = CLASS_NAMES[pred['class_id']]
            results['per_class_stats'][class_name]['predictions'] += 1
        
        results['total_images'] += 1
        results['total_predictions'] += len(predictions)
        
        # Store sample image
        if idx < 10:  # Store first 10 images
            annotated_img, _ = draw_detections(img.copy(), model_results, show_confidence=True)
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = buffer.tobytes().hex()
            results['processed_images'].append({
                'filename': img_path.name,
                'detections': len(predictions),
                'ground_truth': len(ground_truth)
            })
        
        evaluation_progress = int((idx + 1) / total_images * 100)
    
    # Calculate metrics
    overall_tp = sum(s['true_positives'] for s in results['per_class_stats'].values())
    overall_fp = sum(s['false_positives'] for s in results['per_class_stats'].values())
    overall_fn = sum(s['false_negatives'] for s in results['per_class_stats'].values())
    
    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = overall_tp / (overall_tp + overall_fp + overall_fn) if (overall_tp + overall_fp + overall_fn) > 0 else 0
    
    results['overall_metrics'] = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'avg_inference_time': np.mean(results['inference_times']) if results['inference_times'] else 0,
        'avg_confidence': np.mean(results['confidence_scores']) if results['confidence_scores'] else 0,
        'fps': 1000 / np.mean(results['inference_times']) if results['inference_times'] else 0
    }
    
    # Calculate per-class metrics
    for class_name, stats in results['per_class_stats'].items():
        tp = stats['true_positives']
        fp = stats['false_positives']
        fn = stats['false_negatives']
        
        stats['precision'] = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        stats['recall'] = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        stats['f1_score'] = (2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall'])) if (stats['precision'] + stats['recall']) > 0 else 0
    
    evaluation_results = results
    is_evaluating = False
    evaluation_progress = 100


def parse_yolo_label(label_path):
    """Parse YOLO format label"""
    objects = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    objects.append({'class_id': int(parts[0])})
    return objects


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_evaluation', methods=['POST'])
def start_evaluation():
    """Start dataset evaluation"""
    global is_evaluating
    
    if is_evaluating:
        return jsonify({'status': 'already_running'})
    
    # Run evaluation in background thread
    thread = threading.Thread(target=evaluate_dataset)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/evaluation_status')
def evaluation_status():
    """Get evaluation progress"""
    return jsonify({
        'is_evaluating': is_evaluating,
        'progress': evaluation_progress
    })


@app.route('/evaluation_results')
def get_evaluation_results():
    """Get evaluation results"""
    if not evaluation_results:
        return jsonify({'status': 'no_results'})
    
    return jsonify({
        'status': 'complete',
        'results': {
            'overall_metrics': evaluation_results['overall_metrics'],
            'per_class_stats': dict(evaluation_results['per_class_stats']),
            'total_images': evaluation_results['total_images'],
            'total_predictions': evaluation_results['total_predictions'],
            'processed_images': evaluation_results['processed_images']
        }
    })


@app.route('/system_info')
def system_info():
    """Get system information"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    
    return jsonify({
        'device': device,
        'gpu': gpu_name,
        'model': MODEL_PATH,
        'confidence': CONFIDENCE_THRESHOLD,
        'classes': list(CLASS_NAMES.values()),
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
    print("=" * 70)
    print("🚀 AURA Web Application")
    print("=" * 70)
    
    # Load model
    if not load_model():
        print("❌ Failed to load model. Exiting.")
        exit(1)
    
    print("\n✅ Server starting...")
    print("🌐 Open browser: http://localhost:5000")
    print("=" * 70)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
