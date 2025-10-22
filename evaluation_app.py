"""
AURA Evaluation Dashboard
Beautiful web interface for dataset evaluation
"""

from flask import Flask, render_template, jsonify
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import threading
import time

app = Flask(__name__)

# Configuration
MODEL_PATH = r"runs\safety_equipment\weights\best.pt"
TEST_DATA_PATH = r"C:\Users\RADHA SOAMI JI\Downloads\test-20251015T135512Z-1-001\test"

# Global state
evaluation_state = {
    'is_running': False,
    'progress': 0,
    'current_image': '',
    'results': None,
    'error': None
}

class_names = {
    0: 'OxygenTank',
    1: 'NitrogenTank',
    2: 'FirstAidBox',
    3: 'FireAlarm',
    4: 'SafetySwitchPanel',
    5: 'EmergencyPhone',
    6: 'FireExtinguisher'
}


def parse_yolo_label(label_path):
    """Parse YOLO format label"""
    objects = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    objects.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height]
                    })
    return objects


def calculate_iou(box1, box2, img_width, img_height):
    """Calculate Intersection over Union"""
    def yolo_to_corners(bbox, w, h):
        x_center, y_center, width, height = bbox
        x1 = (x_center - width/2) * w
        y1 = (y_center - height/2) * h
        x2 = (x_center + width/2) * w
        y2 = (y_center + height/2) * h
        return [x1, y1, x2, y2]
    
    b1 = yolo_to_corners(box1, img_width, img_height)
    b2 = yolo_to_corners(box2, img_width, img_height)
    
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def run_evaluation():
    """Run the evaluation in background"""
    global evaluation_state
    
    try:
        evaluation_state['is_running'] = True
        evaluation_state['progress'] = 0
        evaluation_state['error'] = None
        
        # Check paths
        model_path = Path(MODEL_PATH)
        test_path = Path(TEST_DATA_PATH)
        
        if not model_path.exists():
            evaluation_state['error'] = f"Model not found: {MODEL_PATH}"
            evaluation_state['is_running'] = False
            return
        
        if not test_path.exists():
            evaluation_state['error'] = f"Test data not found: {TEST_DATA_PATH}"
            evaluation_state['is_running'] = False
            return
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(str(model_path))
        
        images_dir = test_path / 'images'
        labels_dir = test_path / 'labels'
        
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
        total_images = len(image_files)
        
        if total_images == 0:
            evaluation_state['error'] = "No images found in test dataset"
            evaluation_state['is_running'] = False
            return
        
        # Initialize results
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
            'confidence_scores': []
        }
        
        CONF_THRESHOLD = 0.25
        IOU_THRESHOLD = 0.5
        
        # Evaluate each image
        for idx, img_path in enumerate(image_files):
            evaluation_state['current_image'] = img_path.name
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            label_path = labels_dir / (img_path.stem + '.txt')
            ground_truth = parse_yolo_label(label_path)
            
            # Run detection
            start_time = time.time()
            detections = model.predict(
                source=img,
                conf=CONF_THRESHOLD,
                device=device,
                verbose=False
            )[0]
            inference_time = (time.time() - start_time) * 1000
            results['inference_times'].append(inference_time)
            
            # Extract predictions
            predictions = []
            if detections.boxes is not None and len(detections.boxes) > 0:
                for box in detections.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x_center = ((xyxy[0] + xyxy[2]) / 2) / img_width
                    y_center = ((xyxy[1] + xyxy[3]) / 2) / img_height
                    width = (xyxy[2] - xyxy[0]) / img_width
                    height = (xyxy[3] - xyxy[1]) / img_height
                    
                    predictions.append({
                        'class_id': class_id,
                        'confidence': conf,
                        'bbox': [x_center, y_center, width, height]
                    })
                    results['confidence_scores'].append(conf)
            
            # Match predictions with ground truth
            matched_gt = set()
            matched_pred = set()
            
            for pred_idx, pred in enumerate(predictions):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(ground_truth):
                    if gt_idx in matched_gt:
                        continue
                    
                    if pred['class_id'] == gt['class_id']:
                        iou = calculate_iou(pred['bbox'], gt['bbox'], img_width, img_height)
                        if iou > best_iou and iou >= IOU_THRESHOLD:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                class_name = class_names[pred['class_id']]
                if best_gt_idx >= 0:
                    results['per_class_stats'][class_name]['true_positives'] += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                else:
                    results['per_class_stats'][class_name]['false_positives'] += 1
            
            # Count False Negatives
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx not in matched_gt:
                    class_name = class_names[gt['class_id']]
                    results['per_class_stats'][class_name]['false_negatives'] += 1
            
            # Update counts
            for pred in predictions:
                class_name = class_names[pred['class_id']]
                results['per_class_stats'][class_name]['predictions'] += 1
            
            for gt in ground_truth:
                class_name = class_names[gt['class_id']]
                results['per_class_stats'][class_name]['ground_truth'] += 1
            
            results['total_images'] += 1
            results['total_predictions'] += len(predictions)
            evaluation_state['progress'] = int((idx + 1) / total_images * 100)
        
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
        
        evaluation_state['results'] = results
        evaluation_state['progress'] = 100
        
    except Exception as e:
        evaluation_state['error'] = str(e)
    finally:
        evaluation_state['is_running'] = False


@app.route('/')
def index():
    """Main page"""
    return render_template('evaluation.html')


@app.route('/start_evaluation', methods=['POST'])
def start_evaluation():
    """Start evaluation"""
    global evaluation_state
    
    if evaluation_state['is_running']:
        return jsonify({'status': 'already_running'})
    
    # Reset state
    evaluation_state = {
        'is_running': True,
        'progress': 0,
        'current_image': '',
        'results': None,
        'error': None
    }
    
    # Start evaluation in background thread
    thread = threading.Thread(target=run_evaluation)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/evaluation_status')
def get_status():
    """Get evaluation status"""
    return jsonify({
        'is_running': evaluation_state['is_running'],
        'progress': evaluation_state['progress'],
        'current_image': evaluation_state['current_image'],
        'error': evaluation_state['error']
    })


@app.route('/evaluation_results')
def get_results():
    """Get evaluation results"""
    if evaluation_state['results'] is None:
        return jsonify({'status': 'no_results'})
    
    return jsonify({
        'status': 'complete',
        'results': {
            'overall_metrics': evaluation_state['results']['overall_metrics'],
            'per_class_stats': dict(evaluation_state['results']['per_class_stats']),
            'total_images': evaluation_state['results']['total_images'],
            'total_predictions': evaluation_state['results']['total_predictions']
        }
    })


if __name__ == '__main__':
    print("=" * 70)
    print("📊 AURA Evaluation Dashboard")
    print("=" * 70)
    print(f"\n📁 Model: {MODEL_PATH}")
    print(f"📁 Test Data: {TEST_DATA_PATH}")
    print("\n✅ Server starting...")
    print("🌐 Open: http://localhost:5001")
    print("=" * 70)
    
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
