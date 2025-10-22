"""
AURA Model Testing and Evaluation Script
Tests the trained model on the test dataset and provides detailed accuracy metrics
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model_path, test_data_path):
        """Initialize the evaluator"""
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("=" * 70)
        print("🔍 AURA Model Evaluation System")
        print("=" * 70)
        print(f"\n📁 Model: {model_path}")
        print(f"📁 Test Data: {test_data_path}")
        print(f"🖥️  Device: {self.device}")
        
        # Load model
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
            
        # Class names
        self.class_names = {
            0: 'OxygenTank',
            1: 'NitrogenTank',
            2: 'FirstAidBox',
            3: 'FireAlarm',
            4: 'SafetySwitchPanel',
            5: 'EmergencyPhone',
            6: 'FireExtinguisher'
        }
        
        self.results = {
            'total_images': 0,
            'total_predictions': 0,
            'per_class_stats': defaultdict(lambda: {
                'predictions': 0,
                'ground_truth': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }),
            'confidence_scores': [],
            'inference_times': []
        }
    
    def parse_yolo_label(self, label_path):
        """Parse YOLO format label file"""
        objects = []
        if os.path.exists(label_path):
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
    
    def calculate_iou(self, box1, box2, img_width, img_height):
        """Calculate Intersection over Union between two boxes"""
        # Convert from YOLO format (normalized) to pixel coordinates
        def yolo_to_corners(bbox, w, h):
            x_center, y_center, width, height = bbox
            x1 = (x_center - width/2) * w
            y1 = (y_center - height/2) * h
            x2 = (x_center + width/2) * w
            y2 = (y_center + height/2) * h
            return [x1, y1, x2, y2]
        
        b1 = yolo_to_corners(box1, img_width, img_height)
        b2 = yolo_to_corners(box2, img_width, img_height)
        
        # Calculate intersection
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_image(self, image_path, label_path, conf_threshold=0.25, iou_threshold=0.5):
        """Evaluate a single image"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        img_height, img_width = img.shape[:2]
        
        # Get ground truth
        ground_truth = self.parse_yolo_label(label_path)
        
        # Perform inference
        start_time = cv2.getTickCount()
        results = self.model.predict(
            source=img,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )[0]
        inference_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
        
        self.results['inference_times'].append(inference_time)
        
        # Extract predictions
        predictions = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                # Get normalized coordinates
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
                self.results['confidence_scores'].append(conf)
        
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
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'], img_width, img_height)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            class_name = self.class_names[pred['class_id']]
            if best_gt_idx >= 0:
                # True Positive
                self.results['per_class_stats'][class_name]['true_positives'] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
            else:
                # False Positive
                self.results['per_class_stats'][class_name]['false_positives'] += 1
        
        # Count False Negatives (unmatched ground truth)
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx not in matched_gt:
                class_name = self.class_names[gt['class_id']]
                self.results['per_class_stats'][class_name]['false_negatives'] += 1
        
        # Update counts
        for pred in predictions:
            class_name = self.class_names[pred['class_id']]
            self.results['per_class_stats'][class_name]['predictions'] += 1
        
        for gt in ground_truth:
            class_name = self.class_names[gt['class_id']]
            self.results['per_class_stats'][class_name]['ground_truth'] += 1
        
        return {
            'predictions': len(predictions),
            'ground_truth': len(ground_truth),
            'inference_time': inference_time
        }
    
    def calculate_metrics(self):
        """Calculate precision, recall, F1 for each class"""
        print("\n" + "=" * 70)
        print("📊 Calculating Metrics")
        print("=" * 70)
        
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        
        for class_name, stats in self.results['per_class_stats'].items():
            tp = stats['true_positives']
            fp = stats['false_positives']
            fn = stats['false_negatives']
            
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            stats['precision'] = precision
            stats['recall'] = recall
            stats['f1_score'] = f1
        
        # Overall metrics
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        self.results['overall_metrics'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'accuracy': overall_tp / (overall_tp + overall_fp + overall_fn) if (overall_tp + overall_fp + overall_fn) > 0 else 0.0
        }
    
    def run_evaluation(self, conf_threshold=0.25, iou_threshold=0.5):
        """Run evaluation on test dataset"""
        print("\n" + "=" * 70)
        print("🚀 Starting Evaluation")
        print("=" * 70)
        print(f"📊 Confidence Threshold: {conf_threshold}")
        print(f"📊 IoU Threshold: {iou_threshold}")
        
        images_dir = Path(self.test_data_path) / 'images'
        labels_dir = Path(self.test_data_path) / 'labels'
        
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return
        
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg'))
        
        if len(image_files) == 0:
            print(f"❌ No images found in {images_dir}")
            return
        
        print(f"📸 Found {len(image_files)} test images")
        print("\nProcessing images...")
        
        for idx, img_path in enumerate(image_files, 1):
            label_path = labels_dir / (img_path.stem + '.txt')
            
            result = self.evaluate_image(img_path, label_path, conf_threshold, iou_threshold)
            
            if result:
                self.results['total_images'] += 1
                self.results['total_predictions'] += result['predictions']
                
                if idx % 10 == 0:
                    print(f"  Processed: {idx}/{len(image_files)} images")
        
        print(f"\n✅ Completed: {self.results['total_images']} images processed")
        
        # Calculate metrics
        self.calculate_metrics()
    
    def print_results(self):
        """Print detailed results"""
        print("\n" + "=" * 70)
        print("📈 EVALUATION RESULTS")
        print("=" * 70)
        
        # Overall statistics
        print("\n📊 Overall Statistics:")
        print(f"  Total Images: {self.results['total_images']}")
        print(f"  Total Predictions: {self.results['total_predictions']}")
        
        if self.results['confidence_scores']:
            print(f"  Average Confidence: {np.mean(self.results['confidence_scores']):.2%}")
            print(f"  Min Confidence: {np.min(self.results['confidence_scores']):.2%}")
            print(f"  Max Confidence: {np.max(self.results['confidence_scores']):.2%}")
        
        if self.results['inference_times']:
            print(f"  Average Inference Time: {np.mean(self.results['inference_times']):.2f} ms")
            print(f"  FPS: {1000/np.mean(self.results['inference_times']):.1f}")
        
        # Overall metrics
        print("\n🎯 Overall Performance:")
        overall = self.results['overall_metrics']
        print(f"  Accuracy: {overall['accuracy']:.2%}")
        print(f"  Precision: {overall['precision']:.2%}")
        print(f"  Recall: {overall['recall']:.2%}")
        print(f"  F1-Score: {overall['f1_score']:.2%}")
        
        # Per-class metrics
        print("\n📋 Per-Class Performance:")
        print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'GT':<8} {'Pred':<8}")
        print("-" * 85)
        
        for class_name in sorted(self.results['per_class_stats'].keys()):
            stats = self.results['per_class_stats'][class_name]
            print(f"{class_name:<25} {stats['precision']:<12.2%} {stats['recall']:<12.2%} "
                  f"{stats['f1_score']:<12.2%} {stats['ground_truth']:<8} {stats['predictions']:<8}")
        
        print("\n" + "=" * 70)
        print("✨ Evaluation Complete!")
        print("=" * 70)
    
    def save_results(self, output_path='evaluation_results.json'):
        """Save results to JSON file"""
        output_data = {
            'model_path': str(self.model_path),
            'test_data_path': str(self.test_data_path),
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'results': {
                'total_images': self.results['total_images'],
                'total_predictions': self.results['total_predictions'],
                'overall_metrics': self.results['overall_metrics'],
                'per_class_stats': dict(self.results['per_class_stats']),
                'statistics': {
                    'avg_confidence': float(np.mean(self.results['confidence_scores'])) if self.results['confidence_scores'] else 0,
                    'avg_inference_time_ms': float(np.mean(self.results['inference_times'])) if self.results['inference_times'] else 0,
                    'fps': float(1000/np.mean(self.results['inference_times'])) if self.results['inference_times'] else 0
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    # Paths
    model_path = "runs/safety_equipment/weights/best.pt"
    test_data_path = r"C:\Users\RADHA SOAMI JI\Downloads\test-20251015T135512Z-1-001\test"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please ensure the model is trained and saved.")
        sys.exit(1)
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, test_data_path)
    
    # Run evaluation with different confidence thresholds
    print("\n🔍 Testing with confidence threshold: 0.25")
    evaluator.run_evaluation(conf_threshold=0.25, iou_threshold=0.5)
    
    # Print results
    evaluator.print_results()
    
    # Save results
    evaluator.save_results('test_evaluation_results.json')
    
    print("\n💡 Tip: Adjust confidence threshold in the script for better precision/recall trade-off")


if __name__ == "__main__":
    main()
