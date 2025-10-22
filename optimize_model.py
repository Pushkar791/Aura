"""
AURA Model Optimization Script
Tests different confidence thresholds to find the optimal configuration
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
import matplotlib.pyplot as plt

class ModelOptimizer:
    def __init__(self, model_path, test_data_path):
        """Initialize the optimizer"""
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("=" * 70)
        print("⚡ AURA Model Optimization System")
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
            
        self.class_names = {
            0: 'OxygenTank',
            1: 'NitrogenTank',
            2: 'FirstAidBox',
            3: 'FireAlarm',
            4: 'SafetySwitchPanel',
            5: 'EmergencyPhone',
            6: 'FireExtinguisher'
        }
        
        self.optimization_results = []
    
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
    
    def evaluate_with_threshold(self, conf_threshold, iou_threshold=0.5):
        """Evaluate model with specific confidence threshold"""
        results = {
            'total_images': 0,
            'total_predictions': 0,
            'per_class_stats': defaultdict(lambda: {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
            }),
            'confidence_scores': [],
            'inference_times': []
        }
        
        images_dir = Path(self.test_data_path) / 'images'
        labels_dir = Path(self.test_data_path) / 'labels'
        
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg'))
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            label_path = labels_dir / (img_path.stem + '.txt')
            ground_truth = self.parse_yolo_label(label_path)
            
            # Inference
            start_time = cv2.getTickCount()
            model_results = self.model.predict(
                source=img,
                conf=conf_threshold,
                device=self.device,
                verbose=False
            )[0]
            inference_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
            results['inference_times'].append(inference_time)
            
            # Extract predictions
            predictions = []
            if model_results.boxes is not None and len(model_results.boxes) > 0:
                for box in model_results.boxes:
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
            
            for pred in predictions:
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
                    results['per_class_stats'][class_name]['true_positives'] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    results['per_class_stats'][class_name]['false_positives'] += 1
            
            # Count false negatives
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx not in matched_gt:
                    class_name = self.class_names[gt['class_id']]
                    results['per_class_stats'][class_name]['false_negatives'] += 1
            
            results['total_images'] += 1
            results['total_predictions'] += len(predictions)
        
        # Calculate metrics
        overall_tp = sum(stats['true_positives'] for stats in results['per_class_stats'].values())
        overall_fp = sum(stats['false_positives'] for stats in results['per_class_stats'].values())
        overall_fn = sum(stats['false_negatives'] for stats in results['per_class_stats'].values())
        
        precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
        recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = overall_tp / (overall_tp + overall_fp + overall_fn) if (overall_tp + overall_fp + overall_fn) > 0 else 0.0
        
        return {
            'conf_threshold': conf_threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'total_predictions': results['total_predictions'],
            'avg_confidence': np.mean(results['confidence_scores']) if results['confidence_scores'] else 0,
            'avg_inference_time': np.mean(results['inference_times']) if results['inference_times'] else 0,
            'fps': 1000 / np.mean(results['inference_times']) if results['inference_times'] else 0
        }
    
    def optimize(self, thresholds=None):
        """Test multiple confidence thresholds"""
        if thresholds is None:
            thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        
        print("\n" + "=" * 70)
        print("🔍 Testing Multiple Confidence Thresholds")
        print("=" * 70)
        print(f"📊 Testing {len(thresholds)} different thresholds\n")
        
        for idx, threshold in enumerate(thresholds, 1):
            print(f"Testing threshold {idx}/{len(thresholds)}: {threshold:.2f}...", end=" ")
            result = self.evaluate_with_threshold(threshold)
            self.optimization_results.append(result)
            print(f"✓ Accuracy: {result['accuracy']:.2%}, F1: {result['f1_score']:.2%}")
        
        print("\n✅ Optimization complete!")
    
    def print_results(self):
        """Print optimization results"""
        print("\n" + "=" * 70)
        print("📊 OPTIMIZATION RESULTS")
        print("=" * 70)
        
        # Sort by accuracy
        sorted_results = sorted(self.optimization_results, key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'FPS':<8}")
        print("-" * 80)
        
        for result in sorted_results:
            print(f"{result['conf_threshold']:<12.2f} {result['accuracy']:<12.2%} "
                  f"{result['precision']:<12.2%} {result['recall']:<12.2%} "
                  f"{result['f1_score']:<12.2%} {result['fps']:<8.1f}")
        
        # Best configurations
        best_accuracy = sorted_results[0]
        best_f1 = max(self.optimization_results, key=lambda x: x['f1_score'])
        best_precision = max(self.optimization_results, key=lambda x: x['precision'])
        
        print("\n" + "=" * 70)
        print("🏆 BEST CONFIGURATIONS")
        print("=" * 70)
        
        print(f"\n🎯 Best Accuracy: {best_accuracy['accuracy']:.2%}")
        print(f"   Confidence Threshold: {best_accuracy['conf_threshold']:.2f}")
        print(f"   Precision: {best_accuracy['precision']:.2%}")
        print(f"   Recall: {best_accuracy['recall']:.2%}")
        print(f"   F1-Score: {best_accuracy['f1_score']:.2%}")
        print(f"   FPS: {best_accuracy['fps']:.1f}")
        
        print(f"\n🎯 Best F1-Score: {best_f1['f1_score']:.2%}")
        print(f"   Confidence Threshold: {best_f1['conf_threshold']:.2f}")
        print(f"   Accuracy: {best_f1['accuracy']:.2%}")
        print(f"   Precision: {best_f1['precision']:.2%}")
        print(f"   Recall: {best_f1['recall']:.2%}")
        
        print(f"\n🎯 Best Precision: {best_precision['precision']:.2%}")
        print(f"   Confidence Threshold: {best_precision['conf_threshold']:.2f}")
        print(f"   Accuracy: {best_precision['accuracy']:.2%}")
        print(f"   Recall: {best_precision['recall']:.2%}")
        print(f"   F1-Score: {best_precision['f1_score']:.2%}")
        
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATION")
        print("=" * 70)
        print(f"✨ For best overall accuracy, use confidence threshold: {best_accuracy['conf_threshold']:.2f}")
        print(f"   This gives {best_accuracy['accuracy']:.2%} accuracy with {best_accuracy['fps']:.1f} FPS")
        print("=" * 70)
    
    def save_results(self, output_path='optimization_results.json'):
        """Save optimization results"""
        output_data = {
            'model_path': str(self.model_path),
            'test_data_path': str(self.test_data_path),
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'optimization_results': self.optimization_results,
            'best_accuracy': max(self.optimization_results, key=lambda x: x['accuracy']),
            'best_f1': max(self.optimization_results, key=lambda x: x['f1_score']),
            'best_precision': max(self.optimization_results, key=lambda x: x['precision'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    def plot_results(self, output_path='optimization_plot.png'):
        """Create visualization of optimization results"""
        if not self.optimization_results:
            return
        
        thresholds = [r['conf_threshold'] for r in self.optimization_results]
        accuracy = [r['accuracy'] * 100 for r in self.optimization_results]
        precision = [r['precision'] * 100 for r in self.optimization_results]
        recall = [r['recall'] * 100 for r in self.optimization_results]
        f1 = [r['f1_score'] * 100 for r in self.optimization_results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, accuracy, 'o-', label='Accuracy', linewidth=2, markersize=8)
        plt.plot(thresholds, precision, 's-', label='Precision', linewidth=2, markersize=8)
        plt.plot(thresholds, recall, '^-', label='Recall', linewidth=2, markersize=8)
        plt.plot(thresholds, f1, 'd-', label='F1-Score', linewidth=2, markersize=8)
        
        plt.xlabel('Confidence Threshold', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.title('Model Performance vs Confidence Threshold', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_path}")


def main():
    model_path = "runs/safety_equipment/weights/best.pt"
    test_data_path = r"C:\Users\RADHA SOAMI JI\Downloads\test-20251015T135512Z-1-001\test"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        sys.exit(1)
    
    # Create optimizer
    optimizer = ModelOptimizer(model_path, test_data_path)
    
    # Run optimization
    optimizer.optimize()
    
    # Print results
    optimizer.print_results()
    
    # Save results
    optimizer.save_results('optimization_results.json')
    
    # Create plot
    try:
        optimizer.plot_results('optimization_plot.png')
    except Exception as e:
        print(f"⚠️  Could not create plot: {e}")


if __name__ == "__main__":
    main()
