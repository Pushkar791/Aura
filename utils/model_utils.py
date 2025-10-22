"""
AURA Model Utilities
Helper functions for model management, training, and evaluation
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from loguru import logger
import json
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import config, SAFETY_CLASSES

class ModelManager:
    """Utility class for managing YOLO models"""
    
    def __init__(self):
        from configs.config import MODELS_DIR
        self.models_dir = Path(MODELS_DIR)
        self.models_dir.mkdir(exist_ok=True)
        
    def download_pretrained_model(self, model_size: str = "n") -> str:
        """Download a pretrained YOLOv8 model
        
        Args:
            model_size: Model size (n, s, m, l, x)
            
        Returns:
            Path to downloaded model
        """
        model_name = f"yolov8{model_size}.pt"
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            logger.info(f"Downloading {model_name}...")
            model = YOLO(model_name)  # This will download the model
            model.save(str(model_path))
            logger.success(f"Model saved to {model_path}")
        else:
            logger.info(f"Model {model_name} already exists")
        
        return str(model_path)
    
    def train_model(self, data_yaml: str, model_size: str = "n", epochs: int = 100, 
                   imgsz: int = 640, batch: int = 16) -> str:
        """Train a YOLO model on custom data
        
        Args:
            data_yaml: Path to dataset YAML file
            model_size: Model size (n, s, m, l, x)
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            
        Returns:
            Path to trained model
        """
        try:
            # Load pretrained model
            model_name = f"yolov8{model_size}.pt"
            model = YOLO(model_name)
            
            logger.info(f"Starting training with {model_name}")
            logger.info(f"Data: {data_yaml}")
            logger.info(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
            
            # Train the model
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=str(self.models_dir),
                name="safety_equipment_training",
                verbose=True
            )
            
            # Get the best model path
            best_model_path = results.save_dir / "weights" / "best.pt"
            
            logger.success(f"Training completed! Best model saved to {best_model_path}")
            return str(best_model_path)
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate_model(self, model_path: str, data_yaml: str) -> Dict:
        """Evaluate a trained model
        
        Args:
            model_path: Path to model file
            data_yaml: Path to dataset YAML file
            
        Returns:
            Evaluation metrics
        """
        try:
            model = YOLO(model_path)
            
            logger.info(f"Evaluating model {model_path}")
            
            # Run validation
            results = model.val(data=data_yaml, verbose=True)
            
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.p.mean()),
                'recall': float(results.box.r.mean()),
                'f1': float(results.box.f1.mean()),
            }
            
            logger.success(f"Evaluation completed: mAP@0.5 = {metrics['mAP50']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def benchmark_model(self, model_path: str, test_images: List[str] = None, 
                       num_runs: int = 100) -> Dict:
        """Benchmark model performance
        
        Args:
            model_path: Path to model file
            test_images: List of test image paths
            num_runs: Number of inference runs for benchmarking
            
        Returns:
            Performance metrics
        """
        try:
            model = YOLO(model_path)
            
            # Create test image if none provided
            if not test_images:
                test_image = np.zeros((640, 640, 3), dtype=np.uint8)
                test_images = [test_image]
            else:
                test_images = [cv2.imread(img) for img in test_images[:5]]
            
            logger.info(f"Benchmarking model {model_path}")
            
            # Warmup
            for _ in range(10):
                for img in test_images:
                    _ = model.predict(img, verbose=False)
            
            # Benchmark inference time
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                for img in test_images:
                    _ = model.predict(img, verbose=False)
                end_time = time.time()
                times.append((end_time - start_time) / len(test_images))
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time
            
            # Get model info
            model_info = model.info(verbose=False)
            
            metrics = {
                'avg_inference_time_ms': avg_time * 1000,
                'std_inference_time_ms': std_time * 1000,
                'fps': fps,
                'parameters': model_info.get('parameters', 0),
                'gflops': model_info.get('gflops', 0),
                'model_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
            }
            
            logger.success(f"Benchmark completed: {fps:.1f} FPS, {avg_time*1000:.1f}ms avg")
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {str(e)}")
            raise

def test_camera_detection(model_path: str = None, camera_index: int = 0, 
                         duration: int = 30):
    """Test detection on live camera feed
    
    Args:
        model_path: Path to YOLO model
        camera_index: Camera index to use
        duration: Test duration in seconds
    """
    try:
        # Load model
        if model_path is None:
            model_path = ModelManager().download_pretrained_model("n")
        
        model = YOLO(model_path)
        logger.info(f"Loaded model: {model_path}")
        
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        logger.info(f"Testing detection for {duration} seconds...")
        logger.info("Press 'q' to quit early, 's' to save screenshot")
        
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model.predict(frame, verbose=False)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Count detections
            if results[0].boxes is not None:
                detection_count += len(results[0].boxes)
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add text overlay
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {detection_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('AURA Detection Test', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or elapsed > duration:
                break
            elif key == ord('s'):
                filename = f"aura_test_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        logger.info(f"Test completed:")
        logger.info(f"  Duration: {elapsed:.1f}s")
        logger.info(f"  Frames: {frame_count}")
        logger.info(f"  Average FPS: {fps:.1f}")
        logger.info(f"  Total Detections: {detection_count}")
        
    except Exception as e:
        logger.error(f"Camera test failed: {str(e)}")
        raise

def create_sample_dataset(output_dir: str, num_images: int = 100):
    """Create a sample dataset for testing
    
    Args:
        output_dir: Output directory for dataset
        num_images: Number of sample images to create
    """
    try:
        output_path = Path(output_dir)
        
        # Create directory structure
        for split in ['train', 'val']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating sample dataset in {output_path}")
        
        # Generate sample images and labels
        for i in range(num_images):
            # Determine split
            split = 'train' if i < int(num_images * 0.8) else 'val'
            
            # Create random image
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Add some colored rectangles to simulate objects
            num_objects = np.random.randint(1, 4)
            labels = []
            
            for _ in range(num_objects):
                # Random class
                class_id = np.random.randint(0, len(SAFETY_CLASSES))
                
                # Random bounding box
                x_center = np.random.uniform(0.1, 0.9)
                y_center = np.random.uniform(0.1, 0.9)
                width = np.random.uniform(0.05, 0.2)
                height = np.random.uniform(0.05, 0.2)
                
                # Draw rectangle on image
                x1 = int((x_center - width/2) * 640)
                y1 = int((y_center - height/2) * 640)
                x2 = int((x_center + width/2) * 640)
                y2 = int((y_center + height/2) * 640)
                
                color = tuple(np.random.randint(100, 255, 3).tolist())
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
                
                # Add label
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save image
            image_path = output_path / split / 'images' / f"sample_{i:04d}.jpg"
            cv2.imwrite(str(image_path), image)
            
            # Save label
            label_path = output_path / split / 'labels' / f"sample_{i:04d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
        
        logger.success(f"Created {num_images} sample images in {output_path}")
        
        # Create dataset YAML
        yaml_content = f"""
# AURA Sample Dataset
path: {output_path}
train: train/images
val: val/images

nc: {len(SAFETY_CLASSES)}
names: {dict(SAFETY_CLASSES)}
"""
        
        yaml_path = output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
        
        logger.success(f"Dataset YAML created: {yaml_path}")
        
    except Exception as e:
        logger.error(f"Failed to create sample dataset: {str(e)}")
        raise

def main():
    """Main CLI for model utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA Model Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download pretrained model')
    download_parser.add_argument('--size', choices=['n', 's', 'm', 'l', 'x'], default='n',
                                help='Model size')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train custom model')
    train_parser.add_argument('--data', required=True, help='Dataset YAML file')
    train_parser.add_argument('--size', choices=['n', 's', 'm', 'l', 'x'], default='n',
                             help='Model size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', required=True, help='Model file path')
    eval_parser.add_argument('--data', required=True, help='Dataset YAML file')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark model performance')
    bench_parser.add_argument('--model', required=True, help='Model file path')
    bench_parser.add_argument('--runs', type=int, default=100, help='Number of runs')
    
    # Test camera command
    test_parser = subparsers.add_parser('test-camera', help='Test detection on camera')
    test_parser.add_argument('--model', help='Model file path')
    test_parser.add_argument('--camera', type=int, default=0, help='Camera index')
    test_parser.add_argument('--duration', type=int, default=30, help='Test duration (seconds)')
    
    # Create sample dataset command
    sample_parser = subparsers.add_parser('create-sample', help='Create sample dataset')
    sample_parser.add_argument('--output', required=True, help='Output directory')
    sample_parser.add_argument('--num-images', type=int, default=100, help='Number of images')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ModelManager()
    
    try:
        if args.command == 'download':
            model_path = manager.download_pretrained_model(args.size)
            print(f"Model downloaded: {model_path}")
            
        elif args.command == 'train':
            model_path = manager.train_model(args.data, args.size, args.epochs,
                                           args.imgsz, args.batch)
            print(f"Training completed: {model_path}")
            
        elif args.command == 'evaluate':
            metrics = manager.evaluate_model(args.model, args.data)
            print(f"Evaluation results: {json.dumps(metrics, indent=2)}")
            
        elif args.command == 'benchmark':
            metrics = manager.benchmark_model(args.model, num_runs=args.runs)
            print(f"Benchmark results: {json.dumps(metrics, indent=2)}")
            
        elif args.command == 'test-camera':
            test_camera_detection(args.model, args.camera, args.duration)
            
        elif args.command == 'create-sample':
            create_sample_dataset(args.output, args.num_images)
            
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())