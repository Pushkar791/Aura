"""
AURA Detection Engine - CUDA Optimized Version
Enhanced performance with GPU acceleration and optimizations
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import time
import threading
from dataclasses import dataclass
from loguru import logger
import os
from pathlib import Path

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import config, SAFETY_CLASSES

@dataclass
class Detection:
    """Detection result data structure"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    priority: str
    timestamp: float

class AURADetectionEngineOptimized:
    """
    CUDA-Optimized AURA Detection Engine
    High-performance GPU-accelerated detection system
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the optimized detection engine
        
        Args:
            model_path: Path to custom trained model
            device: Force specific device ('cuda', 'cpu', 'cuda:0', etc.)
        """
        self.config = config
        self.model_path = model_path or self.config.model.model_name
        self.model = None
        self.device = device
        self.is_initialized = False
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_inference_time': 0.0,
            'fps': 0.0,
            'start_time': time.time(),
            'gpu_memory_used': 0,
            'gpu_memory_total': 0
        }
        
        # Detection history
        self.detection_history = []
        self.max_history = self.config.dashboard.history_length
        self.lock = threading.Lock()
        
        logger.info("CUDA-Optimized Detection Engine initialized")
    
    def initialize(self) -> bool:
        """
        Initialize YOLO model with CUDA optimizations
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info(f"Loading model: {self.model_path}")
            
            # Device selection with fallback
            if self.device is None:
                if torch.cuda.is_available():
                    self.device = 'cuda:0'
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.success(f"🚀 CUDA enabled - GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                else:
                    self.device = 'cpu'
                    logger.warning("⚠️  CUDA not available - using CPU (slower)")
            
            # Load YOLO model
            self.model = YOLO(self.model_path)
            
            # CUDA optimizations
            if 'cuda' in self.device:
                self._apply_cuda_optimizations()
            
            # Move model to device
            if hasattr(self.model, 'model'):
                self.model.model.to(self.device)
            
            # Warmup inference (important for accurate timing)
            logger.info("Warming up model...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(3):  # Multiple warmup runs
                _ = self.model.predict(dummy_img, verbose=False, device=self.device)
            
            self.is_initialized = True
            logger.success("✅ Detection engine initialized and warmed up")
            
            # Log performance info
            if 'cuda' in self.device:
                self._log_gpu_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize detection engine: {str(e)}")
            logger.info("💡 Tip: Run 'python check_cuda.py' to diagnose CUDA issues")
            return False
    
    def _apply_cuda_optimizations(self):
        """Apply CUDA-specific optimizations"""
        try:
            # Enable cuDNN autotuner (20-30% faster)
            torch.backends.cudnn.benchmark = True
            logger.info("✓ cuDNN autotuner enabled")
            
            # Enable half precision if supported
            # Note: Half precision applied during inference, not to model weights
            if self.config.model.half_precision:
                logger.info("✓ Half precision (FP16) will be used during inference")
            
            # Set optimal memory management
            torch.cuda.empty_cache()
            
            # Enable TF32 on Ampere GPUs (RTX 30xx, A100, etc.)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✓ TF32 enabled (Ampere+ GPU)")
                
        except Exception as e:
            logger.warning(f"Some CUDA optimizations failed: {e}")
    
    def _log_gpu_info(self):
        """Log GPU utilization information"""
        try:
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"GPU Memory: {memory_allocated:.2f} GB allocated, "
                       f"{memory_reserved:.2f} GB reserved, "
                       f"{memory_total:.2f} GB total")
            
            self.performance_stats['gpu_memory_used'] = memory_allocated
            self.performance_stats['gpu_memory_total'] = memory_total
            
        except Exception as e:
            logger.debug(f"Could not get GPU info: {e}")
    
    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[Detection]:
        """
        Perform optimized detection on a single image
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Override default confidence threshold
            
        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            logger.error("Detection engine not initialized")
            return []
        
        start_time = time.time()
        detections = []
        
        try:
            conf = conf_threshold or self.config.model.conf_threshold
            
            # Run inference with optimizations
            results = self.model.predict(
                image,
                conf=conf,
                iou=self.config.model.iou_threshold,
                verbose=False,
                device=self.device,
                half=self.config.model.half_precision and 'cuda' in self.device
            )
            
            # Process results
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    class_name = self.config.get_class_name(class_id)
                    priority = self.config.get_priority_level(class_name)
                    
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        center=(center_x, center_y),
                        priority=priority,
                        timestamp=time.time()
                    )
                    
                    detections.append(detection)
            
            # Update stats
            inference_time = time.time() - start_time
            self._update_stats(inference_time, len(detections))
            
            # Update history
            with self.lock:
                self.detection_history.append({
                    'timestamp': time.time(),
                    'detections': detections,
                    'inference_time': inference_time
                })
                
                if len(self.detection_history) > self.max_history:
                    self.detection_history.pop(0)
            
            logger.debug(f"Detected {len(detections)} objects in {inference_time*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Perform batch detection (GPU optimized)
        Much faster than individual detections
        
        Args:
            images: List of input images
            
        Returns:
            List of detection lists
        """
        if not self.is_initialized:
            return [[] for _ in images]
        
        batch_results = []
        
        try:
            # Batch inference (GPU parallelization)
            results = self.model.predict(
                images,
                conf=self.config.model.conf_threshold,
                iou=self.config.model.iou_threshold,
                verbose=False,
                device=self.device,
                half=self.config.model.half_precision and 'cuda' in self.device
            )
            
            for result in results:
                detections = []
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        class_name = self.config.get_class_name(class_id)
                        priority = self.config.get_priority_level(class_name)
                        
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        
                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                            center=(center_x, center_y),
                            priority=priority,
                            timestamp=time.time()
                        )
                        
                        detections.append(detection)
                
                batch_results.append(detections)
            
        except Exception as e:
            logger.error(f"Batch detection failed: {str(e)}")
            batch_results = [[] for _ in images]
        
        return batch_results
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection results on image"""
        result_image = image.copy()
        
        for detection in detections:
            color = self.config.get_class_color(detection.class_name)
            
            # Bounding box
            cv2.rectangle(
                result_image,
                (detection.bbox[0], detection.bbox[1]),
                (detection.bbox[2], detection.bbox[3]),
                color, 2
            )
            
            # Center point
            cv2.circle(result_image, detection.center, 5, color, -1)
            
            # Label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.priority == "CRITICAL":
                label += " ⚠️"
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Label background
            cv2.rectangle(
                result_image,
                (detection.bbox[0], detection.bbox[1] - label_height - 10),
                (detection.bbox[0] + label_width, detection.bbox[1]),
                color, -1
            )
            
            # Label text
            cv2.putText(
                result_image, label,
                (detection.bbox[0], detection.bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return result_image
    
    def _update_stats(self, inference_time: float, num_detections: int):
        """Update performance statistics"""
        self.performance_stats['total_frames'] += 1
        self.performance_stats['total_detections'] += num_detections
        
        # Running average
        total = self.performance_stats['total_frames']
        prev_avg = self.performance_stats['avg_inference_time']
        self.performance_stats['avg_inference_time'] = (prev_avg * (total - 1) + inference_time) / total
        
        # Calculate FPS
        elapsed = time.time() - self.performance_stats['start_time']
        if elapsed > 0:
            self.performance_stats['fps'] = total / elapsed
        
        # Update GPU memory every 100 frames
        if 'cuda' in self.device and total % 100 == 0:
            try:
                memory = torch.cuda.memory_allocated(0) / 1024**3
                self.performance_stats['gpu_memory_used'] = memory
            except:
                pass
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        with self.lock:
            stats = self.performance_stats.copy()
            stats['device'] = self.device
            stats['cuda_available'] = torch.cuda.is_available()
            return stats
    
    def get_detection_history(self) -> List[Dict]:
        """Get detection history"""
        with self.lock:
            return self.detection_history.copy()
    
    def reset_stats(self):
        """Reset performance statistics"""
        with self.lock:
            self.performance_stats = {
                'total_frames': 0,
                'total_detections': 0,
                'avg_inference_time': 0.0,
                'fps': 0.0,
                'start_time': time.time(),
                'gpu_memory_used': 0,
                'gpu_memory_total': 0
            }
            self.detection_history.clear()
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def shutdown(self):
        """Shutdown detection engine"""
        logger.info("Shutting down optimized detection engine")
        self.is_initialized = False
        
        if self.model:
            del self.model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory released")

if __name__ == "__main__":
    # Test optimized engine
    engine = AURADetectionEngineOptimized()
    
    if engine.initialize():
        # Test image
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "CUDA Test", (200, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Warmup and benchmark
        print("\n🔥 Benchmarking performance...")
        times = []
        for i in range(100):
            start = time.time()
            detections = engine.detect(test_image)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        fps = 1000 / avg_time
        
        print(f"\n📊 Results:")
        print(f"  Average inference: {avg_time:.2f} ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Device: {engine.device}")
        print(f"  Stats: {engine.get_stats()}")
        
        engine.shutdown()
    else:
        print("❌ Failed to initialize - check CUDA installation")
