"""
AURA Detection Engine
Core detection system using YOLOv8 for real-time safety equipment recognition
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
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    priority: str
    timestamp: float

class AURADetectionEngine:
    """
    AURA Detection Engine
    High-performance real-time detection system for safety equipment
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detection engine
        
        Args:
            model_path: Path to custom trained model, if None uses pretrained YOLOv8
        """
        self.config = config
        self.model_path = model_path or self.config.model.model_name
        self.model = None
        self.device = None
        self.is_initialized = False
        self.performance_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_inference_time': 0.0,
            'fps': 0.0,
            'start_time': time.time()
        }
        
        # Detection history for tracking
        self.detection_history = []
        self.max_history = self.config.dashboard.history_length
        
        # Threading for performance
        self.lock = threading.Lock()
        
        logger.info("AURA Detection Engine initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the YOLO model and setup device
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info(f"Loading model: {self.model_path}")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                logger.warning("CUDA not available. Using CPU for inference")
            
            # Load YOLO model
            self.model = YOLO(self.model_path)
            
            # Configure model settings
            if hasattr(self.model, 'model'):
                self.model.model.to(self.device)
                # Note: Half precision is applied per-inference, not to model weights
                if self.config.model.half_precision and self.device != 'cpu':
                    logger.info("Half precision will be used during inference")
            
            # Warm up model with dummy inference
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_img, verbose=False)
            
            self.is_initialized = True
            logger.success("Detection engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize detection engine: {str(e)}")
            return False
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Perform detection on a single image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            logger.error("Detection engine not initialized")
            return []
        
        start_time = time.time()
        detections = []
        
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=self.config.model.conf_threshold,
                iou=self.config.model.iou_threshold,
                verbose=False,
                device=self.device,
                half=self.config.model.half_precision and self.device != 'cpu'
            )
            
            # Process results
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Extract detection data
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class information
                    class_name = self.config.get_class_name(class_id)
                    priority = self.config.get_priority_level(class_name)
                    
                    # Calculate center point
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    
                    # Create detection object
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
            
            # Update performance statistics
            inference_time = time.time() - start_time
            self._update_stats(inference_time, len(detections))
            
            # Update detection history
            with self.lock:
                self.detection_history.append({
                    'timestamp': time.time(),
                    'detections': detections,
                    'inference_time': inference_time
                })
                
                # Maintain history limit
                if len(self.detection_history) > self.max_history:
                    self.detection_history.pop(0)
            
            logger.debug(f"Detected {len(detections)} objects in {inference_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Perform batch detection on multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of detection lists for each image
        """
        if not self.is_initialized:
            logger.error("Detection engine not initialized")
            return [[] for _ in images]
        
        batch_results = []
        
        try:
            # Run batch inference
            results = self.model.predict(
                images,
                conf=self.config.model.conf_threshold,
                iou=self.config.model.iou_threshold,
                verbose=False,
                device=self.device,
                half=self.config.model.half_precision and self.device != 'cpu'
            )
            
            # Process each image result
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
        """
        Draw detection results on image
        
        Args:
            image: Input image
            detections: List of detections to draw
            
        Returns:
            Image with detections drawn
        """
        result_image = image.copy()
        
        for detection in detections:
            # Get colors
            color = self.config.get_class_color(detection.class_name)
            
            # Draw bounding box
            cv2.rectangle(
                result_image,
                (detection.bbox[0], detection.bbox[1]),
                (detection.bbox[2], detection.bbox[3]),
                color,
                2
            )
            
            # Draw center point
            cv2.circle(
                result_image,
                detection.center,
                5,
                color,
                -1
            )
            
            # Draw label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.priority == "CRITICAL":
                label += " ⚠️"
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                result_image,
                (detection.bbox[0], detection.bbox[1] - label_height - 10),
                (detection.bbox[0] + label_width, detection.bbox[1]),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_image,
                label,
                (detection.bbox[0], detection.bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return result_image
    
    def _update_stats(self, inference_time: float, num_detections: int):
        """Update performance statistics"""
        self.performance_stats['total_frames'] += 1
        self.performance_stats['total_detections'] += num_detections
        
        # Update average inference time
        total_time = self.performance_stats['avg_inference_time'] * (self.performance_stats['total_frames'] - 1)
        self.performance_stats['avg_inference_time'] = (total_time + inference_time) / self.performance_stats['total_frames']
        
        # Calculate FPS
        elapsed_time = time.time() - self.performance_stats['start_time']
        if elapsed_time > 0:
            self.performance_stats['fps'] = self.performance_stats['total_frames'] / elapsed_time
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        with self.lock:
            return self.performance_stats.copy()
    
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
                'start_time': time.time()
            }
            self.detection_history.clear()
    
    def shutdown(self):
        """Shutdown detection engine"""
        logger.info("Shutting down detection engine")
        self.is_initialized = False
        if self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test the detection engine
    engine = AURADetectionEngine()
    
    if engine.initialize():
        # Create a test image
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "AURA Test", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Test detection
        detections = engine.detect(test_image)
        print(f"Test completed. Detected {len(detections)} objects.")
        print(f"Performance stats: {engine.get_stats()}")
        
        engine.shutdown()
    else:
        print("Failed to initialize detection engine")