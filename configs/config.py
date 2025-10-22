"""
AURA Configuration Module
Contains all configuration settings for the Autonomous Utility Recognition Assistant
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Base paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
CONFIGS_DIR = ROOT_DIR / "configs"

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_name: str = "yolov8n.pt"
    custom_model_path: str = str(MODELS_DIR / "best.pt")
    config_file: str = str(CONFIGS_DIR / "safety_equipment.yaml")
    img_size: int = 640
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"
    half_precision: bool = True

@dataclass
class CameraConfig:
    """Camera and video processing configuration"""
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1
    auto_exposure: bool = False
    brightness: float = 0.5
    contrast: float = 0.5

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    host: str = "localhost"
    port: int = 8501
    update_interval: float = 0.1
    history_length: int = 100
    alert_threshold: float = 0.8
    show_confidence: bool = True
    show_fps: bool = True

@dataclass
class FalconConfig:
    """Falcon simulation integration configuration"""
    enabled: bool = False
    api_endpoint: str = "http://localhost:8080/api/v1"
    update_interval: int = 3600  # seconds
    data_format: str = "yolo"
    auto_retrain: bool = True
    min_new_samples: int = 100

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    log_file: str = str(LOGS_DIR / "aura.log")
    rotation: str = "10 MB"
    retention: str = "7 days"

# Safety equipment classes
SAFETY_CLASSES = {
    0: "OxygenTank",
    1: "NitrogenTank", 
    2: "FirstAidBox",
    3: "FireAlarm",
    4: "SafetySwitchPanel",
    5: "EmergencyPhone",
    6: "FireExtinguisher"
}

# Class colors for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    "OxygenTank": (0, 255, 0),        # Green
    "NitrogenTank": (255, 0, 0),      # Blue
    "FirstAidBox": (0, 255, 255),     # Yellow
    "FireAlarm": (0, 0, 255),         # Red
    "SafetySwitchPanel": (255, 255, 0), # Cyan
    "EmergencyPhone": (255, 0, 255),   # Magenta
    "FireExtinguisher": (128, 0, 128)  # Purple
}

# Priority levels for different equipment
PRIORITY_LEVELS = {
    "OxygenTank": "CRITICAL",
    "NitrogenTank": "CRITICAL",
    "FirstAidBox": "HIGH",
    "FireAlarm": "CRITICAL",
    "SafetySwitchPanel": "HIGH",
    "EmergencyPhone": "CRITICAL",
    "FireExtinguisher": "CRITICAL"
}

# Alert thresholds per class
ALERT_THRESHOLDS = {
    "OxygenTank": 0.9,
    "NitrogenTank": 0.9,
    "FirstAidBox": 0.7,
    "FireAlarm": 0.8,
    "SafetySwitchPanel": 0.7,
    "EmergencyPhone": 0.8,
    "FireExtinguisher": 0.8
}

class AURAConfig:
    """Main AURA configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.camera = CameraConfig()
        self.dashboard = DashboardConfig()
        self.falcon = FalconConfig()
        self.logging = LoggingConfig()
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID (using COCO classes for pretrained model)"""
        # COCO dataset class names for YOLOv8 pretrained model
        COCO_CLASSES = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        return COCO_CLASSES.get(class_id, f"object-{class_id}")
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get class color for visualization"""
        return CLASS_COLORS.get(class_name, (128, 128, 128))
    
    def get_priority_level(self, class_name: str) -> str:
        """Get priority level for a class"""
        # For demo with COCO classes, all detections are informational
        return "INFO"
    
    def get_alert_threshold(self, class_name: str) -> float:
        """Get alert threshold for a class"""
        return ALERT_THRESHOLDS.get(class_name, 0.7)

# Global configuration instance
config = AURAConfig()

# Expose directory paths as module-level variables for backward compatibility
DATA_DIR = DATA_DIR
MODELS_DIR = MODELS_DIR
LOGS_DIR = LOGS_DIR
