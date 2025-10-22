"""
AURA Falcon Simulation Integration
System for handling synthetic training data from Duality AI's Falcon digital twin simulation
"""

import os
import json
import requests
import websocket
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
import cv2
import numpy as np
from loguru import logger
import shutil
from datetime import datetime

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import config, SAFETY_CLASSES
from src.detection_engine import AURADetectionEngine

class FalconDataProcessor:
    """
    Processes synthetic data from Falcon simulation
    Converts to YOLO format and manages training datasets
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize Falcon data processor
        
        Args:
            data_dir: Directory for storing processed data
        """
        self.data_dir = data_dir or Path(config.DATA_DIR)
        self.falcon_config = config.falcon
        
        # Dataset directories
        self.raw_dir = self.data_dir / "falcon_raw"
        self.processed_dir = self.data_dir / "falcon_processed"
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        
        # Create directories
        self._create_directories()
        
        # Statistics
        self.processed_count = 0
        self.last_processed = None
        
        logger.info("Falcon data processor initialized")
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.raw_dir,
            self.processed_dir,
            self.train_dir / "images",
            self.train_dir / "labels",
            self.val_dir / "images",
            self.val_dir / "labels"
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_falcon_batch(self, batch_data: Dict) -> bool:
        """
        Process a batch of data from Falcon simulation
        
        Args:
            batch_data: Dictionary containing images and annotations
            
        Returns:
            bool: True if processing successful
        """
        try:
            batch_id = batch_data.get('batch_id', f"batch_{int(time.time())}")
            images = batch_data.get('images', [])
            annotations = batch_data.get('annotations', [])
            
            logger.info(f"Processing Falcon batch {batch_id} with {len(images)} images")
            
            processed_images = []
            processed_labels = []
            
            for i, (image_data, annotation_data) in enumerate(zip(images, annotations)):
                try:
                    # Process image
                    image = self._process_image(image_data, f"{batch_id}_{i}")
                    if image is None:
                        continue
                    
                    # Process annotations
                    labels = self._process_annotations(annotation_data, image.shape[:2])
                    if labels is None:
                        continue
                    
                    processed_images.append(image)
                    processed_labels.append(labels)
                    
                except Exception as e:
                    logger.error(f"Error processing image {i} in batch {batch_id}: {str(e)}")
                    continue
            
            # Save processed data
            saved_count = self._save_processed_data(
                batch_id, processed_images, processed_labels
            )
            
            self.processed_count += saved_count
            self.last_processed = datetime.now()
            
            logger.success(f"Successfully processed {saved_count} images from batch {batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process Falcon batch: {str(e)}")
            return False
    
    def _process_image(self, image_data: Dict, image_id: str) -> Optional[np.ndarray]:
        """Process image data from Falcon"""
        try:
            if 'base64' in image_data:
                # Decode base64 image
                import base64
                image_bytes = base64.b64decode(image_data['base64'])
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
            elif 'url' in image_data:
                # Download image from URL
                response = requests.get(image_data['url'])
                image_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
            elif 'path' in image_data:
                # Load from file path
                image = cv2.imread(image_data['path'])
                
            else:
                logger.error(f"Unsupported image format for {image_id}")
                return None
            
            if image is None:
                logger.error(f"Failed to load image {image_id}")
                return None
            
            # Resize if needed
            target_size = (640, 640)
            if image.shape[:2] != target_size:
                image = cv2.resize(image, target_size)
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {str(e)}")
            return None
    
    def _process_annotations(self, annotation_data: Dict, image_shape: tuple) -> Optional[List[str]]:
        """Process annotation data and convert to YOLO format"""
        try:
            height, width = image_shape
            yolo_labels = []
            
            objects = annotation_data.get('objects', [])
            
            for obj in objects:
                # Get class information
                class_name = obj.get('class', '').strip()
                if class_name not in SAFETY_CLASSES.values():
                    logger.warning(f"Unknown class: {class_name}")
                    continue
                
                # Get class ID
                class_id = None
                for id, name in SAFETY_CLASSES.items():
                    if name == class_name:
                        class_id = id
                        break
                
                if class_id is None:
                    continue
                
                # Get bounding box
                bbox = obj.get('bbox', {})
                
                if 'x' in bbox and 'y' in bbox and 'width' in bbox and 'height' in bbox:
                    # Convert to YOLO format (normalized center coordinates)
                    x = bbox['x']
                    y = bbox['y']
                    w = bbox['width']
                    h = bbox['height']
                    
                elif 'x1' in bbox and 'y1' in bbox and 'x2' in bbox and 'y2' in bbox:
                    # Convert from corner coordinates
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    
                else:
                    logger.warning(f"Invalid bbox format for {class_name}")
                    continue
                
                # Normalize coordinates
                x_center = x / width
                y_center = y / height
                norm_width = w / width
                norm_height = h / height
                
                # Create YOLO label string
                yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                yolo_labels.append(yolo_label)
            
            return yolo_labels if yolo_labels else None
            
        except Exception as e:
            logger.error(f"Error processing annotations: {str(e)}")
            return None
    
    def _save_processed_data(self, batch_id: str, images: List[np.ndarray], labels: List[List[str]]) -> int:
        """Save processed images and labels"""
        saved_count = 0
        
        for i, (image, label_list) in enumerate(zip(images, labels)):
            try:
                # Generate filename
                filename = f"{batch_id}_{i:04d}"
                
                # Determine train/val split (80/20)
                is_train = np.random.random() < 0.8
                
                if is_train:
                    image_dir = self.train_dir / "images"
                    label_dir = self.train_dir / "labels"
                else:
                    image_dir = self.val_dir / "images"
                    label_dir = self.val_dir / "labels"
                
                # Save image
                image_path = image_dir / f"{filename}.jpg"
                cv2.imwrite(str(image_path), image)
                
                # Save labels
                label_path = label_dir / f"{filename}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_list))
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving data for {filename}: {str(e)}")
                continue
        
        return saved_count
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'train_images': len(list((self.train_dir / "images").glob("*.jpg"))),
            'train_labels': len(list((self.train_dir / "labels").glob("*.txt"))),
            'val_images': len(list((self.val_dir / "images").glob("*.jpg"))),
            'val_labels': len(list((self.val_dir / "labels").glob("*.txt"))),
            'processed_count': self.processed_count,
            'last_processed': self.last_processed
        }
        
        return stats

class FalconAPIClient:
    """Client for communicating with Falcon simulation API"""
    
    def __init__(self, api_endpoint: str = None):
        """Initialize Falcon API client"""
        self.api_endpoint = api_endpoint or config.falcon.api_endpoint
        self.session = requests.Session()
        self.is_connected = False
        
        logger.info(f"Falcon API client initialized for {self.api_endpoint}")
    
    def test_connection(self) -> bool:
        """Test connection to Falcon API"""
        try:
            response = self.session.get(f"{self.api_endpoint}/health", timeout=5)
            self.is_connected = response.status_code == 200
            
            if self.is_connected:
                logger.success("Connected to Falcon API")
            else:
                logger.warning(f"Falcon API connection failed: {response.status_code}")
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Failed to connect to Falcon API: {str(e)}")
            self.is_connected = False
            return False
    
    def request_synthetic_data(self, parameters: Dict = None) -> Optional[Dict]:
        """Request synthetic data generation"""
        try:
            params = parameters or {
                'scene_type': 'space_station',
                'equipment_types': list(SAFETY_CLASSES.values()),
                'num_images': 50,
                'lighting_variations': True,
                'occlusion_variations': True,
                'angle_variations': True
            }
            
            response = self.session.post(
                f"{self.api_endpoint}/generate",
                json=params,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.success("Synthetic data generation requested")
                return response.json()
            else:
                logger.error(f"Failed to request synthetic data: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error requesting synthetic data: {str(e)}")
            return None
    
    def get_generation_status(self, job_id: str) -> Optional[Dict]:
        """Get status of data generation job"""
        try:
            response = self.session.get(f"{self.api_endpoint}/status/{job_id}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get job status: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return None
    
    def download_generated_data(self, job_id: str) -> Optional[Dict]:
        """Download generated synthetic data"""
        try:
            response = self.session.get(f"{self.api_endpoint}/download/{job_id}")
            
            if response.status_code == 200:
                logger.success(f"Downloaded data for job {job_id}")
                return response.json()
            else:
                logger.error(f"Failed to download data: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return None

class FalconIntegrationManager:
    """Main manager for Falcon simulation integration"""
    
    def __init__(self, detection_engine: Optional[AURADetectionEngine] = None):
        """Initialize Falcon integration manager"""
        self.detection_engine = detection_engine
        self.api_client = FalconAPIClient()
        self.data_processor = FalconDataProcessor()
        
        # Auto-update settings
        self.auto_update_enabled = config.falcon.auto_retrain
        self.update_interval = config.falcon.update_interval
        self.min_new_samples = config.falcon.min_new_samples
        
        # Threading
        self.update_thread = None
        self.is_running = False
        
        # Callbacks
        self.data_callback: Optional[Callable] = None
        self.retrain_callback: Optional[Callable] = None
        
        logger.info("Falcon integration manager initialized")
    
    def start_auto_update(self):
        """Start automatic data updates from Falcon"""
        if not self.auto_update_enabled:
            logger.warning("Auto-update is disabled in configuration")
            return
        
        if self.is_running:
            logger.warning("Auto-update already running")
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info(f"Started auto-update with {self.update_interval}s interval")
    
    def stop_auto_update(self):
        """Stop automatic updates"""
        self.is_running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        
        logger.info("Stopped auto-update")
    
    def _update_loop(self):
        """Main auto-update loop"""
        while self.is_running:
            try:
                # Test API connection
                if not self.api_client.test_connection():
                    logger.warning("Falcon API not available, skipping update")
                    time.sleep(60)  # Wait 1 minute before retry
                    continue
                
                # Request new synthetic data
                logger.info("Requesting new synthetic data from Falcon")
                job_data = self.api_client.request_synthetic_data()
                
                if job_data and 'job_id' in job_data:
                    job_id = job_data['job_id']
                    logger.info(f"Data generation job created: {job_id}")
                    
                    # Wait for completion and download
                    if self._wait_and_download(job_id):
                        # Check if we have enough new samples for retraining
                        stats = self.data_processor.get_dataset_stats()
                        
                        if stats['train_images'] >= self.min_new_samples:
                            if self.retrain_callback:
                                logger.info("Triggering model retraining")
                                self.retrain_callback()
                        
                        if self.data_callback:
                            self.data_callback(stats)
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _wait_and_download(self, job_id: str, max_wait: int = 300) -> bool:
        """Wait for job completion and download data"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = self.api_client.get_generation_status(job_id)
            
            if not status:
                return False
            
            if status['status'] == 'completed':
                # Download and process data
                data = self.api_client.download_generated_data(job_id)
                
                if data:
                    return self.data_processor.process_falcon_batch(data)
                return False
            
            elif status['status'] == 'failed':
                logger.error(f"Job {job_id} failed: {status.get('error', 'Unknown error')}")
                return False
            
            # Still processing, wait a bit
            time.sleep(10)
        
        logger.warning(f"Job {job_id} timed out after {max_wait}s")
        return False
    
    def manual_data_request(self, parameters: Dict = None) -> bool:
        """Manually request synthetic data"""
        if not self.api_client.test_connection():
            logger.error("Cannot connect to Falcon API")
            return False
        
        job_data = self.api_client.request_synthetic_data(parameters)
        
        if job_data and 'job_id' in job_data:
            job_id = job_data['job_id']
            logger.info(f"Manual data request created: {job_id}")
            return self._wait_and_download(job_id)
        
        return False
    
    def get_integration_stats(self) -> Dict:
        """Get integration statistics"""
        dataset_stats = self.data_processor.get_dataset_stats()
        
        return {
            'api_connected': self.api_client.is_connected,
            'auto_update_running': self.is_running,
            'dataset_stats': dataset_stats,
            'config': {
                'auto_retrain': self.auto_update_enabled,
                'update_interval': self.update_interval,
                'min_new_samples': self.min_new_samples
            }
        }
    
    def set_data_callback(self, callback: Callable[[Dict], None]):
        """Set callback for new data events"""
        self.data_callback = callback
    
    def set_retrain_callback(self, callback: Callable[[], None]):
        """Set callback for retraining trigger"""
        self.retrain_callback = callback
    
    def shutdown(self):
        """Shutdown integration manager"""
        logger.info("Shutting down Falcon integration manager")
        self.stop_auto_update()
        logger.success("Falcon integration manager shut down")

if __name__ == "__main__":
    # Test the Falcon integration
    def data_callback(stats):
        print(f"New data received: {stats}")
    
    def retrain_callback():
        print("Retraining triggered!")
    
    manager = FalconIntegrationManager()
    manager.set_data_callback(data_callback)
    manager.set_retrain_callback(retrain_callback)
    
    # Test API connection
    if manager.api_client.test_connection():
        print("API connected, testing manual data request...")
        
        # Test manual request
        success = manager.manual_data_request({
            'num_images': 10,
            'scene_type': 'space_station_test'
        })
        
        print(f"Manual request: {'Success' if success else 'Failed'}")
    
    # Print stats
    stats = manager.get_integration_stats()
    print(f"Integration stats: {json.dumps(stats, indent=2, default=str)}")
    
    manager.shutdown()
