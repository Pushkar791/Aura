"""
AURA Camera Interface
Real-time camera capture and processing system
"""

import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from typing import Optional, Callable, Tuple
from loguru import logger
import sys
from pathlib import Path

# Import configuration and detection engine
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import config
from src.detection_engine import AURADetectionEngine, Detection

class AURACameraInterface:
    """
    Real-time camera interface for AURA system
    Handles camera capture, frame buffering, and real-time processing
    """
    
    def __init__(self, camera_index: int = 0, detection_engine: Optional[AURADetectionEngine] = None):
        """
        Initialize camera interface
        
        Args:
            camera_index: Camera device index
            detection_engine: Detection engine instance
        """
        self.camera_index = camera_index
        self.detection_engine = detection_engine or AURADetectionEngine()
        self.config = config
        
        # Camera properties
        self.cap = None
        self.is_running = False
        self.is_recording = False
        
        # Threading and frame management
        self.capture_thread = None
        self.process_thread = None
        self.frame_queue = Queue(maxsize=5)  # Limit queue size to prevent memory buildup
        self.result_queue = Queue(maxsize=10)
        
        # Frame statistics
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Current frame and detections
        self.current_frame = None
        self.current_detections = []
        self.frame_lock = threading.Lock()
        
        # Callbacks
        self.detection_callback: Optional[Callable] = None
        self.frame_callback: Optional[Callable] = None
        
        logger.info(f"Camera interface initialized for camera {camera_index}")
    
    def initialize(self) -> bool:
        """
        Initialize camera and detection engine
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize detection engine
            if not self.detection_engine.is_initialized:
                if not self.detection_engine.initialize():
                    logger.error("Failed to initialize detection engine")
                    return False
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera {self.camera_index}")
                return False
            
            # Configure camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera.buffer_size)
            
            # Set additional camera properties if available
            if hasattr(cv2, 'CAP_PROP_AUTO_EXPOSURE'):
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if not self.config.camera.auto_exposure else 0.75)
            
            # Verify camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}FPS")
            
            # Test frame capture
            ret, test_frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture test frame")
                return False
            
            logger.success("Camera interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {str(e)}")
            return False
    
    def start(self):
        """Start camera capture and processing"""
        if self.is_running:
            logger.warning("Camera interface already running")
            return
        
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not initialized")
            return
        
        self.is_running = True
        self.fps_start_time = time.time()
        
        # Start capture and processing threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        self.capture_thread.start()
        self.process_thread.start()
        
        logger.info("Camera capture and processing started")
    
    def stop(self):
        """Stop camera capture and processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
        
        logger.info("Camera capture and processing stopped")
    
    def _capture_loop(self):
        """Main camera capture loop"""
        logger.info("Camera capture loop started")
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Failed to capture frame")
                    continue
                
                # Update frame statistics
                self.frame_count += 1
                self.fps_counter += 1
                
                # Calculate FPS
                if time.time() - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (time.time() - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Add frame to processing queue (non-blocking)
                try:
                    self.frame_queue.put_nowait((frame.copy(), time.time()))
                except:
                    # Queue full, skip this frame
                    pass
                
                # Update current frame for display
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Call frame callback if set
                if self.frame_callback:
                    self.frame_callback(frame.copy())
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {str(e)}")
                time.sleep(0.1)
        
        logger.info("Camera capture loop stopped")
    
    def _process_loop(self):
        """Main processing loop for detection"""
        logger.info("Processing loop started")
        
        while self.is_running:
            try:
                # Get frame from queue with timeout
                try:
                    frame_data = self.frame_queue.get(timeout=0.1)
                    frame, timestamp = frame_data
                except Empty:
                    continue
                
                # Perform detection
                detections = self.detection_engine.detect(frame)
                
                # Update current detections
                with self.frame_lock:
                    self.current_detections = detections
                
                # Add results to result queue
                try:
                    self.result_queue.put_nowait({
                        'frame': frame,
                        'detections': detections,
                        'timestamp': timestamp,
                        'processing_time': time.time() - timestamp
                    })
                except:
                    # Queue full, skip this result
                    pass
                
                # Call detection callback if set
                if self.detection_callback:
                    self.detection_callback(frame, detections)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)
        
        logger.info("Processing loop stopped")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current camera frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_current_detections(self) -> list:
        """Get current detections"""
        with self.frame_lock:
            return self.current_detections.copy()
    
    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get current frame with detections drawn"""
        frame = self.get_current_frame()
        detections = self.get_current_detections()
        
        if frame is not None and detections:
            return self.detection_engine.draw_detections(frame, detections)
        
        return frame
    
    def get_latest_result(self) -> Optional[dict]:
        """Get the latest processing result"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def set_detection_callback(self, callback: Callable[[np.ndarray, list], None]):
        """Set callback function for detection results"""
        self.detection_callback = callback
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function for new frames"""
        self.frame_callback = callback
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps
    
    def get_stats(self) -> dict:
        """Get camera and processing statistics"""
        detection_stats = self.detection_engine.get_stats() if self.detection_engine else {}
        
        return {
            'camera_fps': self.current_fps,
            'total_frames': self.frame_count,
            'is_running': self.is_running,
            'queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'detection_stats': detection_stats
        }
    
    def capture_screenshot(self, filename: Optional[str] = None) -> str:
        """Capture and save current frame"""
        frame = self.get_processed_frame()
        
        if frame is None:
            raise ValueError("No frame available for screenshot")
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"aura_screenshot_{timestamp}.jpg"
        
        filepath = self.config.logs_dir / filename
        cv2.imwrite(str(filepath), frame)
        
        logger.info(f"Screenshot saved: {filepath}")
        return str(filepath)
    
    def start_recording(self, filename: Optional[str] = None) -> str:
        """Start video recording"""
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        frame = self.get_current_frame()
        if frame is None:
            raise ValueError("No frame available for recording")
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"aura_recording_{timestamp}.mp4"
        
        filepath = self.config.logs_dir / filename
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frame.shape[:2]
        self.video_writer = cv2.VideoWriter(
            str(filepath), fourcc, self.config.camera.fps, (width, height)
        )
        
        self.is_recording = True
        self.recording_filename = str(filepath)
        
        logger.info(f"Started recording: {filepath}")
        return str(filepath)
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            logger.warning("Not currently recording")
            return
        
        self.is_recording = False
        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()
        
        logger.info(f"Recording stopped: {getattr(self, 'recording_filename', 'unknown')}")
    
    def shutdown(self):
        """Shutdown camera interface"""
        logger.info("Shutting down camera interface")
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Stop capture and processing
        self.stop()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Shutdown detection engine
        if self.detection_engine:
            self.detection_engine.shutdown()
        
        logger.success("Camera interface shut down")

if __name__ == "__main__":
    # Test the camera interface
    def detection_callback(frame, detections):
        print(f"Detected {len(detections)} objects")
        for detection in detections:
            print(f"  - {detection.class_name}: {detection.confidence:.2f}")
    
    # Create camera interface
    camera = AURACameraInterface(camera_index=0)
    
    if camera.initialize():
        camera.set_detection_callback(detection_callback)
        camera.start()
        
        # Run for 10 seconds
        print("Running camera for 10 seconds...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < 10.0:
                stats = camera.get_stats()
                print(f"FPS: {stats['camera_fps']:.1f}, Queue: {stats['queue_size']}")
                time.sleep(1.0)
        
        except KeyboardInterrupt:
            print("\nStopping camera...")
        
        finally:
            camera.shutdown()
            print("Camera test completed")
    
    else:
        print("Failed to initialize camera")
