"""
AURA Improved Training Script
Optimized hyperparameters for maximum accuracy, precision, and recall
"""

import os
import sys
import torch
from ultralytics import YOLO
from datetime import datetime

def train_improved_model():
    print("=" * 70)
    print("🚀 AURA IMPROVED TRAINING - Maximum Performance")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {gpu_memory:.1f} GB")
    else:
        print("❌ No GPU detected!")
        return
    
    # Paths
    data_yaml = "data/dataset.yaml"
    
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset YAML not found: {data_yaml}")
        return
    
    print(f"\n📊 Dataset: {data_yaml}")
    
    # Load model - YOLOv8n (nano) for your GPU
    model = YOLO('yolov8n.pt')
    print("✅ Model loaded: YOLOv8n")
    
    print("\n" + "=" * 70)
    print("🎯 TRAINING CONFIGURATION - OPTIMIZED FOR HACKATHON")
    print("=" * 70)
    print("📈 Epochs: 100 (vs 10 before)")
    print("📦 Batch Size: 16 (optimized for RTX 3050)")
    print("🖼️  Image Size: 640")
    print("🎲 Augmentation: Enhanced")
    print("📉 Learning Rate: Optimized")
    print("⏰ Early Stopping: 20 epochs patience")
    print("=" * 70)
    
    # Start training with OPTIMIZED parameters
    print("\n🚀 Starting Training...\n")
    
    results = model.train(
        # Data
        data=data_yaml,
        
        # Training duration - INCREASED
        epochs=100,              # 10x more than before!
        
        # Batch and image settings
        batch=16,                # Optimal for RTX 3050 4GB
        imgsz=640,
        
        # Optimizer settings - OPTIMIZED
        optimizer='AdamW',       # Better than SGD for small datasets
        lr0=0.001,              # Initial learning rate
        lrf=0.01,               # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation - ENHANCED
        hsv_h=0.015,            # Hue augmentation
        hsv_s=0.7,              # Saturation
        hsv_v=0.4,              # Value
        degrees=10.0,           # Rotation (increased)
        translate=0.1,          # Translation
        scale=0.5,              # Scaling
        shear=2.0,              # Shear (increased)
        perspective=0.0001,     # Perspective
        flipud=0.0,             # Vertical flip
        fliplr=0.5,             # Horizontal flip
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mixup augmentation (added)
        copy_paste=0.1,         # Copy-paste augmentation (added)
        
        # Early stopping and validation
        patience=20,            # Stop if no improvement for 20 epochs
        save=True,
        save_period=10,         # Save every 10 epochs
        val=True,
        
        # Output
        project='runs/train',
        name='improved_model',
        exist_ok=True,
        
        # Performance
        device=0,               # Use GPU
        workers=8,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        
        # Loss weights - TUNED for better recall
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Advanced settings for better performance
        close_mosaic=10,        # Disable mosaic in last 10 epochs
        amp=True,               # Automatic Mixed Precision
        fraction=1.0,           # Use 100% of dataset
        
        # Multi-scale training
        multi_scale=True,       # ENABLED for better generalization
        
        # Learning rate scheduler
        cos_lr=True,            # Cosine learning rate scheduler
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    
    # Get best model path
    best_model = 'runs/train/improved_model/weights/best.pt'
    
    print(f"\n📦 Best Model: {best_model}")
    print(f"📊 Results: runs/train/improved_model/")
    
    # Validate on test set
    print("\n" + "=" * 70)
    print("🔍 VALIDATING BEST MODEL")
    print("=" * 70)
    
    best = YOLO(best_model)
    metrics = best.val(data=data_yaml, device=0)
    
    print("\n📈 VALIDATION METRICS:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    if hasattr(metrics.box, 'mp') and hasattr(metrics.box, 'mr'):
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
    
    print("\n" + "=" * 70)
    print("🎉 NEXT STEPS")
    print("=" * 70)
    print("1. Run: python evaluate_test.py (to test on test dataset)")
    print("2. Run: python optimize_model.py (to find best confidence threshold)")
    print("3. Copy best model to: models/best.pt")
    print("4. Update config with optimal confidence threshold")
    print("=" * 70)


if __name__ == "__main__":
    print("\n⚡ This will train for 100 epochs (~2-3 hours)")
    print("💡 The model will be MUCH better than the 10-epoch version!\n")
    
    response = input("Start improved training? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        train_improved_model()
    else:
        print("❌ Training cancelled")
