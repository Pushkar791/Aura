"""
AURA Model Training Script
Train YOLOv8 on custom safety equipment dataset
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from configs.config import SAFETY_CLASSES

class AURATrainer:
    """Custom trainer for AURA safety equipment detection"""
    
    def __init__(self, data_yaml_path, model_size='n'):
        """
        Initialize trainer
        
        Args:
            data_yaml_path: Path to dataset YAML configuration
            model_size: YOLOv8 model size (n/s/m/l/x)
        """
        self.data_yaml = data_yaml_path
        self.model_size = model_size
        self.model_name = f"yolov8{model_size}.pt"
        self.project_dir = Path(__file__).parent
        self.results_dir = self.project_dir / "runs" / "train"
        
        print("=" * 70)
        print("🚀 AURA Model Training System")
        print("=" * 70)
        print(f"\n📁 Dataset: {data_yaml_path}")
        print(f"🤖 Model: YOLOv8{model_size.upper()}")
        print(f"💾 Results: {self.results_dir}")
        
    def check_gpu(self):
        """Check GPU availability"""
        print("\n" + "=" * 70)
        print("🔍 Checking GPU Availability")
        print("=" * 70)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ VRAM: {gpu_memory:.1f} GB")
            print(f"✅ CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU detected - training will use CPU (much slower)")
            return False
    
    def validate_dataset(self):
        """Validate dataset structure"""
        print("\n" + "=" * 70)
        print("📊 Validating Dataset")
        print("=" * 70)
        
        try:
            # Load YAML
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data_config:
                    print(f"❌ Missing required field: {field}")
                    return False
            
            # Get paths
            data_root = Path(data_config.get('path', self.project_dir / 'data'))
            train_path = data_root / data_config['train']
            val_path = data_root / data_config['val']
            
            # Check directories
            print(f"\n📂 Dataset Root: {data_root}")
            print(f"📂 Train Path: {train_path}")
            print(f"📂 Val Path: {val_path}")
            
            # Count images
            train_images = list(Path(train_path).glob('**/*.jpg')) + list(Path(train_path).glob('**/*.png'))
            val_images = list(Path(val_path).glob('**/*.jpg')) + list(Path(val_path).glob('**/*.png'))
            
            print(f"\n📸 Training Images: {len(train_images)}")
            print(f"📸 Validation Images: {len(val_images)}")
            print(f"🏷️  Number of Classes: {data_config['nc']}")
            print(f"🏷️  Class Names: {data_config['names']}")
            
            if len(train_images) == 0:
                print("\n❌ No training images found!")
                return False
            
            if len(val_images) == 0:
                print("\n⚠️  Warning: No validation images found")
            
            print("\n✅ Dataset validation passed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Dataset validation failed: {str(e)}")
            return False
    
    def train(self, epochs=100, batch_size=16, img_size=640, patience=50, 
              save_period=10, resume=False):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size (reduce if GPU out of memory)
            img_size: Input image size
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            resume: Resume from last checkpoint
        """
        print("\n" + "=" * 70)
        print("🎯 Starting Training")
        print("=" * 70)
        
        # Training configuration
        print(f"\n⚙️  Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Image Size: {img_size}")
        print(f"   Early Stopping Patience: {patience}")
        print(f"   Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        
        try:
            # Load model
            print(f"\n📥 Loading model: {self.model_name}")
            model = YOLO(self.model_name)
            
            # Start training
            print(f"\n🚀 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                patience=patience,
                save_period=save_period,
                project=str(self.results_dir.parent),
                name='safety_equipment',
                exist_ok=True,
                pretrained=True,
                optimizer='SGD',
                verbose=True,
                seed=42,
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=True,
                close_mosaic=10,
                resume=resume,
                amp=True,  # Automatic Mixed Precision
                fraction=1.0,
                profile=False,
                # Augmentation
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                copy_paste=0.0
            )
            
            print("\n" + "=" * 70)
            print(f"✅ Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            # Get best model path
            best_model = self.results_dir / 'safety_equipment' / 'weights' / 'best.pt'
            last_model = self.results_dir / 'safety_equipment' / 'weights' / 'last.pt'
            
            print(f"\n📊 Training Results:")
            print(f"   Best Model: {best_model}")
            print(f"   Last Model: {last_model}")
            print(f"   Results Directory: {self.results_dir / 'safety_equipment'}")
            
            return True, best_model
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            return False, None
    
    def evaluate(self, model_path):
        """Evaluate trained model"""
        print("\n" + "=" * 70)
        print("📈 Evaluating Model")
        print("=" * 70)
        
        try:
            model = YOLO(str(model_path))
            
            # Validate on validation set
            print("\nRunning validation...")
            metrics = model.val(data=self.data_yaml, verbose=True)
            
            print(f"\n📊 Validation Metrics:")
            print(f"   mAP@0.5: {metrics.box.map50:.4f}")
            print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
            print(f"   Precision: {metrics.box.p.mean():.4f}")
            print(f"   Recall: {metrics.box.r.mean():.4f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {str(e)}")
            return False
    
    def export_model(self, model_path, formats=['onnx', 'torchscript']):
        """Export model to different formats"""
        print("\n" + "=" * 70)
        print("📤 Exporting Model")
        print("=" * 70)
        
        try:
            model = YOLO(str(model_path))
            
            for fmt in formats:
                print(f"\n🔄 Exporting to {fmt.upper()}...")
                model.export(format=fmt)
                print(f"✅ {fmt.upper()} export complete")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Export failed: {str(e)}")
            return False


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AURA safety equipment detection model')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip dataset validation')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--export', action='store_true',
                       help='Export model after training')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AURATrainer(args.data, args.model)
    
    # Check GPU
    trainer.check_gpu()
    
    # Validate dataset
    if not args.skip_validation and not args.evaluate_only:
        if not trainer.validate_dataset():
            print("\n❌ Dataset validation failed. Fix errors and try again.")
            return
    
    # Train model
    if not args.evaluate_only:
        success, model_path = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            patience=args.patience,
            resume=args.resume
        )
        
        if not success:
            print("\n❌ Training failed!")
            return
    else:
        # For evaluation only, find best model
        model_path = trainer.results_dir / 'safety_equipment' / 'weights' / 'best.pt'
        if not model_path.exists():
            print(f"\n❌ Model not found at {model_path}")
            return
    
    # Evaluate model
    trainer.evaluate(model_path)
    
    # Export if requested
    if args.export:
        trainer.export_model(model_path)
    
    print("\n" + "=" * 70)
    print("🎉 All done! Your model is ready to use.")
    print("=" * 70)
    print(f"\n📝 Next steps:")
    print(f"   1. Copy trained model to: models/best.pt")
    print(f"   2. Update configs/config.py to use your model")
    print(f"   3. Run: python test_detection.py")
    print(f"   4. Run: python hackathon_demo.py")
    print("\n💡 Model location: {model_path}")


if __name__ == "__main__":
    main()
