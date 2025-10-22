import torch
import sys

print("="*50)
print("CUDA DIAGNOSTIC REPORT")
print("="*50)

# PyTorch info
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
else:
    print("\n⚠️  CUDA NOT AVAILABLE")
    print("Possible reasons:")
    print("  1. No NVIDIA GPU")
    print("  2. CUDA not installed")
    print("  3. PyTorch CPU-only version installed")
    
print("\n" + "="*50)
