# AURA CUDA Setup Script
# Automated PyTorch CUDA installation

Write-Host "🚀 AURA CUDA Setup Script" -ForegroundColor Cyan
Write-Host "=" * 50

# Check for NVIDIA GPU
Write-Host "`n📊 Checking for NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaCheck = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ NVIDIA GPU detected!" -ForegroundColor Green
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    } else {
        Write-Host "⚠️  No NVIDIA GPU found or drivers not installed" -ForegroundColor Red
        Write-Host "Install drivers from: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "⚠️  nvidia-smi not found - install NVIDIA drivers" -ForegroundColor Red
    exit 1
}

# Detect CUDA version
Write-Host "`n🔍 Detecting CUDA version..." -ForegroundColor Yellow
$cudaVersion = (nvidia-smi | Select-String "CUDA Version: (\d+\.\d+)").Matches.Groups[1].Value
Write-Host "CUDA Version: $cudaVersion" -ForegroundColor Cyan

# Determine PyTorch CUDA version
$torchCuda = "cu118"  # Default to CUDA 11.8 (most compatible)
if ($cudaVersion -ge "12.1") {
    $torchCuda = "cu121"
    Write-Host "Using PyTorch with CUDA 12.1" -ForegroundColor Green
} else {
    Write-Host "Using PyTorch with CUDA 11.8 (most compatible)" -ForegroundColor Green
}

# Uninstall old PyTorch
Write-Host "`n🗑️  Uninstalling CPU-only PyTorch..." -ForegroundColor Yellow
pip uninstall torch torchvision torchaudio -y

# Install CUDA-enabled PyTorch
Write-Host "`n📦 Installing CUDA-enabled PyTorch..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Cyan

$installUrl = "https://download.pytorch.org/whl/$torchCuda"
pip install torch torchvision torchaudio --index-url $installUrl

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PyTorch installation complete!" -ForegroundColor Green
} else {
    Write-Host "❌ Installation failed!" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host "`n✔️  Verifying CUDA installation..." -ForegroundColor Yellow
python check_cuda.py

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "`n📝 Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run 'python hackathon_demo.py' to test AURA" -ForegroundColor White
Write-Host "  2. Check GPU usage with 'nvidia-smi -l 1'" -ForegroundColor White
Write-Host "  3. Read CUDA_OPTIMIZATION_GUIDE.md for tuning tips" -ForegroundColor White
