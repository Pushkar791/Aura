# 🚀 AURA Web Application - Quick Start Guide

## 📋 What You Get

A professional web interface with **2 modes**:

1. **🎥 Real-Time Detection** - Live webcam feed with safety equipment detection
2. **📊 Dataset Evaluation** - Process test images and display accuracy metrics

---

## ⚡ Quick Start

### 1. Install Flask (if not already installed)
```bash
pip install flask
```

### 2. Start the Server
```bash
python app.py
```

### 3. Open Browser
Navigate to: **http://localhost:5000**

---

## 🎯 Features

### Real-Time Detection Mode
✅ Live webcam/camera feed  
✅ Real-time object detection  
✅ FPS counter  
✅ Detection confidence overlay  
✅ Color-coded bounding boxes  
✅ System information display  

### Dataset Evaluation Mode
✅ One-click evaluation  
✅ Progress bar with live updates  
✅ Comprehensive metrics (Accuracy, Precision, Recall, F1)  
✅ Per-class performance breakdown  
✅ Professional results display  
✅ Interactive UI  

---

## 🎨 UI Highlights

- **Modern gradient background** - Purple/blue gradient
- **Glass morphism design** - Frosted glass effect cards
- **Responsive layout** - Works on all screen sizes
- **Smooth animations** - Professional transitions
- **Color-coded detections** - Each class has unique color
- **Real-time metrics** - Live FPS, inference time, detections

---

## 🔧 Configuration

Edit `app.py` to customize:

```python
MODEL_PATH = "runs/safety_equipment/weights/best.pt"  # Your model
TEST_DATA_PATH = r"C:\Users\...\test"  # Test dataset path
CONFIDENCE_THRESHOLD = 0.20  # Detection threshold
```

---

## 📊 What Judges Will See

### Mode 1: Real-Time Detection
1. Click "Real-Time Detection" button
2. **Live camera feed** with colored bounding boxes
3. **Detection stats** showing FPS and inference time
4. **Color legend** for each safety equipment class
5. **System info** showing GPU, device, confidence

### Mode 2: Dataset Evaluation
1. Click "Dataset Evaluation" button
2. Click "Start Evaluation" - shows **progress bar**
3. After completion, displays:
   - **Big metric cards**: Accuracy, Precision, Recall, F1
   - **Stats grid**: Images processed, detections, FPS
   - **Performance table**: Per-class breakdown
   - All in **real percentages** (e.g., 31.4%, 67.4%)

---

## 🏆 Hackathon Demo Tips

### For Judges:

1. **Start with Real-Time Mode**
   - Show live detection working
   - Point out the FPS (21 FPS)
   - Show different objects being detected

2. **Switch to Evaluation Mode**
   - Click "Start Evaluation"
   - Let them see the progress bar (impressive!)
   - Results appear automatically

3. **Highlight Key Metrics**
   - Point to big numbers (Accuracy, Precision, Recall)
   - Show per-class performance table
   - Mention GPU acceleration

### Talking Points:
- "Real-time detection at 21 FPS on RTX 3050"
- "Trained on 1,000+ images across 7 equipment types"
- "Achieves X% accuracy with Y% precision"
- "Handles varying lighting and clutter conditions"
- "Production-ready web interface"

---

## 🔥 Making It Even Better

### Before the Hackathon:

1. **Train improved model** (100 epochs)
   ```bash
   python train_improved.py
   ```
   
2. **Update app.py** with new model path:
   ```python
   MODEL_PATH = "runs/train/improved_model/weights/best.pt"
   ```

3. **Test everything**:
   ```bash
   python app.py
   ```
   Open browser and test both modes

### Expected Results After Improved Training:
- Accuracy: **50-60%** (vs 31% now)
- Precision: **75-85%** (vs 67% now)  
- Recall: **60-70%** (vs 37% now)
- Much more impressive for judges!

---

## 🐛 Troubleshooting

### Camera not working?
- Check if another app is using camera
- Try different camera ID in `app.py`: `cv2.VideoCapture(1)`

### Model not found?
- Verify MODEL_PATH in `app.py`
- Check if model file exists

### Port 5000 already in use?
- Change port in `app.py`: `app.run(port=5001)`

### Slow performance?
- Make sure GPU is detected
- Check CUDA installation
- Reduce batch size if needed

---

## 📁 File Structure

```
AURA/
├── app.py                          # Flask backend
├── templates/
│   └── index.html                  # Frontend UI
├── runs/
│   └── safety_equipment/
│       └── weights/
│           └── best.pt             # Your model
├── data/
│   └── dataset.yaml                # Dataset config
└── test dataset/                   # Test images
```

---

## 🎬 Demo Sequence

### Perfect 3-Minute Demo:

**Minute 1: Introduction**
- "AURA is an AI-powered safety equipment detection system"
- "It uses YOLOv8 trained on custom dataset"
- "Let me show you the web interface"

**Minute 2: Real-Time Detection**
- Switch to Real-Time mode
- Show camera detecting objects
- Point out FPS, accuracy, colored boxes
- "Running at 21 FPS on consumer GPU"

**Minute 3: Evaluation Results**
- Switch to Dataset Evaluation
- Click Start Evaluation
- Show progress bar
- Results appear: "31% accuracy, 67% precision"
- Show per-class table
- **"With improved training (100 epochs), we expect 50-60% accuracy"**

**Closing:**
- "Production-ready, real-time, GPU-accelerated"
- "Thank you!"

---

## 💡 Pro Tips

1. **Practice the demo** - Know where to click
2. **Have backup screenshots** - In case of technical issues
3. **Explain the metrics** - Accuracy, Precision, Recall
4. **Show the code** - If judges ask, show app.py
5. **Be honest** - Explain current limitations and future improvements

---

## 🌟 What Makes This Special

✨ **Dual-mode interface** - Both live and evaluation  
✨ **Professional UI** - Not a basic terminal app  
✨ **Real metrics** - Actual performance numbers  
✨ **GPU-accelerated** - Fast, production-ready  
✨ **Easy to use** - One click to evaluate  
✨ **Impressive visuals** - Judges will love it  

---

## 🚀 Run It Now!

```bash
python app.py
```

Then open: **http://localhost:5000**

**Good luck with your hackathon! 🏆**
