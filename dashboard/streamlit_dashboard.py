"""
AURA Streamlit Dashboard
Real-time visualization dashboard for safety equipment detection
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.camera_interface import AURACameraInterface
from src.detection_engine import AURADetectionEngine
from configs.config import config, SAFETY_CLASSES, CLASS_COLORS, PRIORITY_LEVELS

class AURADashboard:
    """AURA Real-time Dashboard"""
    
    def __init__(self):
        self.camera = None
        self.detection_history = []
        self.max_history = 100
        self.last_update = time.time()
        
        # Initialize session state
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = False
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        if 'detection_stats' not in st.session_state:
            st.session_state.detection_stats = []
    
    def initialize_camera(self):
        """Initialize camera interface"""
        try:
            self.camera = AURACameraInterface(camera_index=0)
            if self.camera.initialize():
                self.camera.set_detection_callback(self._detection_callback)
                st.session_state.dashboard_initialized = True
                return True
            else:
                st.error("Failed to initialize camera")
                return False
        except Exception as e:
            st.error(f"Camera initialization error: {str(e)}")
            return False
    
    def _detection_callback(self, frame, detections):
        """Callback for new detections"""
        current_time = time.time()
        detection_data = {
            'timestamp': current_time,
            'frame_id': len(self.detection_history),
            'detections': detections,
            'detection_count': len(detections)
        }
        
        # Add class-specific counts
        class_counts = {class_name: 0 for class_name in SAFETY_CLASSES.values()}
        for detection in detections:
            class_counts[detection.class_name] += 1
        detection_data.update(class_counts)
        
        self.detection_history.append(detection_data)
        
        # Maintain history limit
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        # Update session state
        st.session_state.detection_stats = self.detection_history.copy()
        self.last_update = current_time
    
    def start_camera(self):
        """Start camera capture"""
        if self.camera and not st.session_state.camera_running:
            self.camera.start()
            st.session_state.camera_running = True
            st.success("Camera started successfully")
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.camera and st.session_state.camera_running:
            self.camera.stop()
            st.session_state.camera_running = False
            st.success("Camera stopped")
    
    def shutdown(self):
        """Shutdown dashboard"""
        if self.camera:
            self.camera.shutdown()
        st.session_state.dashboard_initialized = False
        st.session_state.camera_running = False

def create_detection_chart(detection_history):
    """Create detection timeline chart"""
    if not detection_history:
        return go.Figure()
    
    df = pd.DataFrame(detection_history)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Total Detections Over Time', 'Detections by Class'],
        vertical_spacing=0.1
    )
    
    # Total detections over time
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['detection_count'],
            mode='lines+markers',
            name='Total Detections',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # Detections by class
    class_names = list(SAFETY_CLASSES.values())
    for class_name in class_names:
        if class_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=df[class_name],
                    mode='lines',
                    name=class_name,
                    stackgroup='classes'
                ),
                row=2, col=1
            )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="AURA Detection Analytics",
        title_x=0.5
    )
    
    return fig

def create_class_distribution_chart(detection_history):
    """Create class distribution pie chart"""
    if not detection_history:
        return go.Figure()
    
    # Aggregate class counts
    class_totals = {class_name: 0 for class_name in SAFETY_CLASSES.values()}
    
    for entry in detection_history:
        for class_name in SAFETY_CLASSES.values():
            if class_name in entry:
                class_totals[class_name] += entry[class_name]
    
    # Filter out classes with zero detections
    filtered_totals = {k: v for k, v in class_totals.items() if v > 0}
    
    if not filtered_totals:
        return go.Figure()
    
    fig = go.Figure(data=[go.Pie(
        labels=list(filtered_totals.keys()),
        values=list(filtered_totals.values()),
        hole=0.3
    )])
    
    fig.update_layout(
        title_text="Equipment Detection Distribution",
        title_x=0.5
    )
    
    return fig

def create_alerts_table(detections):
    """Create alerts table for high-priority detections"""
    if not detections:
        return pd.DataFrame()
    
    alerts = []
    current_time = datetime.now()
    
    for detection in detections:
        if detection.priority == "CRITICAL" and detection.confidence >= config.get_alert_threshold(detection.class_name):
            alerts.append({
                'Time': current_time.strftime('%H:%M:%S'),
                'Equipment': detection.class_name,
                'Confidence': f"{detection.confidence:.2f}",
                'Priority': detection.priority,
                'Location': f"({detection.center[0]}, {detection.center[1]})"
            })
    
    return pd.DataFrame(alerts)

def main():
    """Main dashboard function"""
    st.set_page_config(
        page_title="AURA - Autonomous Utility Recognition Assistant",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .status-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        background-color: #f8f9fa;
        text-align: center;
    }
    .critical-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🚀 AURA Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**Autonomous Utility Recognition Assistant** - Real-time Safety Equipment Detection")
    
    # Initialize dashboard
    if 'dashboard_obj' not in st.session_state:
        st.session_state.dashboard_obj = AURADashboard()
    
    dashboard = st.session_state.dashboard_obj
    
    # Sidebar Controls
    st.sidebar.header("System Controls")
    
    if not st.session_state.dashboard_initialized:
        if st.sidebar.button("🔧 Initialize System"):
            with st.spinner("Initializing camera and detection engine..."):
                if dashboard.initialize_camera():
                    st.sidebar.success("System initialized!")
                    st.rerun()
    else:
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if not st.session_state.camera_running:
                if st.button("▶️ Start"):
                    dashboard.start_camera()
                    st.rerun()
            else:
                if st.button("⏸️ Stop"):
                    dashboard.stop_camera()
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset"):
                dashboard.shutdown()
                st.rerun()
    
    # Sidebar Configuration
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    if dashboard.camera and dashboard.camera.detection_engine:
        dashboard.camera.detection_engine.config.model.conf_threshold = conf_threshold
        dashboard.camera.detection_engine.config.model.iou_threshold = iou_threshold
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        refresh_rate = st.sidebar.selectbox("Refresh Rate (seconds)", [0.5, 1.0, 2.0, 5.0], index=1)
    
    # Main Dashboard
    if not st.session_state.camera_running:
        st.warning("⚠️ Camera not running. Please start the system to begin detection.")
        return
    
    # Live Video Feed
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎥 Live Detection Feed")
        video_placeholder = st.empty()
        
        if dashboard.camera:
            frame = dashboard.camera.get_processed_frame()
            if frame is not None:
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    with col2:
        st.subheader("📊 System Status")
        
        # System stats
        if dashboard.camera:
            stats = dashboard.camera.get_stats()
            
            st.metric("Camera FPS", f"{stats['camera_fps']:.1f}")
            st.metric("Total Frames", stats['total_frames'])
            st.metric("Queue Size", stats['queue_size'])
            
            if 'detection_stats' in stats:
                detection_stats = stats['detection_stats']
                st.metric("Detection FPS", f"{detection_stats.get('fps', 0):.1f}")
                st.metric("Total Detections", detection_stats.get('total_detections', 0))
        
        # Current detections
        st.subheader("🎯 Current Detections")
        if dashboard.camera:
            current_detections = dashboard.camera.get_current_detections()
            
            if current_detections:
                for detection in current_detections:
                    priority_color = "🔴" if detection.priority == "CRITICAL" else "🟡"
                    st.write(f"{priority_color} **{detection.class_name}** ({detection.confidence:.2f})")
            else:
                st.write("No detections currently")
    
    # Analytics Section
    st.subheader("📈 Detection Analytics")
    
    if st.session_state.detection_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            # Detection timeline
            chart = create_detection_chart(st.session_state.detection_stats)
            st.plotly_chart(chart, use_container_width=True)
        
        with col2:
            # Class distribution
            pie_chart = create_class_distribution_chart(st.session_state.detection_stats)
            st.plotly_chart(pie_chart, use_container_width=True)
    
    # Alerts Section
    st.subheader("🚨 Critical Alerts")
    
    if dashboard.camera:
        current_detections = dashboard.camera.get_current_detections()
        alerts_df = create_alerts_table(current_detections)
        
        if not alerts_df.empty:
            st.dataframe(alerts_df, use_container_width=True)
            
            # Sound alert for critical detections
            critical_count = len(alerts_df)
            if critical_count > 0:
                st.error(f"⚠️ {critical_count} CRITICAL equipment detected!")
        else:
            st.success("✅ No critical alerts")
    
    # Equipment Status Grid
    st.subheader("🛠️ Equipment Status Overview")
    
    # Create equipment status grid
    cols = st.columns(4)
    col_idx = 0
    
    for class_id, class_name in SAFETY_CLASSES.items():
        with cols[col_idx % 4]:
            # Count current detections for this class
            current_count = 0
            if dashboard.camera:
                current_detections = dashboard.camera.get_current_detections()
                current_count = sum(1 for d in current_detections if d.class_name == class_name)
            
            # Determine status
            status = "🟢 OK" if current_count > 0 else "🔴 NOT DETECTED"
            priority = PRIORITY_LEVELS.get(class_name, "MEDIUM")
            
            st.markdown(f"""
            <div class="status-card">
                <h4>{class_name}</h4>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Count:</strong> {current_count}</p>
                <p><strong>Priority:</strong> {priority}</p>
            </div>
            """, unsafe_allow_html=True)
        
        col_idx += 1
    
    # Footer
    st.markdown("---")
    st.markdown("**AURA** - Powered by YOLOv8 | Real-time Space Station Safety Monitoring")
    
    # Auto-refresh
    if auto_refresh and st.session_state.camera_running:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()