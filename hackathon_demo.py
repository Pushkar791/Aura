#!/usr/bin/env python3
"""
AURA Hackathon Demo Launcher
Easy-to-use launcher for hackathon presentations and demonstrations
"""

import sys
import os
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    """Print AURA banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       🚀 AURA - Autonomous Utility Recognition Assistant     ║
    ║                                                              ║
    ║         Real-time Safety Equipment Detection System          ║
    ║              Perfect for Hackathon Demonstrations            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['flask', 'flask_socketio', 'ultralytics', 'cv2', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'flask_socketio':
                import flask_socketio
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install flask flask-socketio ultralytics opencv-python numpy")
        return False
    
    print("✅ All dependencies installed!")
    return True

def launch_demo():
    """Launch the AURA web demo"""
    print("\n🚀 Launching AURA Web Demo...")
    print("=" * 60)
    print("🌐 Dashboard URL: http://localhost:5000")
    print("📱 Demo Mode: Perfect for presentations without camera")
    print("🎥 Camera Mode: Uses your webcam for real detection")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Launch web application
    try:
        # Change to AURA directory
        os.chdir(Path(__file__).parent)
        
        # Start the web application
        print("\n🔧 Starting web server...")
        
        # Auto-open browser after a short delay
        def open_browser():
            time.sleep(3)
            try:
                webbrowser.open('http://localhost:5000')
                print("🌐 Browser opened automatically")
            except:
                print("🌐 Please open http://localhost:5000 manually")
        
        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Import and run the web app
        from web_app import socketio, app
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False,
                    allow_unsafe_werkzeug=True)
                    
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching demo: {e}")
        print("\nTry running manually:")
        print("  python web_app.py")

def show_hackathon_tips():
    """Show tips for hackathon presentations"""
    tips = """
    💡 HACKATHON PRESENTATION TIPS:
    
    🎯 DEMO MODE (Recommended for presentations):
       • No camera required - perfect for stage demos
       • Simulates real-time detection with fake data
       • Consistent and reliable for presentations
       • Click "Demo Mode" button in the web interface
    
    📷 CAMERA MODE (For real demonstrations):
       • Uses your webcam for actual detection
       • Shows real-time safety equipment detection
       • May need good lighting and clear camera view
       • Click "Start Detection" button
    
    🎨 FEATURES TO HIGHLIGHT:
       • Real-time detection at 30 FPS
       • 7 types of safety equipment detection
       • Priority-based alert system
       • Performance metrics and analytics
       • Modern web interface with live updates
       • Falcon simulation integration ready
    
    🚀 QUICK START:
       1. Run: python hackathon_demo.py
       2. Open: http://localhost:5000
       3. Click "Demo Mode" for presentations
       4. Show the live detection feed and alerts
       5. Highlight the equipment status grid
    
    📊 KEY TALKING POINTS:
       • Space station safety equipment monitoring
       • AI-powered computer vision with YOLOv8
       • Real-time alerts for critical equipment
       • Synthetic data training pipeline
       • Scalable web-based monitoring system
    """
    print(tips)

def main():
    """Main demo launcher"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first!")
        print("Run: pip install -r requirements.txt")
        return 1
    
    # Show hackathon tips
    show_hackathon_tips()
    
    # Ask user what they want to do
    print("\n" + "="*60)
    print("CHOOSE YOUR OPTION:")
    print("1. 🚀 Launch Web Demo (Recommended)")
    print("2. 💡 Show Hackathon Tips Again")  
    print("3. ❌ Exit")
    print("="*60)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                launch_demo()
                break
            elif choice == '2':
                show_hackathon_tips()
                continue
            elif choice == '3':
                print("👋 Good luck with your hackathon!")
                break
            else:
                print("❌ Please enter 1, 2, or 3")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    return 0

if __name__ == "__main__":
    sys.exit(main())