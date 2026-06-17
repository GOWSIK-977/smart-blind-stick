# app.py - Complete Working Code for Render - FIXED
import cv2
import numpy as np
import threading
import time
import queue
import warnings
import json
import os
import base64
import urllib.request
import socket
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import hashlib

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Check if running on Render
IS_RENDER = os.environ.get('RENDER', False)
PORT = int(os.environ.get('PORT', 5000))

# Store shared device views
device_views = {}

class SmartBlindStick:
    def __init__(self):
        print("\n" + "="*60)
        print("🦯 Initializing Smart Blind Stick System")
        print("="*60)
        
        # Generate unique device ID
        self.device_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        # Load YOLO model
        print("\n📷 Loading YOLO model...")
        self.model = None
        self.model_loaded = False
        
        try:
            model_path = 'yolov8n.pt'
            if not os.path.exists(model_path):
                print("   ⬇️ Downloading YOLO model (first time only)...")
                url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
                urllib.request.urlretrieve(url, model_path)
                print("   ✅ Model downloaded!")
            
            self.model = YOLO(model_path)
            
            # Test with dummy image
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(test_img, verbose=False)
            self.model_loaded = True
            print("✅ YOLO model loaded and tested successfully!")
            
        except Exception as e:
            print(f"❌ YOLO loading error: {e}")
            self.model = None
            self.model_loaded = False
        
        self.important_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck', 11: 'stop sign'
        }
        
        # State variables
        self.person_count = 0
        self.vehicle_count = 0
        self.detected_objects = []
        self.fps = 0
        self.emergency_mode = False
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.total_frames_processed = 0
        self.total_detections_found = 0
        
        # Current location
        self.current_location = {
            "lat": 11.2745,
            "lng": 77.5831,
            "address": "Perundurai, Tamil Nadu, India"
        }
        
        # Frame queue
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing = False
        self.last_detection = []
        
        # Start processing thread
        threading.Thread(target=self.process_frames, daemon=True).start()
        
        # Get network info
        self.local_ip = self.get_local_ip()
        
        print("\n" + "="*60)
        print("✅ SYSTEM READY!")
        print(f"   Device ID: {self.device_id}")
        print(f"   YOLO: {'✅ Loaded' if self.model_loaded else '❌ Not Available'}")
        print(f"   Mode: {'☁️ Cloud Mode' if IS_RENDER else '💻 Local Mode'}")
        print(f"   Port: {PORT}")
        print("="*60 + "\n")
    
    def get_local_ip(self):
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def process_frames(self):
        """Process frames in background"""
        self.processing = True
        print("🔍 Frame processing thread started")
        
        while self.processing:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    continue
                
                try:
                    image_bytes = base64.b64decode(frame_data)
                    np_arr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        continue
                    
                    self.total_frames_processed += 1
                    
                    # Resize for performance
                    height, width = frame.shape[:2]
                    if width > 480:
                        scale = 480 / width
                        new_width = 480
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Detect objects
                    detections = self.detect_with_yolo(frame)
                    
                    # Update stats
                    self.person_count = sum(1 for d in detections if d['class'] == 'person')
                    self.vehicle_count = sum(1 for d in detections if d['class'] in ['car', 'truck', 'bus', 'bicycle', 'motorcycle'])
                    self.detected_objects = detections
                    
                    if len(detections) > 0:
                        self.total_detections_found += len(detections)
                        print(f"✅ Detected {len(detections)} objects")
                    
                    # Calculate FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_frame_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_frame_time = current_time
                    
                    self.last_detection = detections
                    
                    # Store in shared view
                    self.store_shared_view()
                    
                except Exception as e:
                    print(f"⚠️ Frame processing error: {e}")
                    continue
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")
                continue
    
    def detect_with_yolo(self, frame):
        """Run YOLO detection"""
        detections = []
        height, width = frame.shape[:2]
        
        if self.model is None or not self.model_loaded:
            return detections
        
        try:
            results = self.model(frame, stream=True, conf=0.2, iou=0.45, verbose=False)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls not in self.important_classes or conf < 0.2:
                            continue
                        
                        class_name = self.important_classes[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate distance
                        box_height = y2 - y1
                        box_width = x2 - x1
                        area_ratio = (box_height * box_width) / (height * width)
                        
                        if area_ratio > 0.3:
                            distance = "very close"
                            distance_cm = 30
                        elif area_ratio > 0.15:
                            distance = "close"
                            distance_cm = 60
                        elif area_ratio > 0.05:
                            distance = "medium"
                            distance_cm = 120
                        else:
                            distance = "far"
                            distance_cm = 200
                        
                        center_x = (x1 + x2) / 2
                        if center_x < width * 0.3:
                            direction = "left"
                        elif center_x > width * 0.7:
                            direction = "right"
                        else:
                            direction = "center"
                        
                        detections.append({
                            'class': class_name,
                            'confidence': round(conf, 3),
                            'distance': distance,
                            'distance_cm': distance_cm,
                            'direction': direction
                        })
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
        
        return detections
    
    def store_shared_view(self):
        """Store current detection results for sharing"""
        global device_views
        
        data = {
            'device_id': self.device_id,
            'timestamp': datetime.now().isoformat(),
            'person_count': self.person_count,
            'vehicle_count': self.vehicle_count,
            'fps': self.fps,
            'detections': self.last_detection[:15],
            'location': self.current_location,
            'emergency': self.emergency_mode,
            'active': True,
            'last_update': time.time(),
            'model_loaded': self.model_loaded,
            'total_frames': self.total_frames_processed,
            'total_detections': self.total_detections_found
        }
        
        device_views[self.device_id] = data
        
        # Clean old entries
        current_time = time.time()
        for device_id in list(device_views.keys()):
            if current_time - device_views[device_id].get('last_update', 0) > 15:
                device_views[device_id]['active'] = False
    
    def get_current_data(self):
        """Get current detection data"""
        return {
            'device_id': self.device_id,
            'detections': self.last_detection[:15],
            'person_count': self.person_count,
            'vehicle_count': self.vehicle_count,
            'fps': self.fps,
            'emergency': self.emergency_mode,
            'location': self.current_location,
            'timestamp': datetime.now().isoformat(),
            'total_detections': len(self.last_detection),
            'debug': {
                'model_loaded': self.model_loaded,
                'frames_processed': self.total_frames_processed,
                'total_detections_found': self.total_detections_found,
                'queue_size': self.frame_queue.qsize()
            }
        }

# ============================================
# HTML TEMPLATE - FIXED VERSION
# ============================================
HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Smart Blind Stick</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 100%);
            min-height: 100vh;
            padding: 12px;
            color: #fff;
            touch-action: manipulation;
        }
        .container { max-width: 500px; margin: 0 auto; }
        
        .header {
            text-align: center;
            padding: 10px 0 15px 0;
        }
        .header h1 { 
            font-size: 22px; 
            background: linear-gradient(135deg, #4caf50, #2196f3); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }
        .header .subtitle { 
            font-size: 12px; 
            opacity: 0.6; 
            color: #888;
        }
        
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 6px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .badge {
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 10px;
            font-weight: 600;
        }
        .badge-success { background: rgba(76,175,80,0.2); border: 1px solid #4caf50; color: #4caf50; }
        .badge-danger { background: rgba(244,67,54,0.2); border: 1px solid #f44336; color: #f44336; }
        .badge-warning { background: rgba(255,193,7,0.2); border: 1px solid #ffc107; color: #ffc107; }
        .badge-info { background: rgba(33,150,243,0.2); border: 1px solid #2196f3; color: #2196f3; }
        
        .video-container {
            background: #000;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 12px;
            position: relative;
            aspect-ratio: 4/3;
            border: 1px solid rgba(255,255,255,0.1);
        }
        #video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            background: #111;
        }
        .video-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            display: flex;
            justify-content: space-between;
            pointer-events: none;
        }
        .video-overlay .left {
            background: rgba(0,0,0,0.7);
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 11px;
            color: #fff;
        }
        .video-overlay .right {
            background: rgba(0,0,0,0.7);
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 11px;
            color: #4caf50;
        }
        .video-overlay .fps { color: #4caf50; font-weight: bold; }
        .video-placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: rgba(255,255,255,0.5);
            font-size: 14px;
        }
        .video-placeholder .icon { font-size: 48px; margin-bottom: 10px; }
        
        .emergency-btn {
            background: linear-gradient(135deg, #ff4444, #cc0000);
            border: none;
            width: 100%;
            padding: 16px;
            border-radius: 50px;
            color: white;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            margin-bottom: 12px;
            animation: pulse 2s infinite;
            box-shadow: 0 4px 20px rgba(255,68,68,0.3);
            touch-action: manipulation;
        }
        @keyframes pulse {
            0%,100% { transform: scale(1); box-shadow: 0 4px 20px rgba(255,68,68,0.3); }
            50% { transform: scale(1.02); box-shadow: 0 4px 30px rgba(255,68,68,0.5); }
        }
        .emergency-btn:active { transform: scale(0.95); }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-bottom: 12px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            padding: 12px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .stat-value { font-size: 22px; font-weight: bold; color: #4caf50; }
        .stat-label { font-size: 11px; opacity: 0.6; margin-top: 3px; }
        
        .detection-list {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 12px;
            max-height: 180px;
            overflow-y: auto;
            margin-bottom: 12px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .detection-item {
            padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            display: flex;
            justify-content: space-between;
            font-size: 13px;
        }
        .detection-item:last-child { border-bottom: none; }
        .detection-item .emoji { margin-right: 8px; }
        .detection-item .conf { color: #4caf50; font-weight: bold; }
        .detection-item .distance-very-close { color: #f44336; }
        .detection-item .distance-close { color: #ff9800; }
        .detection-item .distance-medium { color: #ffc107; }
        .detection-item .distance-far { color: #4caf50; }
        
        .location-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 12px;
        }
        .location-card .label { font-size: 11px; opacity: 0.6; }
        .location-card .address { font-size: 14px; font-weight: 500; margin: 4px 0; }
        .location-card .coords { font-size: 12px; opacity: 0.7; }
        .location-card button {
            margin-top: 8px;
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            background: #4caf50;
            color: #fff;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
        }
        
        .controls {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }
        .controls button {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            background: rgba(255,255,255,0.1);
            color: white;
            touch-action: manipulation;
        }
        .controls button:active { transform: scale(0.95); }
        .controls .btn-camera { background: rgba(33,150,243,0.3); color: #2196f3; }
        .controls .btn-camera.active { background: rgba(76,175,80,0.3); color: #4caf50; }
        
        .dashboard-links {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 16px;
            margin: 12px 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .dashboard-links .title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #4caf50;
            text-align: center;
        }
        .dashboard-links .link-item {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .dashboard-links .link-item:last-child { margin-bottom: 0; }
        .dashboard-links .link-label {
            font-size: 11px;
            opacity: 0.6;
            margin-bottom: 4px;
        }
        .dashboard-links .link-url {
            font-size: 13px;
            font-family: monospace;
            word-break: break-all;
            color: #4caf50;
            background: rgba(0,0,0,0.2);
            padding: 6px 10px;
            border-radius: 6px;
            display: block;
        }
        .dashboard-links .copy-btn {
            background: rgba(76,175,80,0.2);
            border: 1px solid #4caf50;
            color: #4caf50;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            margin-top: 4px;
        }
        .dashboard-links .copy-btn:active { transform: scale(0.95); }
        .dashboard-links .device-info {
            font-size: 11px;
            opacity: 0.5;
            margin-top: 8px;
            text-align: center;
        }
        
        .debug-info {
            font-size: 10px;
            opacity: 0.4;
            text-align: center;
            padding: 8px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            font-family: monospace;
            margin-top: 8px;
        }
        
        .hidden { display: none; }
        
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦯 Smart Blind Stick</h1>
            <div class="subtitle">Real-time Object Detection & GPS Tracking</div>
        </div>
        
        <div class="status-bar">
            <span id="cameraStatus" class="badge badge-warning">📷 Click Start</span>
            <span id="gpsStatus" class="badge badge-warning">📍 GPS...</span>
            <span id="serverStatus" class="badge badge-success">🌐 Connected</span>
            <span id="modelStatus" class="badge badge-warning">🤖 Loading...</span>
        </div>
        
        <div class="dashboard-links">
            <div class="title">🔗 Connection Links</div>
            <div class="link-item">
                <div class="link-label">📱 Mobile / 💻 Laptop</div>
                <span class="link-url" id="mainUrl">Loading...</span>
                <button class="copy-btn" onclick="copyUrl('mainUrl')">📋 Copy</button>
            </div>
            <div class="device-info" id="deviceInfo">Device ID: Loading...</div>
        </div>
        
        <div class="video-container">
            <video id="video" autoplay playsinline muted></video>
            <div class="video-overlay">
                <span class="left" id="detectionOverlay">👤 0 | 🚗 0</span>
                <span class="right"><span class="fps" id="fpsOverlay">0</span> FPS</span>
            </div>
            <div id="videoPlaceholder" class="video-placeholder">
                <div class="icon">📷</div>
                <div>Tap "Start Camera" below</div>
                <div style="font-size:11px;margin-top:8px;opacity:0.5;">Camera access required</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn-camera" id="cameraBtn" onclick="toggleCamera()">📷 Start Camera</button>
            <button onclick="switchCamera()">🔄 Switch</button>
        </div>
        
        <button class="emergency-btn" onclick="sendEmergency()">🚨 EMERGENCY</button>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="personCount">0</div>
                <div class="stat-label">👤 Persons</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="vehicleCount">0</div>
                <div class="stat-label">🚗 Vehicles</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="fpsValue">0</div>
                <div class="stat-label">📊 FPS</div>
            </div>
        </div>
        
        <div class="detection-list" id="detectionList">
            <div style="text-align:center; opacity:0.5; padding:10px; font-size:13px;">🔍 Looking for objects...</div>
        </div>
        
        <div class="location-card">
            <div class="label">📍 Current Location</div>
            <div class="address" id="addressText">Getting location...</div>
            <div class="coords" id="coordsText">11.2745°N, 77.5831°E</div>
            <button onclick="openGoogleMaps()">🗺️ Open Google Maps</button>
        </div>
        
        <div class="debug-info" id="debugInfo">Model: Loading... | Frames: 0 | Detections: 0</div>
    </div>
    
    <script>
        // ============================================
        // CONFIGURATION
        // ============================================
        let video = document.getElementById('video');
        let stream = null;
        let isCameraOn = false;
        let facingMode = 'environment';
        let captureInterval = null;
        let lastFrameTime = Date.now();
        let frameCount = 0;
        let retryCount = 0;
        const MAX_RETRIES = 3;
        let isInitialized = false;
        
        // ============================================
        // COPY URL
        // ============================================
        function copyUrl(elementId) {
            const element = document.getElementById(elementId);
            const text = element.textContent;
            if (text && text !== 'Loading...' && text !== '') {
                navigator.clipboard.writeText(text).then(() => {
                    const btn = element.parentElement.querySelector('.copy-btn');
                    const originalText = btn.textContent;
                    btn.textContent = '✅ Copied!';
                    setTimeout(() => { btn.textContent = originalText; }, 2000);
                }).catch(() => {
                    const textArea = document.createElement('textarea');
                    textArea.value = text;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    alert('URL copied to clipboard!');
                });
            } else {
                alert('Please wait for URL to load...');
            }
        }
        
        // ============================================
        // UPDATE LINKS - IMMEDIATELY
        // ============================================
        function updateLinks() {
            const currentUrl = window.location.href;
            document.getElementById('mainUrl').textContent = currentUrl;
            
            fetch('/test')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('deviceInfo').textContent = 
                        `Device ID: ${data.device_id || 'Unknown'} | Model: ${data.model_loaded ? '✅' : '❌'}`;
                    if (data.model_loaded) {
                        document.getElementById('modelStatus').textContent = '🤖 Active';
                        document.getElementById('modelStatus').className = 'badge badge-success';
                    }
                })
                .catch(() => {
                    document.getElementById('deviceInfo').textContent = 'Device ID: Connected';
                });
        }
        
        // ============================================
        // CAMERA - With Retry Logic
        // ============================================
        async function startCamera() {
            try {
                document.getElementById('cameraStatus').textContent = '📷 Starting...';
                document.getElementById('cameraStatus').className = 'badge badge-warning';
                
                const constraints = {
                    video: {
                        facingMode: facingMode,
                        width: { ideal: 480 },
                        height: { ideal: 360 }
                    },
                    audio: false
                };
                
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                await video.play();
                
                isCameraOn = true;
                retryCount = 0;
                document.getElementById('videoPlaceholder').classList.add('hidden');
                document.getElementById('cameraBtn').textContent = '⏹️ Stop Camera';
                document.getElementById('cameraBtn').classList.add('active');
                document.getElementById('cameraStatus').textContent = '📷 Active';
                document.getElementById('cameraStatus').className = 'badge badge-success';
                
                startFrameCapture();
                console.log('📷 Camera started');
                
            } catch(err) {
                console.error('Camera error:', err);
                document.getElementById('cameraStatus').textContent = '❌ Error';
                document.getElementById('cameraStatus').className = 'badge badge-danger';
                
                if (retryCount < MAX_RETRIES) {
                    retryCount++;
                    console.log(`Retrying camera (${retryCount}/${MAX_RETRIES})...`);
                    setTimeout(startCamera, 2000);
                } else {
                    alert('Camera access denied. Please:\n1. Use HTTPS (Render URL)\n2. Allow camera permissions\n3. Try Chrome browser');
                    document.getElementById('cameraStatus').textContent = '📷 Click Start';
                    document.getElementById('cameraStatus').className = 'badge badge-warning';
                }
            }
        }
        
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            video.srcObject = null;
            isCameraOn = false;
            
            if (captureInterval) {
                clearInterval(captureInterval);
                captureInterval = null;
            }
            
            document.getElementById('videoPlaceholder').classList.remove('hidden');
            document.getElementById('cameraBtn').textContent = '📷 Start Camera';
            document.getElementById('cameraBtn').classList.remove('active');
            document.getElementById('cameraStatus').textContent = '📷 Stopped';
            document.getElementById('cameraStatus').className = 'badge badge-warning';
        }
        
        function toggleCamera() {
            if (isCameraOn) {
                stopCamera();
            } else {
                startCamera();
            }
        }
        
        function switchCamera() {
            facingMode = (facingMode === 'environment') ? 'user' : 'environment';
            if (isCameraOn) {
                stopCamera();
                setTimeout(startCamera, 500);
            }
        }
        
        // ============================================
        // FRAME CAPTURE
        // ============================================
        function startFrameCapture() {
            if (captureInterval) clearInterval(captureInterval);
            
            const canvas = document.createElement('canvas');
            canvas.width = 480;
            canvas.height = 360;
            const ctx = canvas.getContext('2d');
            
            captureInterval = setInterval(() => {
                if (!isCameraOn || video.readyState !== video.HAVE_ENOUGH_DATA) {
                    return;
                }
                
                try {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const imageData = canvas.toDataURL('image/jpeg', 0.8);
                    
                    fetch('/process_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData })
                    })
                    .then(res => res.json())
                    .then(data => {
                        updateUI(data);
                        if (data.debug) {
                            document.getElementById('debugInfo').textContent = 
                                `Model: ${data.debug.model_loaded ? '✅ Loaded' : '❌ Not Loaded'} | ` +
                                `Frames: ${data.debug.frames_processed} | ` +
                                `Detections: ${data.debug.total_detections_found}`;
                        }
                    })
                    .catch(err => console.error('Send error:', err));
                    
                    frameCount++;
                    const now = Date.now();
                    if (now - lastFrameTime >= 1000) {
                        document.getElementById('fpsOverlay').textContent = frameCount;
                        frameCount = 0;
                        lastFrameTime = now;
                    }
                    
                } catch(e) {
                    console.error('Frame capture error:', e);
                }
            }, 200);
            
            console.log('📷 Frame capture started');
        }
        
        // ============================================
        // GPS
        // ============================================
        let watchId = null;
        let currentLocation = { lat: 11.2745, lng: 77.5831 };
        
        function startGPS() {
            if (!navigator.geolocation) {
                document.getElementById('gpsStatus').textContent = '❌ Not Supported';
                document.getElementById('gpsStatus').className = 'badge badge-danger';
                return;
            }
            
            document.getElementById('gpsStatus').textContent = '📍 Getting...';
            document.getElementById('gpsStatus').className = 'badge badge-warning';
            
            watchId = navigator.geolocation.watchPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    currentLocation = { lat, lng };
                    
                    document.getElementById('coordsText').textContent = 
                        `${lat.toFixed(6)}°N, ${lng.toFixed(6)}°E`;
                    document.getElementById('gpsStatus').textContent = '📍 Active';
                    document.getElementById('gpsStatus').className = 'badge badge-success';
                    
                    fetch('/location', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ lat, lng })
                    }).catch(() => {});
                    
                    reverseGeocode(lat, lng);
                },
                (error) => {
                    console.error('GPS Error:', error);
                    document.getElementById('gpsStatus').textContent = '📍 Fallback';
                    document.getElementById('gpsStatus').className = 'badge badge-warning';
                    document.getElementById('coordsText').textContent = '11.2745°N, 77.5831°E';
                },
                { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
            );
        }
        
        function reverseGeocode(lat, lng) {
            const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=18&addressdetails=1`;
            fetch(url)
                .then(res => res.json())
                .then(data => {
                    if (data && data.display_name) {
                        document.getElementById('addressText').textContent = data.display_name;
                    }
                })
                .catch(() => {
                    document.getElementById('addressText').textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                });
        }
        
        function openGoogleMaps() {
            const url = `https://www.google.com/maps?q=${currentLocation.lat},${currentLocation.lng}`;
            window.open(url, '_blank');
        }
        
        // ============================================
        // UI UPDATE
        // ============================================
        function updateUI(data) {
            if (!data) return;
            
            document.getElementById('serverStatus').textContent = '🌐 Connected';
            document.getElementById('serverStatus').className = 'badge badge-success';
            
            if (data.person_count !== undefined) {
                document.getElementById('personCount').textContent = data.person_count;
                document.getElementById('detectionOverlay').textContent = `👤 ${data.person_count} | 🚗 ${data.vehicle_count || 0}`;
            }
            if (data.vehicle_count !== undefined) {
                document.getElementById('vehicleCount').textContent = data.vehicle_count;
            }
            if (data.fps !== undefined && data.fps > 0) {
                document.getElementById('fpsValue').textContent = data.fps;
            }
            
            if (data.detections && data.detections.length > 0) {
                let html = '';
                data.detections.forEach(d => {
                    const emoji = d.class === 'person' ? '👤' : 
                                  (d.class.includes('car') || d.class.includes('vehicle') ? '🚗' : 
                                  (d.class === 'bicycle' ? '🚲' : '📦'));
                    const distanceClass = `distance-${d.distance.replace(' ', '-')}`;
                    html += `<div class="detection-item">
                        <span><span class="emoji">${emoji}</span> ${d.class}</span>
                        <span>
                            <span class="${distanceClass}">${d.distance}</span>
                            <span class="conf">${Math.round(d.confidence * 100)}%</span>
                        </span>
                    </div>`;
                });
                document.getElementById('detectionList').innerHTML = html;
            } else {
                document.getElementById('detectionList').innerHTML = 
                    '<div style="text-align:center; opacity:0.5; padding:10px; font-size:13px;">🔍 Looking for objects...</div>';
            }
            
            if (data.location) {
                document.getElementById('coordsText').textContent = 
                    `${data.location.lat.toFixed(6)}°N, ${data.location.lng.toFixed(6)}°E`;
                if (data.location.address) {
                    document.getElementById('addressText').textContent = data.location.address;
                }
            }
        }
        
        // ============================================
        // EMERGENCY
        // ============================================
        async function sendEmergency() {
            if (!confirm('🚨 Send emergency alert with your location?')) return;
            
            try {
                document.querySelector('.emergency-btn').textContent = '🚨 SENDING...';
                document.querySelector('.emergency-btn').disabled = true;
                
                const response = await fetch('/emergency', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'success') {
                    alert('🚨 Emergency alert sent successfully!');
                    if (navigator.vibrate) navigator.vibrate([500, 300, 500]);
                    setTimeout(() => {
                        if (confirm('🚨 Open Google Maps for location?')) {
                            openGoogleMaps();
                        }
                    }, 1000);
                }
            } catch(e) {
                console.error('Emergency error:', e);
                alert('Network error sending emergency');
            } finally {
                document.querySelector('.emergency-btn').textContent = '🚨 EMERGENCY';
                document.querySelector('.emergency-btn').disabled = false;
            }
        }
        
        // ============================================
        // POLL SERVER
        // ============================================
        function pollServer() {
            fetch('/stats')
                .then(res => res.json())
                .then(data => {
                    if (data.person_count !== undefined) {
                        document.getElementById('personCount').textContent = data.person_count;
                        document.getElementById('detectionOverlay').textContent = `👤 ${data.person_count} | 🚗 ${data.vehicle_count || 0}`;
                    }
                    if (data.vehicle_count !== undefined) {
                        document.getElementById('vehicleCount').textContent = data.vehicle_count;
                    }
                    if (data.fps !== undefined && data.fps > 0) {
                        document.getElementById('fpsValue').textContent = data.fps;
                    }
                    
                    if (data.debug) {
                        document.getElementById('debugInfo').textContent = 
                            `Model: ${data.debug.model_loaded ? '✅ Loaded' : '❌ Not Loaded'} | ` +
                            `Frames: ${data.debug.frames_processed} | ` +
                            `Detections: ${data.debug.total_detections_found}`;
                    }
                })
                .catch(() => {});
        }
        
        // ============================================
        // INITIALIZATION
        // ============================================
        function init() {
            if (isInitialized) return;
            isInitialized = true;
            
            // Update links immediately
            setTimeout(updateLinks, 500);
            
            // Start GPS
            setTimeout(startGPS, 1000);
            
            // Don't auto-start camera - let user click
            document.getElementById('cameraStatus').textContent = '📷 Click Start';
            document.getElementById('cameraStatus').className = 'badge badge-warning';
            
            // Poll server every 2 seconds
            setInterval(pollServer, 2000);
            
            // Update links periodically
            setInterval(updateLinks, 10000);
            
            console.log('✅ System initialized');
            console.log('📱 Open on mobile:', window.location.href);
        }
        
        // Start when page loads
        document.addEventListener('DOMContentLoaded', init);
        
        // Also try on load
        window.addEventListener('load', function() {
            setTimeout(updateLinks, 100);
        });
        
        window.addEventListener('beforeunload', () => {
            if (captureInterval) clearInterval(captureInterval);
            if (watchId) navigator.geolocation.clearWatch(watchId);
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        });
    </script>
</body>
</html>
'''

# ============================================
# FLASK ROUTES
# ============================================

blind_stick = None

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/stats')
def stats():
    if blind_stick:
        return jsonify(blind_stick.get_current_data())
    return jsonify({'person_count': 0, 'vehicle_count': 0, 'fps': 0})

@app.route('/devices')
def get_devices():
    if blind_stick:
        return jsonify({'devices': device_views})
    return jsonify({'devices': {}})

@app.route('/view/<device_id>')
def view_device(device_id):
    if device_id in device_views:
        return jsonify(device_views[device_id])
    return jsonify({'error': 'Device not found'}), 404

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if not blind_stick:
        return jsonify({'error': 'System not ready'}), 503
    
    try:
        data = request.json
        image_data = data.get('image', '')
        
        if image_data and ',' in image_data:
            image_data = image_data.split(',')[1]
            if blind_stick.frame_queue.qsize() < 10:
                blind_stick.frame_queue.put(image_data)
        
        return jsonify(blind_stick.get_current_data())
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/emergency', methods=['POST'])
def emergency():
    if blind_stick:
        blind_stick.emergency_mode = True
        
        print(f"\n{'='*60}")
        print("🚨 EMERGENCY ALERT!")
        print(f"📍 Location: {blind_stick.current_location['address']}")
        print(f"📍 Coords: {blind_stick.current_location['lat']}, {blind_stick.current_location['lng']}")
        print(f"👥 Persons: {blind_stick.person_count}")
        print(f"{'='*60}\n")
        
        def reset_emergency():
            time.sleep(30)
            blind_stick.emergency_mode = False
        
        threading.Thread(target=reset_emergency, daemon=True).start()
        
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error'}), 500

@app.route('/location', methods=['POST'])
def update_location():
    if blind_stick:
        try:
            data = request.json
            lat = data.get('lat')
            lng = data.get('lng')
            
            if lat is not None and lng is not None:
                blind_stick.current_location = {
                    "lat": lat,
                    "lng": lng,
                    "address": f"{lat:.6f}, {lng:.6f}"
                }
                return jsonify({'status': 'success'})
        except Exception as e:
            print(f"Location error: {e}")
    return jsonify({'status': 'error'}), 500

@app.route('/test')
def test():
    return jsonify({
        'status': 'running',
        'is_render': IS_RENDER,
        'local_ip': blind_stick.local_ip if blind_stick else '127.0.0.1',
        'port': PORT,
        'model_loaded': blind_stick.model_loaded if blind_stick else False,
        'device_id': blind_stick.device_id if blind_stick else 'Unknown',
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    blind_stick = SmartBlindStick()
    
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER...")
    print("="*60)
    
    if IS_RENDER:
        print("☁️ Render Cloud Deployment")
        print("📱 Open on mobile: https://your-app.onrender.com")
        print("💻 Open on laptop: https://your-app.onrender.com")
    else:
        print(f"📱 Open on mobile: http://{blind_stick.local_ip}:{PORT}")
        print(f"💻 Open on laptop: http://{blind_stick.local_ip}:{PORT}")
    
    print("\n💡 HOW TO USE:")
    print("   1. Open the URL on your mobile")
    print("   2. Click 'Start Camera'")
    print("   3. Grant camera permission")
    print("   4. Point at objects to detect")
    print("="*60 + "\n")
    
    try:
        # Bind to 0.0.0.0 to accept connections from anywhere
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if blind_stick:
            blind_stick.processing = False
