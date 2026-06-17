# app.py - Fixed for Render Deployment
import cv2
import numpy as np
import threading
import time
import queue
import warnings
import json
import requests
import os
import base64
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import math

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Check if running on Render/Cloud
IS_RENDER = os.environ.get('RENDER', False)
PORT = int(os.environ.get('PORT', 5000))

class SmartBlindStick:
    def __init__(self):
        print("\n" + "="*60)
        print("🦯 Initializing Smart Blind Stick System")
        print("="*60)
        
        # Download and load YOLO model
        print("\n📷 Loading YOLO model...")
        try:
            # Download model if not exists
            model_path = 'yolov8n.pt'
            if not os.path.exists(model_path):
                print("   Downloading YOLO model (first time only)...")
                import urllib.request
                url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
                urllib.request.urlretrieve(url, model_path)
                print("   ✅ Model downloaded!")
            
            self.model = YOLO(model_path)
            print("✅ YOLO model loaded!")
        except Exception as e:
            print(f"⚠️ YOLO not available: {e}")
            self.model = None
        
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
        
        # Current location (demo/fallback)
        self.current_location = {
            "lat": 11.2745,
            "lng": 77.5831,
            "address": "Perundurai, Tamil Nadu, India",
            "source": "demo"
        }
        
        # Frame queue for processing
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.processing = False
        self.last_detection = []
        self.last_update_time = time.time()
        
        # Start processing thread
        threading.Thread(target=self.process_frames, daemon=True).start()
        
        print("\n" + "="*60)
        print("✅ SYSTEM READY!")
        print(f"   YOLO: {'✅ Loaded' if self.model else '⚠️ Not Available'}")
        print(f"   Mode: {'☁️ Cloud Mode' if IS_RENDER else '💻 Local Mode'}")
        print("="*60 + "\n")
    
    def process_frames(self):
        """Process frames in background"""
        self.processing = True
        
        while self.processing:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=0.5)
                if frame_data is None:
                    continue
                
                # Decode image
                try:
                    image_bytes = base64.b64decode(frame_data)
                    np_arr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        continue
                    
                    # Process with YOLO
                    detections = self.detect_with_yolo(frame)
                    
                    # Update stats
                    self.person_count = sum(1 for d in detections if d['class'] == 'person')
                    self.vehicle_count = sum(1 for d in detections if d['class'] in ['car', 'truck', 'bus', 'bicycle', 'motorcycle'])
                    self.detected_objects = detections
                    
                    # Calculate FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_frame_time > 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_frame_time = current_time
                    
                    # Store result
                    self.last_detection = detections
                    self.last_update_time = time.time()
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Unexpected error in process_frames: {e}")
                continue
    
    def detect_with_yolo(self, frame):
        """Run YOLO detection on frame"""
        detections = []
        height, width = frame.shape[:2]
        
        if self.model is None:
            return detections
        
        try:
            results = self.model(frame, stream=True, conf=0.3, verbose=False)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if conf < 0.3:
                            continue
                        
                        class_name = self.important_classes.get(cls, f"object_{cls}")
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate distance based on box size
                        box_height = y2 - y1
                        
                        if box_height > height * 0.5:
                            distance = "very close"
                            distance_cm = 30
                        elif box_height > height * 0.3:
                            distance = "close"
                            distance_cm = 60
                        elif box_height > height * 0.15:
                            distance = "medium"
                            distance_cm = 120
                        else:
                            distance = "far"
                            distance_cm = 200
                        
                        # Determine direction
                        center_x = (x1 + x2) / 2
                        if center_x < width * 0.3:
                            direction = "left"
                        elif center_x > width * 0.7:
                            direction = "right"
                        else:
                            direction = "center"
                        
                        detection = {
                            'class': class_name,
                            'confidence': conf,
                            'distance': distance,
                            'distance_cm': distance_cm,
                            'direction': direction
                        }
                        detections.append(detection)
            
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detections
    
    def get_current_data(self):
        """Get current detection data"""
        return {
            'detections': self.last_detection[:15],
            'person_count': self.person_count,
            'vehicle_count': self.vehicle_count,
            'fps': self.fps,
            'emergency': self.emergency_mode,
            'location': self.current_location,
            'timestamp': datetime.now().isoformat(),
            'total_detections': len(self.last_detection)
        }

# ============================================
# HTML TEMPLATE - Fixed for Render
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
        .header h1 { font-size: 22px; background: linear-gradient(135deg, #4caf50, #2196f3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header .subtitle { font-size: 12px; opacity: 0.6; color: #888; }
        
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
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
        
        .hidden { display: none; }
        
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🦯 Smart Blind Stick</h1>
            <div class="subtitle">Real-time Object Detection & GPS Tracking</div>
        </div>
        
        <!-- Status -->
        <div class="status-bar">
            <span id="cameraStatus" class="badge badge-warning">📷 Starting...</span>
            <span id="gpsStatus" class="badge badge-warning">📍 GPS...</span>
            <span id="serverStatus" class="badge badge-warning">🌐 Connecting...</span>
        </div>
        
        <!-- Video Container -->
        <div class="video-container">
            <video id="video" autoplay playsinline muted></video>
            <div class="video-overlay">
                <span class="left" id="detectionOverlay">👤 0 | 🚗 0</span>
                <span class="right"><span class="fps" id="fpsOverlay">0</span> FPS</span>
            </div>
            <div id="videoPlaceholder" class="video-placeholder">
                <div class="icon">📷</div>
                <div>Starting camera...</div>
                <div style="font-size:11px;margin-top:8px;opacity:0.5;">Tap "Start Camera" below</div>
            </div>
        </div>
        
        <!-- Controls -->
        <div class="controls">
            <button class="btn-camera" id="cameraBtn" onclick="toggleCamera()">📷 Start Camera</button>
            <button onclick="switchCamera()">🔄 Switch</button>
        </div>
        
        <!-- Emergency Button -->
        <button class="emergency-btn" onclick="sendEmergency()">🚨 EMERGENCY</button>
        
        <!-- Stats -->
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
        
        <!-- Detections -->
        <div class="detection-list" id="detectionList">
            <div style="text-align:center; opacity:0.5; padding:10px; font-size:13px;">No objects detected</div>
        </div>
        
        <!-- Location -->
        <div class="location-card">
            <div class="label">📍 Current Location</div>
            <div class="address" id="addressText">Getting location...</div>
            <div class="coords" id="coordsText">11.2745°N, 77.5831°E</div>
            <div style="margin-top:8px;display:flex;gap:8px;">
                <button onclick="openGoogleMaps()" style="flex:1;padding:8px;border:none;border-radius:8px;background:#4caf50;color:#fff;font-size:12px;font-weight:600;cursor:pointer;">🗺️ Open Maps</button>
            </div>
        </div>
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
        let serverUrl = window.location.origin;
        
        // ============================================
        // CAMERA
        // ============================================
        async function startCamera() {
            try {
                const constraints = {
                    video: {
                        facingMode: facingMode,
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    },
                    audio: false
                };
                
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                await video.play();
                
                isCameraOn = true;
                document.getElementById('videoPlaceholder').classList.add('hidden');
                document.getElementById('cameraBtn').textContent = '⏹️ Stop Camera';
                document.getElementById('cameraBtn').classList.add('active');
                document.getElementById('cameraStatus').textContent = '📷 Active';
                document.getElementById('cameraStatus').className = 'badge badge-success';
                
                // Start sending frames
                startFrameCapture();
                
                console.log('📷 Camera started');
                
            } catch(err) {
                console.error('Camera error:', err);
                document.getElementById('cameraStatus').textContent = '❌ Camera Error';
                document.getElementById('cameraStatus').className = 'badge badge-danger';
                alert('Camera access denied. Please allow camera permissions.');
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
            
            console.log('📷 Camera stopped');
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
            if (captureInterval) {
                clearInterval(captureInterval);
            }
            
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            const ctx = canvas.getContext('2d');
            
            captureInterval = setInterval(() => {
                if (!isCameraOn || video.readyState !== video.HAVE_ENOUGH_DATA) {
                    return;
                }
                
                try {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Compress to JPEG
                    const imageData = canvas.toDataURL('image/jpeg', 0.6);
                    
                    // Send to server via HTTP POST (more reliable than WebSocket on Render)
                    fetch('/process_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData })
                    }).then(res => res.json())
                      .then(data => updateUI(data))
                      .catch(err => console.error('Frame send error:', err));
                    
                    // Update FPS
                    frameCount++;
                    const now = Date.now();
                    if (now - lastFrameTime >= 1000) {
                        const fps = frameCount;
                        document.getElementById('fpsOverlay').textContent = fps;
                        frameCount = 0;
                        lastFrameTime = now;
                    }
                    
                } catch(e) {
                    console.error('Frame capture error:', e);
                }
            }, 150); // ~7 FPS for better performance
            
            console.log('📷 Frame capture started');
        }
        
        // ============================================
        // GPS TRACKING
        // ============================================
        let watchId = null;
        let currentLocation = { lat: 11.2745, lng: 77.5831 };
        
        function startGPS() {
            if (!navigator.geolocation) {
                document.getElementById('gpsStatus').textContent = '❌ GPS Not Supported';
                document.getElementById('gpsStatus').className = 'badge badge-danger';
                return;
            }
            
            watchId = navigator.geolocation.watchPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    const accuracy = position.coords.accuracy;
                    
                    currentLocation = { lat, lng };
                    
                    document.getElementById('coordsText').textContent = 
                        `${lat.toFixed(6)}°N, ${lng.toFixed(6)}°E`;
                    document.getElementById('gpsStatus').textContent = '📍 GPS Active';
                    document.getElementById('gpsStatus').className = 'badge badge-success';
                    
                    // Send to server
                    fetch('/location', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ lat, lng })
                    }).catch(() => {});
                    
                    // Reverse geocode
                    reverseGeocode(lat, lng);
                    
                    console.log(`📍 GPS: ${lat.toFixed(6)}, ${lng.toFixed(6)}`);
                },
                (error) => {
                    console.error('GPS Error:', error);
                    document.getElementById('gpsStatus').textContent = '⚠️ GPS Error';
                    document.getElementById('gpsStatus').className = 'badge badge-danger';
                },
                { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
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
                    document.getElementById('addressText').textContent = 
                        `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                });
        }
        
        function openGoogleMaps() {
            const lat = currentLocation.lat;
            const lng = currentLocation.lng;
            const url = `https://www.google.com/maps?q=${lat},${lng}`;
            window.open(url, '_blank');
        }
        
        // ============================================
        // UI UPDATE
        // ============================================
        function updateUI(data) {
            if (!data) return;
            
            // Update server status
            document.getElementById('serverStatus').textContent = '🌐 Connected';
            document.getElementById('serverStatus').className = 'badge badge-success';
            
            // Stats
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
            
            // Detections
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
                    '<div style="text-align:center; opacity:0.5; padding:10px; font-size:13px;">No objects detected</div>';
            }
            
            // Location
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
                } else {
                    alert('Failed to send emergency alert');
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
        // POLL SERVER FOR UPDATES
        // ============================================
        function pollServer() {
            fetch('/stats')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('serverStatus').textContent = '🌐 Connected';
                    document.getElementById('serverStatus').className = 'badge badge-success';
                    
                    // Update stats
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
                })
                .catch(() => {
                    document.getElementById('serverStatus').textContent = '🌐 Connecting...';
                    document.getElementById('serverStatus').className = 'badge badge-warning';
                });
        }
        
        // ============================================
        // INITIALIZATION
        // ============================================
        function init() {
            startGPS();
            
            // Auto-start camera
            setTimeout(startCamera, 1000);
            
            // Poll server every 2 seconds
            setInterval(pollServer, 2000);
            
            console.log('✅ System initialized');
        }
        
        // Start when page loads
        document.addEventListener('DOMContentLoaded', init);
        
        // Cleanup on page unload
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
# FLASK ROUTES - Fixed for Render
# ============================================

blind_stick = None

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/stats')
def stats():
    if blind_stick:
        data = blind_stick.get_current_data()
        return jsonify(data)
    return jsonify({'person_count': 0, 'vehicle_count': 0, 'fps': 0})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frame from mobile camera"""
    if not blind_stick:
        return jsonify({'error': 'System not ready'}), 503
    
    try:
        data = request.json
        image_data = data.get('image', '')
        
        if image_data and ',' in image_data:
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
            
            # Add to queue for processing
            if blind_stick.frame_queue.qsize() < 5:
                blind_stick.frame_queue.put(image_data)
        
        # Return current detection results
        return jsonify(blind_stick.get_current_data())
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/emergency', methods=['POST'])
def emergency():
    if blind_stick:
        blind_stick.emergency_mode = True
        
        maps_url = f"https://www.google.com/maps?q={blind_stick.current_location['lat']},{blind_stick.current_location['lng']}"
        
        print(f"\n{'='*60}")
        print("🚨 EMERGENCY ALERT!")
        print(f"{'='*60}")
        print(f"📍 Location: {blind_stick.current_location['address']}")
        print(f"📍 Coordinates: {blind_stick.current_location['lat']}, {blind_stick.current_location['lng']}")
        print(f"👥 Persons detected: {blind_stick.person_count}")
        print(f"{'='*60}\n")
        
        # Reset emergency after 30 seconds
        def reset_emergency():
            time.sleep(30)
            blind_stick.emergency_mode = False
            print("🔴 Emergency mode reset")
        
        threading.Thread(target=reset_emergency, daemon=True).start()
        
        return jsonify({'status': 'success', 'maps_url': maps_url})
    
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
                    "address": f"{lat:.6f}, {lng:.6f}",
                    "source": "mobile_gps"
                }
                return jsonify({'status': 'success'})
        except Exception as e:
            print(f"Location update error: {e}")
    return jsonify({'status': 'error'}), 500

@app.route('/test')
def test():
    return jsonify({
        'status': 'running',
        'render': IS_RENDER,
        'model_loaded': blind_stick.model is not None if blind_stick else False,
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Initialize system
    blind_stick = SmartBlindStick()
    
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER...")
    print("="*60)
    
    if IS_RENDER:
        print("☁️ Render Cloud Deployment")
        print(f"📱 Open on mobile: https://your-app.onrender.com")
    else:
        print(f"📱 Open on mobile: http://{socket.gethostbyname(socket.gethostname())}:{PORT}")
        print(f"💻 Open on computer: http://127.0.0.1:{PORT}")
    
    print("\n💡 FEATURES:")
    print("   📱 Mobile camera as video source")
    print("   🎯 YOLO object detection")
    print("   📍 GPS tracking")
    print("   🚨 Emergency alerts")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if blind_stick:
            blind_stick.processing = False
