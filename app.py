# app.py - Smart Blind Stick System (Fully Optimized for Render)

import cv2
import numpy as np
import threading
import time
import queue
import warnings
import asyncio
import json
import websockets
import socket
import requests
import subprocess
import re
import os
import urllib.request
import hashlib
import gc
import resource
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import pyttsx3
import math

# ============================================
# MEMORY OPTIMIZATION - SET LIMITS
# ============================================
try:
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, -1))  # 512MB limit
    print("✅ Memory limit set to 512MB")
except:
    pass

# ============================================
# DETECT CLOUD ENVIRONMENT
# ============================================
IS_CLOUD = os.environ.get('RENDER', False) or os.environ.get('RAILWAY', False) or os.environ.get('HEROKU', False)

if IS_CLOUD:
    print("☁️ Running on cloud platform")
    print("⚠️ Camera will use test pattern mode")

# MongoDB Atlas Connection
try:
    from pymongo import MongoClient
    import pymongo
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("⚠️ PyMongo not installed")

# Try to import serial for Arduino (optional)
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ PySerial not available")

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ============================================
# MONGODB ATLAS CONFIGURATION
# ============================================
MONGO_URI = "mongodb+srv://gowsik977_db_user:gowsik123@cluster1.t4w8mul.mongodb.net/?appName=Cluster1"
DB_NAME = "blind_stick_db"

print("\n🔌 Connecting to MongoDB Atlas...")
db = None
mongo_client = None

try:
    if PYMONGO_AVAILABLE:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        db = mongo_client[DB_NAME]
        
        # Create collections if they don't exist
        collections = ['alerts', 'detections', 'device_registry', 'emergency_events', 
                       'live_locations', 'location_tracking', 'system_logs']
        
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
                print(f"📁 Created collection: {collection}")
        
        # Create indexes for better performance
        if db is not None:
            db.detections.create_index([("timestamp", pymongo.DESCENDING)])
            db.detections.create_index([("device_id", pymongo.ASCENDING)])
            db.alerts.create_index([("timestamp", pymongo.DESCENDING)])
            db.emergency_events.create_index([("timestamp", pymongo.DESCENDING)])
            db.location_tracking.create_index([("timestamp", pymongo.DESCENDING)])
            db.location_tracking.create_index([("device_id", pymongo.ASCENDING)])
            db.live_locations.create_index([("device_id", pymongo.ASCENDING)], unique=True)
        
        print("✅ Connected to MongoDB Atlas successfully!")
        print(f"📊 Database: {DB_NAME}")
        print(f"📁 Collections: {', '.join(db.list_collection_names())}")
    else:
        print("❌ PyMongo not available")
        
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    db = None
    mongo_client = None

# Generate unique device ID
DEVICE_ID = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
DEVICE_NAME = f"Device-{DEVICE_ID[:6]}"

# Your Google Maps API Key
GOOGLE_MAPS_API_KEY = "AIzaSyCdQGVYnjmSAzxnTu4g_zEXKGhgzqbZDvc"

# ============================================
# ARDUINO MANAGER CLASS
# ============================================
class ArduinoManager:
    def __init__(self, port=None, baudrate=9600):
        self.serial_connection = None
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        self.last_alert_time = 0
        self.alert_cooldown = 0.3
        self.last_heartbeat = 0
        self.heartbeat_timeout = 10
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.enabled = SERIAL_AVAILABLE
        
    def find_arduino_port(self):
        if not self.enabled:
            return None
        try:
            print("🔍 Scanning for Arduino ports...")
            ports = serial.tools.list_ports.comports()
            valid_ports = []
            for port in ports:
                desc = port.description.lower()
                device = port.device
                print(f"   Found: {device} - {port.description}")
                if any(x in desc for x in ['bluetooth', 'bt link', 'bthnum', 'hands-free', 'wireless', 'modem']):
                    print(f"   ⚠️ Skipping Bluetooth port: {device}")
                    continue
                valid_ports.append(port)
            
            for port in valid_ports:
                desc = port.description.lower()
                if any(x in desc for x in ['arduino', 'usb', 'ch340', 'cp210', 'uart', 'ftdi', 'pl2303', 'prolific']):
                    print(f"   ✅ Selected Arduino: {port.device}")
                    return port.device
            
            for port in valid_ports:
                try:
                    test_ser = serial.Serial(port.device, 9600, timeout=0.5)
                    test_ser.close()
                    print(f"✅ Selected serial port: {port.device}")
                    return port.device
                except:
                    continue
            return None
        except Exception as e:
            print(f"Error finding Arduino: {e}")
            return None
    
    def connect(self):
        if not self.enabled:
            print("⚠️ Serial library not available. Running without Arduino.")
            return False
        try:
            if self.port is None:
                self.port = self.find_arduino_port()
                if self.port is None:
                    print("⚠️ Arduino not found! Running in software-only mode.")
                    return False
            
            print(f"🔌 Attempting to connect to Arduino on {self.port}...")
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(3)
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            
            start_time = time.time()
            while time.time() - start_time < 5:
                if self.serial_connection.in_waiting:
                    response = self.serial_connection.readline().decode().strip()
                    print(f"   Arduino: {response}")
                    if response == "READY":
                        self.connected = True
                        print(f"✅ Arduino connected on {self.port}")
                        return True
                time.sleep(0.1)
            
            if self.serial_connection.is_open:
                self.connected = True
                print(f"✅ Arduino connected on {self.port} (no response)")
                return True
                
        except serial.SerialException as e:
            if "Access is denied" in str(e):
                print(f"⚠️ Port {self.port} is in use! Close Arduino IDE or other programs.")
            else:
                print(f"⚠️ Arduino connection error: {e}")
            print("   Running without Arduino hardware feedback")
        except Exception as e:
            print(f"⚠️ Arduino connection error: {e}")
            print("   Running without Arduino hardware feedback")
        
        self.connected = False
        return False
    
    def send_command(self, command, wait_response=False, timeout=1):
        if not self.connected or not self.enabled or self.serial_connection is None:
            return None
        try:
            if not command.endswith('\n'):
                command += '\n'
            self.serial_connection.write(command.encode())
            if wait_response:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.serial_connection.in_waiting:
                        response = self.serial_connection.readline().decode().strip()
                        return response
                    time.sleep(0.05)
                return None
            return True
        except Exception as e:
            print(f"⚠️ Error sending to Arduino: {e}")
            self.connected = False
            return None
    
    def send_alert(self, alert_type, distance):
        if not self.connected or not self.enabled or self.serial_connection is None:
            return False
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        try:
            command = f"ALERT:{alert_type},{distance}\n"
            self.serial_connection.write(command.encode())
            self.last_alert_time = current_time
            print(f"📟 Arduino Alert: {alert_type} at {distance}cm")
            return True
        except Exception as e:
            print(f"⚠️ Error sending to Arduino: {e}")
            self.connected = False
            return False
    
    def stop_alert(self):
        if not self.connected or not self.enabled or self.serial_connection is None:
            return
        try:
            self.serial_connection.write(b"STOP\n")
        except:
            pass
    
    def check_heartbeat(self):
        if not self.connected:
            return False
        try:
            self.serial_connection.write(b"STATUS\n")
            return True
        except:
            self.connected = False
            return False
    
    def close(self):
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.stop_alert()
                self.serial_connection.close()
            except:
                pass
            print("🔌 Arduino disconnected")

# ============================================
# SMART BLIND STICK MAIN CLASS
# ============================================
class SmartBlindStick:
    def __init__(self):
        print("\n" + "="*60)
        print("🦯 Initializing Smart Blind Stick System")
        print("="*60)
        
        # Device ID
        self.device_id = DEVICE_ID
        self.device_name = DEVICE_NAME
        
        # Register device in MongoDB
        self.register_device()
        
        # Initialize Arduino (optional)
        print("\n🔌 Checking for Arduino...")
        self.arduino = ArduinoManager()
        self.arduino.connect()
        
        # Initialize text-to-speech
        print("\n🔊 Initializing Text-to-Speech...")
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.tts_available = True
            print("✅ Text-to-speech initialized")
        except Exception as e:
            print(f"⚠️ Text-to-speech not available: {e}")
            self.tts_available = False
        
        self.speech_queue = queue.Queue()
        self.last_spoken = {}
        self.speech_cooldown = {
            'person': 2.0, 'stairs': 3.0, 'pothole': 2.5, 'wall': 2.5, 'vehicle': 2.0, 'emergency': 10.0
        }
        
        # Load YOLO model with memory optimization
        print("\n📷 Loading YOLO model...")
        try:
            import torch
            # Fix for PyTorch 2.6+
            try:
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
            except:
                pass
            model_name = 'yolov8n.pt'
            self.model = YOLO(model_name)
            # Memory optimization for YOLO
            self.model.overrides['imgsz'] = 320  # Smaller image size
            self.model.overrides['conf'] = 0.5   # Higher confidence
            self.model.overrides['device'] = 'cpu'
            self.model.overrides['verbose'] = False
            print(f"✅ YOLO model loaded! (Using {model_name})")
        except Exception as e:
            print(f"⚠️ YOLO not available: {e}")
            self.model = None
        
        # Important classes for detection
        self.important_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 11: 'stop sign'
        }
        
        # Camera setup - Skip if on cloud
        print("\n🎥 Opening camera...")
        self.cap = None
        
        if IS_CLOUD:
            print("⚠️ Cloud environment detected - using test pattern")
            self.use_test_pattern = True
        else:
            # Try to open camera normally
            for i in range(5):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap = cap
                        print(f"✅ Camera {i} opened successfully!")
                        self.use_test_pattern = False
                        break
                    else:
                        cap.release()
                except Exception as e:
                    print(f"⚠️ Camera {i} error: {e}")
                    continue
            
            if self.cap is None:
                print("❌ No camera found! Using test pattern.")
                self.use_test_pattern = True
        
        self.clients = set()
        self.current_data = {}
        self.emergency_mode = False
        self.person_count = 0
        self.vehicle_count = 0
        self.detected_objects = []
        self.fps = 0
        self.detection_count = 0
        self.location_update_count = 0
        self.frame_counter = 0
        self.gc_counter = 0
        
        # Get local IP
        self.local_ip = self.get_local_ip()
        
        # Current location
        self.current_location = {
            "lat": 0.0,
            "lng": 0.0,
            "address": "Getting exact location...",
            "accuracy": 0,
            "altitude": 0,
            "speed": 0,
            "heading": 0,
            "source": "waiting",
            "timestamp": datetime.now().isoformat()
        }
        
        # Start threads
        threading.Thread(target=self.process_speech_queue, daemon=True).start()
        threading.Thread(target=self.arduino_heartbeat_check, daemon=True).start()
        
        # Log system start
        self.log_system_event('SYSTEM_START', 'Smart Blind Stick system initialized')
        
        print("\n" + "="*60)
        print("✅ SYSTEM READY!")
        print(f"   Device ID: {self.device_id}")
        print(f"   Device Name: {self.device_name}")
        print(f"   IP Address: {self.local_ip}")
        print(f"   Arduino: {'✅ Connected' if self.arduino.connected else '⚠️ Not Connected'}")
        print(f"   Camera: {'✅ OK' if self.cap else '⚠️ Test Pattern'}")
        print(f"   TTS: {'✅ OK' if self.tts_available else '⚠️ Disabled'}")
        print(f"   MongoDB: {'✅ Connected' if db is not None else '❌ Disabled'}")
        print(f"   Cloud Mode: {'✅ Enabled' if IS_CLOUD else '❌ Disabled'}")
        print("="*60 + "\n")
    
    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def register_device(self):
        """Register device in MongoDB"""
        if db is None:
            return
        try:
            db.device_registry.update_one(
                {"device_id": self.device_id},
                {"$set": {
                    "device_id": self.device_id,
                    "device_name": self.device_name,
                    "ip_address": self.get_local_ip(),
                    "status": "active",
                    "registered_at": datetime.now(),
                    "last_seen": datetime.now()
                }},
                upsert=True
            )
            print(f"✅ Device registered in MongoDB: {self.device_id}")
        except Exception as e:
            print(f"⚠️ Failed to register device: {e}")
    
    def arduino_heartbeat_check(self):
        while True:
            time.sleep(5)
            if self.arduino.connected:
                if not self.arduino.check_heartbeat():
                    print("⚠️ Arduino connection lost! Attempting to reconnect...")
                    self.arduino.connect()
    
    def process_speech_queue(self):
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                if self.tts_available:
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    print(f"🔊 VOICE: {text}")
                self.speech_queue.task_done()
            except:
                pass
    
    def speak(self, text, alert_type='obstacle'):
        now = time.time()
        if alert_type in self.last_spoken:
            if now - self.last_spoken[alert_type] < self.speech_cooldown.get(alert_type, 2):
                return
        self.last_spoken[alert_type] = now
        self.speech_queue.put(text)
        
        # Log to MongoDB
        self.save_alert_to_db(alert_type, text)
        
        print(f"🔊 Speaking: {text}")
    
    def save_alert_to_db(self, alert_type, message):
        """Save alert to MongoDB"""
        if db is None:
            return
        try:
            db.alerts.insert_one({
                "device_id": self.device_id,
                "device_name": self.device_name,
                "alert_type": alert_type,
                "message": message,
                "location": self.current_location,
                "person_count": self.person_count,
                "vehicle_count": self.vehicle_count,
                "ip_address": self.local_ip,
                "timestamp": datetime.now()
            })
        except Exception as e:
            print(f"⚠️ Failed to save alert: {e}")
    
    def save_detection_to_db(self, detection):
        """Save detection to MongoDB"""
        if db is None:
            return
        try:
            db.detections.insert_one({
                "device_id": self.device_id,
                "device_name": self.device_name,
                "object_type": detection.get('class'),
                "confidence": detection.get('confidence'),
                "distance": detection.get('distance'),
                "distance_cm": detection.get('distance_cm'),
                "direction": detection.get('direction'),
                "location": self.current_location,
                "ip_address": self.local_ip,
                "timestamp": datetime.now()
            })
        except Exception as e:
            print(f"⚠️ Failed to save detection: {e}")
    
    def save_location_to_db(self):
        """Save current location to MongoDB"""
        if db is None:
            return
        try:
            # Update live location
            db.live_locations.update_one(
                {"device_id": self.device_id},
                {"$set": {
                    "device_name": self.device_name,
                    "latitude": self.current_location.get('lat'),
                    "longitude": self.current_location.get('lng'),
                    "address": self.current_location.get('address'),
                    "accuracy": self.current_location.get('accuracy', 0),
                    "altitude": self.current_location.get('altitude', 0),
                    "speed": self.current_location.get('speed', 0),
                    "heading": self.current_location.get('heading', 0),
                    "source": self.current_location.get('source', 'unknown'),
                    "ip_address": self.local_ip,
                    "timestamp": datetime.now()
                }},
                upsert=True
            )
            
            # Insert location history
            db.location_tracking.insert_one({
                "device_id": self.device_id,
                "device_name": self.device_name,
                "latitude": self.current_location.get('lat'),
                "longitude": self.current_location.get('lng'),
                "address": self.current_location.get('address'),
                "accuracy": self.current_location.get('accuracy', 0),
                "altitude": self.current_location.get('altitude', 0),
                "speed": self.current_location.get('speed', 0),
                "heading": self.current_location.get('heading', 0),
                "source": self.current_location.get('source', 'unknown'),
                "ip_address": self.local_ip,
                "timestamp": datetime.now()
            })
            
            self.location_update_count += 1
            if self.location_update_count % 10 == 0:
                print(f"📍 Exact Location saved to MongoDB: {self.current_location.get('lat')}, {self.current_location.get('lng')} (Accuracy: {self.current_location.get('accuracy', 0)}m)")
                
        except Exception as e:
            print(f"⚠️ Failed to save location: {e}")
    
    def log_system_event(self, event_type, details):
        """Log system event to MongoDB"""
        if db is None:
            return
        try:
            db.system_logs.insert_one({
                "device_id": self.device_id,
                "device_name": self.device_name,
                "event_type": event_type,
                "details": details,
                "ip_address": self.local_ip,
                "timestamp": datetime.now()
            })
        except Exception as e:
            print(f"⚠️ Failed to log system event: {e}")
    
    def send_arduino_alert(self, alert_type, distance_cm):
        if self.arduino.connected and distance_cm < 100:
            self.arduino.send_alert(alert_type, distance_cm)
    
    def detect_with_yolo(self, frame):
        detections = []
        height, width = frame.shape[:2]
        
        if self.model is None:
            return frame, detections
        
        try:
            results = self.model(frame, stream=True, conf=0.5)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if conf < 0.5:
                            continue
                        
                        class_name = self.important_classes.get(cls, f"object")
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        box_height = y2 - y1
                        
                        if box_height > height * 0.5:
                            distance = "very close"
                            distance_cm = 30
                            color = (0, 0, 255)
                        elif box_height > height * 0.3:
                            distance = "close"
                            distance_cm = 60
                            color = (0, 165, 255)
                        elif box_height > height * 0.15:
                            distance = "medium"
                            distance_cm = 120
                            color = (0, 255, 255)
                        else:
                            distance = "far"
                            distance_cm = 200
                            color = (0, 255, 0)
                        
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
                            'direction': direction,
                            'bbox': (x1, y1, x2, y2)
                        }
                        detections.append(detection)
                        
                        # Save to MongoDB (with limit to prevent spam)
                        if self.detection_count % 5 == 0:
                            self.save_detection_to_db(detection)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name}: {conf:.2f} ({distance}, {direction})"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if class_name == 'person':
                            if distance == "very close":
                                self.speak(f"Person {direction}, very close!", 'person')
                                self.send_arduino_alert('PERSON', distance_cm)
                            elif distance == "close":
                                self.speak(f"Person {direction}", 'person')
                                self.send_arduino_alert('PERSON', distance_cm)
                            self.detection_count += 1
                            
                        elif class_name in ['car', 'truck', 'bus', 'bicycle', 'motorcycle']:
                            if distance in ["very close", "close"]:
                                self.speak(f"Vehicle {direction}, {distance}!", 'vehicle')
                                self.send_arduino_alert('VEHICLE', distance_cm)
                            self.detection_count += 1
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
        
        return frame, detections
    
    def generate_test_pattern(self):
        """Generate test pattern with movement for cloud deployment"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Animated bars
        t = time.time()
        
        # Background gradient
        for i in range(480):
            for j in range(640):
                frame[i][j] = [int(50 + 50 * np.sin(i/20 + t)), 
                               int(50 + 50 * np.cos(j/20 + t)), 
                               int(100 + 50 * np.sin((i+j)/30 + t))]
        
        # Main text
        cv2.putText(frame, "🦯 SMART BLIND STICK", (100, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, "☁️ CLOUD MODE ACTIVE", (180, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "📱 Connected: " + str(len(self.clients)), (180, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"📍 {self.current_location.get('address', 'Location')[:30]}", (100, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Device: {self.device_name} | {self.local_ip}", (100, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {self.fps} | Persons: {self.person_count}", (100, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "✅ System Running on Render", (180, 420), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Moving indicator
        pos = int((np.sin(t/2) + 1) * 300)
        cv2.circle(frame, (pos, 450), 10, (0, 255, 255), -1)
        
        return frame
    
    def generate_frames(self):
        fps_start = time.time()
        frame_count = 0
        
        while True:
            # Memory management - periodic GC
            self.frame_counter += 1
            if self.frame_counter % 50 == 0:
                gc.collect()
                print(f"🧹 Garbage collected (frame {self.frame_counter})")
            
            if self.use_test_pattern or self.cap is None:
                # Use animated test pattern
                frame = self.generate_test_pattern()
                detections = []
            else:
                ret, frame = self.cap.read()
                if not ret:
                    frame = self.generate_test_pattern()
                    detections = []
                else:
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - fps_start
                        self.fps = int(30 / elapsed) if elapsed > 0 else 30
                        fps_start = time.time()
                    
                    # Process every 3rd frame to reduce CPU/GPU usage
                    if frame_count % 3 == 0:
                        frame, detections = self.detect_with_yolo(frame)
                    else:
                        detections = self.detected_objects
            
            self.person_count = sum(1 for d in detections if d['class'] == 'person')
            self.vehicle_count = sum(1 for d in detections if d['class'] in ['car', 'truck', 'bus', 'bicycle', 'motorcycle'])
            self.detected_objects = detections
            
            # Add HUD overlay
            y_offset = 30
            cv2.putText(frame, "SMART BLIND STICK SYSTEM", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.arduino.connected:
                cv2.putText(frame, "Arduino: ✅ Connected", (10, y_offset + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Arduino: ⚠️ Software Mode", (10, y_offset + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            cv2.putText(frame, f"FPS: {self.fps} | Persons: {self.person_count} | Vehicles: {self.vehicle_count}", 
                       (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"Mobile Connected: {len(self.clients)}", (10, y_offset + 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Display exact location with accuracy
            loc_text = f"📍 {self.current_location.get('address', 'Unknown')[:30]}"
            cv2.putText(frame, loc_text, (10, y_offset + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            
            acc_text = f"Accuracy: {self.current_location.get('accuracy', 0)}m | Source: {self.current_location.get('source', 'unknown')}"
            cv2.putText(frame, acc_text, (10, y_offset + 118), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)
            
            cv2.putText(frame, f"Device: {self.device_name} | IP: {self.local_ip}", (10, y_offset + 138), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
            
            if self.emergency_mode:
                cv2.putText(frame, "EMERGENCY MODE ACTIVE", (10, y_offset + 158), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if IS_CLOUD:
                cv2.putText(frame, "☁️ Cloud Mode", (10, y_offset + 178), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            
            self.current_data = {
                'device_id': self.device_id,
                'device_name': self.device_name,
                'detections': detections,
                'person_count': self.person_count,
                'vehicle_count': self.vehicle_count,
                'fps': self.fps,
                'emergency': self.emergency_mode,
                'location': self.current_location,
                'timestamp': datetime.now().isoformat(),
                'connected_clients': len(self.clients),
                'detection_count': self.detection_count,
                'arduino_connected': self.arduino.connected,
                'ip_address': self.local_ip,
                'cloud_mode': IS_CLOUD
            }
            
            frame = np.ascontiguousarray(frame)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                # If encoding fails, skip this frame
                time.sleep(0.05)
                continue
            
            # Small delay to prevent CPU overuse
            time.sleep(0.02)
    
    async def handle_client(self, websocket):
        self.clients.add(websocket)
        print(f"📱 Mobile connected! Total: {len(self.clients)}")
        
        try:
            if self.current_data:
                await websocket.send(json.dumps(self.current_data))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'register':
                        await websocket.send(json.dumps({
                            'type': 'registered', 
                            'status': 'ok',
                            'device_id': self.device_id,
                            'device_name': self.device_name
                        }))
                    elif data.get('type') == 'request_location':
                        await websocket.send(json.dumps({
                            'type': 'location_update',
                            'location': self.current_location
                        }))
                    elif data.get('type') == 'location_update':
                        lat = data.get('lat')
                        lng = data.get('lng')
                        address = data.get('address')
                        accuracy = data.get('accuracy', 0)
                        altitude = data.get('altitude', 0)
                        speed = data.get('speed', 0)
                        heading = data.get('heading', 0)
                        
                        if lat is not None and lng is not None:
                            self.current_location = {
                                "lat": lat,
                                "lng": lng,
                                "address": address or f"{lat:.6f}, {lng:.6f}",
                                "accuracy": accuracy,
                                "altitude": altitude,
                                "speed": speed,
                                "heading": heading,
                                "source": "GPS_HighAccuracy" if accuracy < 20 else "GPS" if accuracy < 100 else "WiFi",
                                "timestamp": datetime.now().isoformat()
                            }
                            self.save_location_to_db()
                            self.log_system_event('LOCATION_UPDATE', 
                                f"Exact Location: {self.current_location['address']} (Accuracy: {accuracy}m)")
                            print(f"📍 EXACT LOCATION: {lat}, {lng} (Accuracy: {accuracy}m, Source: {self.current_location['source']})")
                    elif data.get('type') == 'emergency':
                        await self.handle_emergency_request()
                except Exception as e:
                    print(f"WebSocket message error: {e}")
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.clients.remove(websocket)
            print(f"📱 Mobile disconnected. Total: {len(self.clients)}")
    
    async def handle_emergency_request(self):
        self.emergency_mode = True
        self.speak("EMERGENCY! Help needed!", "emergency")
        await self.send_emergency(self.current_location, self.person_count)
        
        def reset():
            time.sleep(30)
            self.emergency_mode = False
        threading.Thread(target=reset, daemon=True).start()
    
    async def broadcast_updates(self):
        while True:
            if self.clients and self.current_data:
                clients_copy = list(self.clients)
                dead = set()
                for client in clients_copy:
                    try:
                        await client.send(json.dumps(self.current_data))
                    except:
                        dead.add(client)
                if dead:
                    self.clients -= dead
            await asyncio.sleep(0.1)
    
    def run_websocket(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.ws_loop = loop
        
        async def server():
            try:
                async with websockets.serve(self.handle_client, '0.0.0.0', 8765):
                    print("🔌 WebSocket server running on ws://0.0.0.0:8765")
                    await asyncio.gather(self.broadcast_updates(), asyncio.Future())
            except Exception as e:
                print(f"⚠️ WebSocket server error: {e}")
                print("   WebSocket may not be supported on this platform")
        
        try:
            loop.run_until_complete(server())
        except Exception as e:
            print(f"⚠️ WebSocket error: {e}")
        finally:
            loop.close()
    
    async def send_emergency(self, location, person_count):
        maps_url = f"https://www.google.com/maps?q={location['lat']},{location['lng']}"
        
        emergency_data = {
            'type': 'emergency',
            'title': '🚨 EMERGENCY ALERT! 🚨',
            'message': f'Emergency button pressed! Immediate assistance needed!',
            'location': location,
            'maps_url': maps_url,
            'person_count': person_count,
            'device_id': self.device_id,
            'device_name': self.device_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n{'='*60}")
        print("🚨 EMERGENCY ALERT SENT!")
        print(f"{'='*60}")
        print(f"📍 EXACT Location: {location['address']}")
        print(f"📍 Coordinates: {location['lat']}, {location['lng']}")
        print(f"📍 Accuracy: {location.get('accuracy', 0)}m")
        print(f"📍 Source: {location.get('source', 'unknown')}")
        print(f"📍 Google Maps: {maps_url}")
        print(f"👥 Persons detected: {person_count}")
        print(f"📱 Device: {self.device_name}")
        print(f"🌐 IP: {self.local_ip}")
        
        # Trigger Arduino emergency
        if self.arduino.connected:
            self.arduino.send_alert('EMERGENCY', 0)
        
        # Save to MongoDB
        if db is not None:
            try:
                db.emergency_events.insert_one({
                    "device_id": self.device_id,
                    "device_name": self.device_name,
                    "location": location,
                    "person_count": person_count,
                    "vehicle_count": self.vehicle_count,
                    "ip_address": self.local_ip,
                    "timestamp": datetime.now()
                })
                db.alerts.insert_one({
                    "device_id": self.device_id,
                    "device_name": self.device_name,
                    "alert_type": "emergency",
                    "message": f"🚨 EMERGENCY: Alert triggered at {location.get('address', 'Unknown location')}",
                    "location": location,
                    "person_count": person_count,
                    "ip_address": self.local_ip,
                    "timestamp": datetime.now()
                })
                print("✅ Emergency logged to MongoDB")
            except Exception as e:
                print(f"⚠️ Failed to log emergency: {e}")
        
        # Send to all connected clients
        if self.clients:
            for client in self.clients:
                try:
                    await client.send(json.dumps(emergency_data))
                    print("✅ Alert sent to mobile")
                except Exception as e:
                    print(f"❌ Failed: {e}")
        
        print(f"{'='*60}\n")
        return emergency_data
    
    def run(self):
        ws_thread = threading.Thread(target=self.run_websocket, daemon=True)
        ws_thread.start()
        
        self.speak("Smart Blind Stick system started", "system")
        
        print("\n" + "="*60)
        print("🌐 SERVER RUNNING!")
        print("="*60)
        print(f"📱 Open on your MOBILE PHONE: http://{self.local_ip}:5000")
        print(f"💻 Open on this computer: http://127.0.0.1:5000")
        print(f"🔌 WebSocket: ws://{self.local_ip}:8765")
        print(f"📱 Device ID: {self.device_id}")
        print(f"📱 Device Name: {self.device_name}")
        print(f"📊 MongoDB: {'✅ Connected' if db is not None else '❌ Disabled'}")
        print(f"☁️ Cloud Mode: {'✅ Enabled' if IS_CLOUD else '❌ Disabled'}")
        print("\n💡 TIPS:")
        print("   • All detections and alerts are saved to MongoDB Atlas")
        print("   • EXACT LOCATION tracking with GPS + WiFi fallback")
        print("   • Location accuracy shown in meters")
        print("   • Press 'E' key on keyboard for emergency alert")
        print("="*60 + "\n")
    
    def cleanup(self):
        self.arduino.close()
        if self.cap:
            self.cap.release()
        if mongo_client:
            mongo_client.close()
            print("🔌 MongoDB connection closed")

# ============================================
# HTML TEMPLATE (Same as before)
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
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 16px;
            color: #fff;
        }
        .container { max-width: 500px; margin: 0 auto; }
        h1 { text-align: center; font-size: 24px; margin-bottom: 20px; }
        .status { text-align: center; margin-bottom: 20px; }
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin: 5px;
        }
        .connected { background: rgba(76,175,80,0.3); border: 1px solid #4caf50; color: #4caf50; }
        .disconnected { background: rgba(244,67,54,0.3); border: 1px solid #f44336; color: #f44336; }
        .warning { background: rgba(255,193,7,0.3); border: 1px solid #ffc107; color: #ffc107; }
        .video-container {
            background: #000;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 16px;
        }
        .video-container img { width: 100%; display: block; }
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
            margin-bottom: 16px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%,100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 15px;
            margin-bottom: 16px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        .stat {
            text-align: center;
            background: rgba(0,0,0,0.4);
            padding: 10px;
            border-radius: 10px;
        }
        .stat-value { font-size: 24px; font-weight: bold; color: #4caf50; }
        .stat-label { font-size: 12px; opacity: 0.7; margin-top: 5px; }
        .detection-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .detection-item {
            background: rgba(0,0,0,0.4);
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
        }
        .detection-danger { border-left: 3px solid #ff4444; }
        .detection-warning { border-left: 3px solid #ff9800; }
        .alert-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .alert-item {
            background: rgba(0,0,0,0.4);
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            border-left: 3px solid #ff9800;
        }
        .alert-emergency {
            border-left-color: #ff4444;
            animation: blink 0.5s;
        }
        @keyframes blink {
            0%,100% { background: rgba(255,68,68,0.2); }
            50% { background: rgba(255,68,68,0.4); }
        }
        .location {
            font-size: 12px;
            text-align: center;
            margin-top: 10px;
            word-wrap: break-word;
        }
        .location .coords { font-size: 11px; opacity: 0.7; }
        .location .accuracy { font-size: 10px; opacity: 0.5; }
        .device-info {
            font-size: 10px;
            opacity: 0.5;
            text-align: center;
            margin-top: 5px;
        }
        #map {
            height: 250px;
            border-radius: 12px;
            margin-top: 10px;
            margin-bottom: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .map-btn {
            width: 100%;
            padding: 10px;
            background: #4caf50;
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-weight: bold;
        }
        .map-btn:hover { background: #45a049; }
        .connection-section {
            background: rgba(0,0,0,0.4);
            border-radius: 16px;
            padding: 16px;
            margin-top: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .connection-section h3 {
            font-size: 14px;
            margin-bottom: 12px;
            color: #4caf50;
            text-align: center;
        }
        .connection-link {
            background: rgba(255,255,255,0.05);
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 10px;
            word-break: break-all;
        }
        .connection-link .label {
            font-size: 10px;
            opacity: 0.5;
            margin-bottom: 4px;
        }
        .connection-link .url {
            font-size: 14px;
            font-weight: 500;
            color: #2196f3;
            font-family: monospace;
            cursor: pointer;
        }
        .connection-link .url:hover { color: #4caf50; }
        .connection-link .copy-btn {
            background: rgba(33,150,243,0.3);
            border: 1px solid #2196f3;
            color: #2196f3;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 10px;
            cursor: pointer;
            margin-top: 6px;
        }
        .connection-link .copy-btn:hover { background: rgba(33,150,243,0.5); }
        .qr-container {
            text-align: center;
            margin: 10px 0;
        }
        .qr-container img {
            width: 150px;
            height: 150px;
            background: white;
            padding: 10px;
            border-radius: 12px;
        }
        .device-id-display {
            text-align: center;
            font-size: 11px;
            opacity: 0.6;
            margin-top: 8px;
            font-family: monospace;
        }
        .connected-devices {
            margin-top: 10px;
        }
        .connected-devices .device-item {
            background: rgba(76,175,80,0.1);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 11px;
            margin: 3px 0;
            border-left: 2px solid #4caf50;
        }
        .accuracy-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 10px;
            margin-top: 4px;
        }
        .accuracy-high { background: rgba(76,175,80,0.3); color: #4caf50; border: 1px solid #4caf50; }
        .accuracy-medium { background: rgba(255,193,7,0.3); color: #ffc107; border: 1px solid #ffc107; }
        .accuracy-low { background: rgba(244,67,54,0.3); color: #f44336; border: 1px solid #f44336; }
        .cloud-badge {
            background: rgba(33,150,243,0.2);
            border: 1px solid #2196f3;
            color: #2196f3;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 10px;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🦯 Smart Blind Stick</h1>
        
        <div class="status">
            <span id="wsStatus" class="badge disconnected">🔴 Connecting...</span>
            <span id="arduinoStatus" class="badge disconnected">🔌 Arduino: Unknown</span>
            <span id="dbStatus" class="badge disconnected">📊 DB: Unknown</span>
            <span id="cloudStatus" class="badge cloud-badge" style="display:none;">☁️ Cloud</span>
        </div>
        
        <div class="video-container">
            <img id="videoFeed" src="/video_feed" alt="Camera Feed">
        </div>
        
        <button class="emergency-btn" onclick="sendEmergency()">🚨 EMERGENCY BUTTON 🚨</button>
        
        <div class="card">
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="personCount">0</div>
                    <div class="stat-label">Persons</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="vehicleCount">0</div>
                    <div class="stat-label">Vehicles</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="fpsValue">0</div>
                    <div class="stat-label">FPS</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>👤 Detected Objects</h3>
            <div id="detectionList" class="detection-list">No detections yet...</div>
        </div>
        
        <div class="card">
            <h3>🔔 Alerts</h3>
            <div id="alertList" class="alert-list"></div>
        </div>
        
        <div class="card">
            <h3>📍 EXACT Location</h3>
            <div id="locationInfo" class="location">
                <div id="locationAddress">Getting exact location...</div>
                <div class="coords" id="coordsText">--</div>
                <div class="accuracy" id="accuracyText">Accuracy: --</div>
                <div id="locationSource" style="font-size:10px;opacity:0.6;">Source: --</div>
            </div>
            <div id="map"></div>
            <button class="map-btn" onclick="openGoogleMaps()">🗺️ Open in Google Maps</button>
            <div class="device-info" id="deviceInfo">Device: Loading...</div>
        </div>
        
        <div class="connection-section">
            <h3>🔗 Share This Device</h3>
            
            <div class="connection-link">
                <div class="label">📱 Open on Other Mobile / Laptop</div>
                <div class="url" id="connectionUrl" onclick="copyToClipboard('connectionUrl')">Loading URL...</div>
                <button class="copy-btn" onclick="copyToClipboard('connectionUrl')">📋 Copy Link</button>
            </div>
            
            <div class="connection-link">
                <div class="label">🖥️ Local IP Address</div>
                <div class="url" id="localIp" onclick="copyToClipboard('localIp')">Detecting...</div>
                <button class="copy-btn" onclick="copyToClipboard('localIp')">📋 Copy IP</button>
            </div>
            
            <div class="connection-link">
                <div class="label">📱 Device ID</div>
                <div class="url" id="deviceIdDisplay" style="font-size:12px;color:#ffc107;">Loading...</div>
            </div>
            
            <div class="qr-container">
                <div id="qrCodeContainer">
                    <p style="font-size:11px;opacity:0.5;">Scan to connect</p>
                    <img id="qrCodeImage" src="" alt="QR Code" style="display:none;">
                    <div id="qrLoading" style="padding:20px;font-size:12px;opacity:0.5;">Loading QR Code...</div>
                </div>
            </div>
            
            <div class="connected-devices">
                <div class="label" style="font-size:10px;opacity:0.5;margin-bottom:5px;">📱 Connected Devices:</div>
                <div id="connectedDevicesList">
                    <div style="font-size:11px;opacity:0.4;">No devices connected</div>
                </div>
            </div>
            
            <div class="device-id-display" id="deviceIdFull">Device ID: --</div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let map = null;
        let marker = null;
        let currentLocation = { lat: 0, lng: 0 };
        let geocoder = null;
        let locationWatchId = null;
        let deviceInfo = {};
        let connectedDevices = [];
        
        function initMap() {
            const center = { lat: currentLocation.lat || 11.2745, lng: currentLocation.lng || 77.5831 };
            map = new google.maps.Map(document.getElementById('map'), {
                zoom: 18,
                center: center,
                styles: [
                    { featureType: 'all', elementType: 'all', stylers: [{ saturation: -80 }, { lightness: 20 }] }
                ]
            });
            
            marker = new google.maps.Marker({
                position: center,
                map: map,
                title: 'Current Location',
                icon: {
                    url: 'https://maps.google.com/mapfiles/ms/icons/red-dot.png',
                    scaledSize: new google.maps.Size(40, 40)
                }
            });
            
            const circle = new google.maps.Circle({
                map: map,
                radius: 50,
                fillColor: '#2196f3',
                fillOpacity: 0.15,
                strokeColor: '#2196f3',
                strokeOpacity: 0.3,
                strokeWeight: 1
            });
            circle.bindTo('center', marker, 'position');
            
            geocoder = new google.maps.Geocoder();
        }
        
        function updateLocationOnMap(lat, lng, accuracy) {
            const pos = { lat: lat, lng: lng };
            if (marker) {
                marker.setPosition(pos);
                map.setCenter(pos);
                map.setZoom(18);
            }
            if (accuracy !== undefined && accuracy !== null) {
                document.getElementById('accuracyText').textContent = `Accuracy: ${accuracy}m`;
            }
        }
        
        function startTracking() {
            if (navigator.geolocation) {
                const options = {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 0
                };
                
                locationWatchId = navigator.geolocation.watchPosition(
                    (position) => {
                        const lat = position.coords.latitude;
                        const lng = position.coords.longitude;
                        const accuracy = position.coords.accuracy || 0;
                        const altitude = position.coords.altitude || 0;
                        const speed = position.coords.speed || 0;
                        const heading = position.coords.heading || 0;
                        
                        currentLocation = { lat, lng, accuracy };
                        
                        document.getElementById('coordsText').textContent = 
                            `${lat.toFixed(6)}°N, ${lng.toFixed(6)}°E`;
                        document.getElementById('accuracyText').textContent = 
                            `Accuracy: ${accuracy.toFixed(0)}m | Speed: ${(speed * 3.6).toFixed(1)} km/h`;
                        
                        const source = accuracy < 20 ? 'GPS (High Accuracy)' : accuracy < 100 ? 'GPS' : 'WiFi/Network';
                        document.getElementById('locationSource').textContent = `Source: ${source}`;
                        
                        const badge = document.getElementById('accuracyBadge') || document.createElement('span');
                        badge.id = 'accuracyBadge';
                        badge.className = `accuracy-badge ${accuracy < 20 ? 'accuracy-high' : accuracy < 100 ? 'accuracy-medium' : 'accuracy-low'}`;
                        badge.textContent = accuracy < 20 ? '🟢 High Accuracy' : accuracy < 100 ? '🟡 Medium Accuracy' : '🔴 Low Accuracy';
                        document.getElementById('accuracyText').appendChild(badge);
                        
                        updateLocationOnMap(lat, lng, accuracy);
                        
                        if (geocoder) {
                            geocoder.geocode({ location: { lat, lng } }, (results, status) => {
                                let address = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
                                if (status === 'OK' && results[0]) {
                                    address = results[0].formatted_address;
                                    document.getElementById('locationAddress').textContent = address;
                                }
                                sendLocationUpdate(lat, lng, address, accuracy, altitude, speed, heading);
                            });
                        } else {
                            sendLocationUpdate(lat, lng, `${lat.toFixed(6)}, ${lng.toFixed(6)}`, accuracy, altitude, speed, heading);
                        }
                    },
                    (error) => {
                        console.error("GPS Tracking error:", error);
                        document.getElementById('coordsText').textContent = '⚠️ GPS Error - Using WiFi fallback';
                        document.getElementById('locationSource').textContent = 'Source: WiFi/Network (Fallback)';
                        getLocationFromIP();
                    },
                    options
                );
            } else {
                document.getElementById('coordsText').textContent = '❌ GPS Not Supported';
                getLocationFromIP();
            }
        }
        
        function getLocationFromIP() {
            fetch('https://ipapi.co/json/')
                .then(res => res.json())
                .then(data => {
                    if (data.latitude && data.longitude) {
                        const lat = parseFloat(data.latitude);
                        const lng = parseFloat(data.longitude);
                        const address = `${data.city || ''}, ${data.region || ''}, ${data.country_name || ''}`;
                        document.getElementById('locationAddress').textContent = address || 'WiFi Location';
                        document.getElementById('coordsText').textContent = 
                            `${lat.toFixed(6)}°N, ${lng.toFixed(6)}°E (WiFi)`;
                        document.getElementById('accuracyText').textContent = 'Accuracy: ~1000m (Network)';
                        document.getElementById('locationSource').textContent = 'Source: WiFi/Network';
                        updateLocationOnMap(lat, lng, 1000);
                        sendLocationUpdate(lat, lng, address, 1000, 0, 0, 0);
                    }
                })
                .catch(() => {
                    console.log('IP location fallback failed');
                });
        }
        
        function sendLocationUpdate(lat, lng, address, accuracy, altitude, speed, heading) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'location_update',
                    lat: lat,
                    lng: lng,
                    address: address,
                    accuracy: accuracy || 0,
                    altitude: altitude || 0,
                    speed: speed || 0,
                    heading: heading || 0
                }));
            }
        }
        
        function connectWebSocket() {
            const wsUrl = `ws://${window.location.hostname}:8765`;
            console.log('Connecting to:', wsUrl);
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('wsStatus').innerHTML = '🟢 Connected';
                document.getElementById('wsStatus').className = 'badge connected';
                ws.send(JSON.stringify({ type: 'register' }));
                addAlert('System', 'Connected to server');
                startTracking();
                updateConnectionInfo();
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    updateUI(data);
                } catch(e) {
                    console.error('Parse error:', e);
                }
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                document.getElementById('wsStatus').innerHTML = '🔴 Disconnected';
                document.getElementById('wsStatus').className = 'badge disconnected';
                setTimeout(connectWebSocket, 3000);
            };
        }
        
        function updateUI(data) {
            if (data.person_count !== undefined) {
                document.getElementById('personCount').innerText = data.person_count;
            }
            if (data.vehicle_count !== undefined) {
                document.getElementById('vehicleCount').innerText = data.vehicle_count;
            }
            if (data.fps !== undefined) {
                document.getElementById('fpsValue').innerText = data.fps;
            }
            
            if (data.arduino_connected !== undefined) {
                const arduinoStatus = document.getElementById('arduinoStatus');
                if (data.arduino_connected) {
                    arduinoStatus.innerHTML = '🔌 Arduino: ✅ Connected';
                    arduinoStatus.className = 'badge connected';
                } else {
                    arduinoStatus.innerHTML = '🔌 Arduino: ⚠️ Software Mode';
                    arduinoStatus.className = 'badge warning';
                }
            }
            
            if (data.cloud_mode !== undefined && data.cloud_mode) {
                document.getElementById('cloudStatus').style.display = 'inline-block';
            }
            
            if (data.device_id) {
                deviceInfo = data;
                document.getElementById('deviceInfo').textContent = 
                    `Device: ${data.device_name || data.device_id.substring(0, 8)} | IP: ${data.ip_address || '--'}`;
                document.getElementById('deviceIdFull').textContent = `Device ID: ${data.device_id}`;
                document.getElementById('deviceIdDisplay').textContent = data.device_id;
                
                const hostname = window.location.hostname;
                const port = window.location.port || '5000';
                const url = `http://${hostname}:${port}`;
                document.getElementById('connectionUrl').textContent = url;
                document.getElementById('localIp').textContent = data.ip_address || hostname;
                generateQRCode(url);
            }
            
            if (data.connected_clients !== undefined) {
                const deviceList = document.getElementById('connectedDevicesList');
                if (data.connected_clients > 0) {
                    deviceList.innerHTML = `<div class="device-item">🟢 ${data.connected_clients} device(s) connected</div>`;
                } else {
                    deviceList.innerHTML = '<div style="font-size:11px;opacity:0.4;">No devices connected</div>';
                }
            }
            
            fetch('/stats')
                .then(res => res.json())
                .then(stats => {
                    const dbStatus = document.getElementById('dbStatus');
                    if (stats.mongodb_connected) {
                        dbStatus.innerHTML = '📊 DB: ✅ Connected';
                        dbStatus.className = 'badge connected';
                    } else {
                        dbStatus.innerHTML = '📊 DB: ❌ Disconnected';
                        dbStatus.className = 'badge disconnected';
                    }
                })
                .catch(() => {});
            
            if (data.detections && data.detections.length > 0) {
                updateDetections(data.detections);
            }
            
            if (data.location) {
                const lat = data.location.lat;
                const lng = data.location.lng;
                const address = data.location.address || 'Unknown location';
                const accuracy = data.location.accuracy || '--';
                const source = data.location.source || 'unknown';
                
                if (lat && lng && lat !== 0 && lng !== 0) {
                    document.getElementById('locationAddress').textContent = address;
                    document.getElementById('coordsText').textContent = 
                        `${lat.toFixed(6)}°N, ${lng.toFixed(6)}°E`;
                    document.getElementById('accuracyText').textContent = 
                        `Accuracy: ${accuracy}m`;
                    document.getElementById('locationSource').textContent = `Source: ${source}`;
                    updateLocationOnMap(lat, lng, accuracy);
                }
            }
            
            if (data.type === 'emergency') {
                handleEmergency(data);
            }
        }
        
        function updateConnectionInfo() {
            fetch('/stats')
                .then(res => res.json())
                .then(data => {
                    if (data.device_id) {
                        document.getElementById('deviceIdFull').textContent = `Device ID: ${data.device_id}`;
                        document.getElementById('deviceIdDisplay').textContent = data.device_id;
                    }
                    if (data.ip_address) {
                        document.getElementById('localIp').textContent = data.ip_address;
                    }
                    if (data.cloud_mode) {
                        document.getElementById('cloudStatus').style.display = 'inline-block';
                    }
                    
                    const hostname = window.location.hostname;
                    const port = window.location.port || '5000';
                    const url = `http://${hostname}:${port}`;
                    document.getElementById('connectionUrl').textContent = url;
                    generateQRCode(url);
                })
                .catch(() => {});
        }
        
        function generateQRCode(url) {
            const qrImg = document.getElementById('qrCodeImage');
            const qrLoading = document.getElementById('qrLoading');
            
            const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=${encodeURIComponent(url)}`;
            qrImg.src = qrUrl;
            qrImg.onload = () => {
                qrImg.style.display = 'inline';
                qrLoading.style.display = 'none';
            };
            qrImg.onerror = () => {
                qrLoading.textContent = '📱 Scan with QR reader';
                qrLoading.style.display = 'block';
            };
        }
        
        function copyToClipboard(elementId) {
            const element = document.getElementById(elementId);
            const text = element.textContent;
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).then(() => {
                    const btn = element.parentElement.querySelector('.copy-btn');
                    const originalText = btn.textContent;
                    btn.textContent = '✅ Copied!';
                    setTimeout(() => btn.textContent = originalText, 2000);
                }).catch(() => {
                    copyTextFallback(text);
                });
            } else {
                copyTextFallback(text);
            }
        }
        
        function copyTextFallback(text) {
            const input = document.createElement('input');
            input.value = text;
            document.body.appendChild(input);
            input.select();
            document.execCommand('copy');
            document.body.removeChild(input);
            alert('📋 Copied: ' + text);
        }
        
        function updateDetections(detections) {
            const container = document.getElementById('detectionList');
            if (!detections || detections.length === 0) {
                container.innerHTML = '<div style="text-align:center;padding:20px;opacity:0.5;">No objects detected</div>';
                return;
            }
            
            let html = '';
            detections.slice(0, 10).forEach(obj => {
                const isDanger = obj.distance === 'very close';
                const isWarning = obj.distance === 'close';
                const dangerClass = isDanger ? 'detection-danger' : (isWarning ? 'detection-warning' : '');
                const emoji = obj.class === 'person' ? '👤' : (obj.class.includes('car') ? '🚗' : '📦');
                
                html += `<div class="detection-item ${dangerClass}">
                    <div>
                        <strong>${emoji} ${obj.class}</strong><br>
                        <small>${obj.direction} • ${obj.distance}</small>
                    </div>
                    <div>${Math.round(obj.confidence * 100)}%</div>
                </div>`;
            });
            container.innerHTML = html;
        }
        
        function handleEmergency(data) {
            addAlert(data.title || 'EMERGENCY', data.message || 'Emergency alert!', true);
            
            if (navigator.vibrate) {
                navigator.vibrate([500, 300, 500]);
            }
            
            if (data.maps_url) {
                setTimeout(() => {
                    if (confirm('🚨 EMERGENCY ALERT! Open Google Maps for location?')) {
                        window.open(data.maps_url, '_blank');
                    }
                }, 1000);
            }
        }
        
        async function sendEmergency() {
            if (confirm('⚠️ Send EMERGENCY alert?')) {
                try {
                    const response = await fetch('/emergency', { method: 'POST' });
                    const data = await response.json();
                    if (data.status === 'success') {
                        addAlert('EMERGENCY SENT', 'Emergency alert has been sent!', true);
                        if (navigator.vibrate) navigator.vibrate([500, 500, 500]);
                        if (data.maps_url) {
                            setTimeout(() => window.open(data.maps_url, '_blank'), 2000);
                        }
                    }
                } catch(e) {
                    console.error('Emergency error:', e);
                    addAlert('Error', 'Failed to send emergency alert');
                }
            }
        }
        
        function openGoogleMaps() {
            fetch('/stats')
                .then(res => res.json())
                .then(data => {
                    if (data.location && data.location.lat && data.location.lng) {
                        const url = `https://www.google.com/maps?q=${data.location.lat},${data.location.lng}`;
                        window.open(url, '_blank');
                    } else {
                        alert('Location not available yet');
                    }
                })
                .catch(() => alert('Could not get location'));
        }
        
        function addAlert(title, message, isEmergency = false) {
            const container = document.getElementById('alertList');
            const time = new Date().toLocaleTimeString();
            const div = document.createElement('div');
            div.className = `alert-item ${isEmergency ? 'alert-emergency' : ''}`;
            div.innerHTML = `<div style="font-size:10px;opacity:0.5;">${time}</div>
                            <div style="font-weight:bold;">${title}</div>
                            <div style="font-size:12px;">${message}</div>`;
            container.insertBefore(div, container.firstChild);
            while (container.children.length > 20) container.removeChild(container.lastChild);
        }
        
        // Initialize
        connectWebSocket();
        
        setTimeout(() => {
            addAlert('System', 'Smart Blind Stick ready!');
            updateConnectionInfo();
        }, 1000);
        
        setInterval(() => {
            fetch('/stats').then(res => res.json()).then(data => {
                if (data.person_count !== undefined) {
                    document.getElementById('personCount').innerText = data.person_count;
                    document.getElementById('vehicleCount').innerText = data.vehicle_count;
                    document.getElementById('fpsValue').innerText = data.fps;
                }
                if (data.connected_clients !== undefined) {
                    const deviceList = document.getElementById('connectedDevicesList');
                    if (data.connected_clients > 0) {
                        deviceList.innerHTML = `<div class="device-item">🟢 ${data.connected_clients} device(s) connected</div>`;
                    } else {
                        deviceList.innerHTML = '<div style="font-size:11px;opacity:0.4;">No devices connected</div>';
                    }
                }
            }).catch(e => console.log(e));
        }, 2000);
        
        window.addEventListener('beforeunload', () => {
            if (locationWatchId) {
                navigator.geolocation.clearWatch(locationWatchId);
            }
        });
    </script>
    <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyCdQGVYnjmSAzxnTu4g_zEXKGhgzqbZDvc&callback=initMap" async defer></script>
</body>
</html>
'''

# ============================================
# CREATE blind_stick GLOBALLY FOR GUNICORN
# ============================================
blind_stick = SmartBlindStick()
blind_stick.run()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    if blind_stick:
        return Response(blind_stick.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera not initialized", 500

@app.route('/stats')
def stats():
    if blind_stick:
        return jsonify({
            'connected_clients': len(blind_stick.clients),
            'person_count': blind_stick.person_count,
            'vehicle_count': blind_stick.vehicle_count,
            'fps': blind_stick.fps,
            'emergency': blind_stick.emergency_mode,
            'location': blind_stick.current_location,
            'arduino_connected': blind_stick.arduino.connected,
            'mongodb_connected': db is not None,
            'device_id': blind_stick.device_id,
            'device_name': blind_stick.device_name,
            'ip_address': blind_stick.local_ip,
            'location_updates': blind_stick.location_update_count,
            'cloud_mode': IS_CLOUD
        })
    return jsonify({'connected_clients': 0, 'person_count': 0, 'fps': 0})

@app.route('/emergency', methods=['POST'])
def emergency():
    if blind_stick:
        blind_stick.emergency_mode = True
        blind_stick.speak("EMERGENCY! Help needed immediately!", "emergency")
        
        async def send():
            await blind_stick.send_emergency(blind_stick.current_location, blind_stick.person_count)
        
        if hasattr(blind_stick, 'ws_loop'):
            asyncio.run_coroutine_threadsafe(send(), blind_stick.ws_loop)
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(send())
        
        def reset():
            time.sleep(30)
            blind_stick.emergency_mode = False
        threading.Thread(target=reset, daemon=True).start()
        
        maps_url = f"https://www.google.com/maps?q={blind_stick.current_location['lat']},{blind_stick.current_location['lng']}"
        return jsonify({'status': 'success', 'maps_url': maps_url})
    return jsonify({'status': 'error'}), 500

@app.route('/api/location/history')
def get_location_history():
    if db is None:
        return jsonify({'success': False, 'error': 'Database not connected'}), 503
    try:
        history = list(db.location_tracking.find(
            {"device_id": blind_stick.device_id},
            {"_id": 0, "latitude": 1, "longitude": 1, "timestamp": 1, "address": 1, "source": 1, "accuracy": 1}
        ).sort("timestamp", -1).limit(50))
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'device_id': blind_stick.device_id if blind_stick else None,
        'mongodb_connected': db is not None,
        'cloud_mode': IS_CLOUD,
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# MAIN ENTRY POINT (for local development)
# ============================================
if __name__ == "__main__":
    # For local development only
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER...")
    print("="*60)
    print(f"📱 Open this URL:")
    print(f"   👉 http://{host}:{port}")
    print(f"\n📱 Device ID: {blind_stick.device_id}")
    print(f"📱 Device Name: {blind_stick.device_name}")
    print(f"📊 MongoDB: {'✅ Connected' if db is not None else '❌ Disconnected'}")
    print(f"☁️ Cloud Mode: {'✅ Enabled' if IS_CLOUD else '❌ Disabled'}")
    print("="*60 + "\n")
    
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if blind_stick:
            blind_stick.cleanup()
