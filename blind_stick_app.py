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
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from ultralytics import YOLO
import pyttsx3
import math

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Your Google Maps API Key
GOOGLE_MAPS_API_KEY = "AIzaSyCdQGVYnjmSAzxnTu4g_zEXKGhgzqbZDvc"

# Try to import serial for Arduino (optional)
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ PySerial not installed. Arduino features disabled.")
    print("   Install with: pip install pyserial")

# Arduino Manager Class (with better error handling)
class ArduinoManager:
    def __init__(self, port=None, baudrate=9600):
        self.serial_connection = None
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        self.last_alert_time = 0
        self.alert_cooldown = 0.3
        self.enabled = SERIAL_AVAILABLE
        
    def find_arduino_port(self):
        """Automatically find Arduino port"""
        if not self.enabled:
            return None
            
        try:
            ports = serial.tools.list_ports.comports()
            
            # Filter out virtual/Bluetooth/modem serial ports
            valid_ports = []
            for port in ports:
                desc = port.description.lower()
                device = port.device
                
                # Check for Bluetooth or other wireless links to exclude
                if any(x in desc for x in ['bluetooth', 'bt link', 'bthnum', 'hands-free', 'wireless', 'modem']):
                    continue
                valid_ports.append(port)
            
            # Try to find a port with explicit USB or Arduino keywords
            for port in valid_ports:
                desc = port.description.lower()
                if any(x in desc for x in ['arduino', 'usb', 'ch340', 'cp210', 'uart', 'ftdi', 'pl2303', 'prolific']):
                    return port.device
            
            # If no matches with explicit keywords, check other remaining non-Bluetooth ports
            for port in valid_ports:
                try:
                    test_ser = serial.Serial(port.device, 9600, timeout=0.5)
                    test_ser.close()
                    return port.device
                except:
                    continue
            return None
        except Exception as e:
            print(f"Error finding Arduino: {e}")
            return None
    
    def connect(self):
        """Connect to Arduino"""
        if not self.enabled:
            print("⚠️ Serial library not available. Running without Arduino.")
            return False
            
        try:
            if self.port is None:
                self.port = self.find_arduino_port()
                if self.port is None:
                    print("⚠️ Arduino not found! Running in software-only mode.")
                    print("   Connect Arduino or install drivers if you want hardware feedback.")
                    return False
            
            print(f"🔌 Attempting to connect to Arduino on {self.port}...")
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            
            # Try to read response
            start_time = time.time()
            while time.time() - start_time < 3:
                if self.serial_connection.in_waiting:
                    response = self.serial_connection.readline().decode().strip()
                    if response == "READY":
                        self.connected = True
                        print(f"✅ Arduino connected on {self.port}")
                        return True
                time.sleep(0.1)
            
            # If no response but connection is open, consider it connected
            if self.serial_connection.is_open:
                self.connected = True
                print(f"✅ Arduino connected on {self.port} (no response)")
                return True
                
        except Exception as e:
            print(f"⚠️ Arduino connection error: {e}")
            print("   Running without Arduino hardware feedback")
        
        self.connected = False
        return False
    
    def send_alert(self, alert_type, distance):
        """Send alert command to Arduino"""
        if not self.connected or not self.enabled or self.serial_connection is None:
            return False
        
        # Check cooldown
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
        """Stop all alerts"""
        if not self.connected or not self.enabled or self.serial_connection is None:
            return
        try:
            self.serial_connection.write(b"STOP\n")
        except:
            pass
    
    def test_arduino(self):
        """Test Arduino connection"""
        if not self.connected:
            return False
        try:
            self.serial_connection.write(b"TEST\n")
            return True
        except:
            return False
    
    def close(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.stop_alert()
                self.serial_connection.close()
            except:
                pass
            print("🔌 Arduino disconnected")

# MongoDB Manager Class (with fallback if not available)
class MongoDBManager:
    def __init__(self, connection_string=None):
        self.enabled = False
        try:
            if connection_string is None:
                connection_string = "mongodb://localhost:27017/"
            
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=2000)
            # Test connection
            self.client.server_info()
            self.db = self.client['blind_stick_db']
            self.enabled = True
            print("✅ MongoDB connected successfully!")
        except Exception as e:
            print(f"⚠️ MongoDB not available: {e}")
            print("   Running without database logging")
            self.client = None
            self.enabled = False
    
    def save_alert(self, alert_type, message, location, person_count):
        if not self.enabled:
            return False
        try:
            self.db.alerts.insert_one({
                'alert_type': alert_type,
                'message': message,
                'location': location,
                'timestamp': datetime.now()
            })
            return True
        except:
            return False
    
    def save_detection(self, object_type, confidence, distance, direction):
        if not self.enabled:
            return False
        try:
            self.db.detections.insert_one({
                'object_type': object_type,
                'confidence': confidence,
                'distance': distance,
                'direction': direction,
                'timestamp': datetime.now()
            })
            return True
        except:
            return False
    
    def save_emergency(self, location, person_count, detections):
        if not self.enabled:
            return False
        try:
            self.db.emergency_events.insert_one({
                'location': location,
                'person_count': person_count,
                'timestamp': datetime.now()
            })
            return True
        except:
            return False
    
    def log_system_event(self, event_type, details):
        if not self.enabled:
            return False
        try:
            self.db.system_logs.insert_one({
                'event_type': event_type,
                'details': details,
                'timestamp': datetime.now()
            })
            return True
        except:
            return False
    
    def get_stats(self):
        if not self.enabled:
            return {'total_detections': 0, 'total_alerts': 0}
        try:
            return {
                'total_detections': self.db.detections.count_documents({}),
                'total_alerts': self.db.alerts.count_documents({})
            }
        except:
            return {'total_detections': 0, 'total_alerts': 0}
    
    def get_recent_detections(self, limit=50):
        return []
    
    def get_recent_alerts(self, limit=50):
        return []
    
    def close(self):
        if self.client:
            self.client.close()

class SmartBlindStick:
    def __init__(self, mongodb_uri=None):
        print("\n" + "="*60)
        print("🦯 Initializing Smart Blind Stick System")
        print("="*60)
        
        self.db = MongoDBManager(mongodb_uri)
        
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
        
        # Load YOLO model
        print("\n📷 Loading YOLO model...")
        try:
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded!")
        except Exception as e:
            print(f"⚠️ YOLO not available: {e}")
            self.model = None
        
        # Important classes for detection
        self.important_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 11: 'stop sign'
        }
        
        # Camera setup
        print("\n🎥 Opening camera...")
        self.cap = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap = cap
                print(f"✅ Camera {i} opened successfully!")
                break
            else:
                cap.release()
        
        if self.cap is None:
            print("❌ No camera found! Using test pattern.")
            # Create a blank frame generator
            self.use_test_pattern = True
        else:
            self.use_test_pattern = False
        
        self.clients = set()
        self.current_data = {}
        self.emergency_mode = False
        self.person_count = 0
        self.vehicle_count = 0
        self.detected_objects = []
        self.fps = 0
        self.detection_count = 0
        
        # Demo location
        self.current_location = {
            "lat": 11.2745,
            "lng": 77.5831,
            "address": "Perundurai, Tamil Nadu, India",
            "source": "demo"
        }
        
        # Start threads
        threading.Thread(target=self.process_speech_queue, daemon=True).start()
        
        self.db.log_system_event('SYSTEM_START', 'Smart Blind Stick system initialized')
        
        print("\n" + "="*60)
        print("✅ SYSTEM READY!")
        print(f"   Arduino: {'Connected' if self.arduino.connected else 'Not Connected (Software Mode)'}")
        print(f"   Camera: {'OK' if self.cap else 'Test Pattern'}")
        print(f"   TTS: {'OK' if self.tts_available else 'Disabled'}")
        print(f"   Database: {'OK' if self.db.enabled else 'Disabled'}")
        print("="*60 + "\n")
    
    def process_speech_queue(self):
        """Process speech queue"""
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
        """Speak with cooldown"""
        now = time.time()
        if alert_type in self.last_spoken:
            if now - self.last_spoken[alert_type] < self.speech_cooldown.get(alert_type, 2):
                return
        self.last_spoken[alert_type] = now
        self.speech_queue.put(text)
        
        if alert_type != 'system':
            self.db.save_alert(alert_type, text, self.current_location, self.person_count)
        
        print(f"🔊 Speaking: {text}")
    
    def send_arduino_alert(self, alert_type, distance_cm):
        """Send alert to Arduino"""
        if self.arduino.connected and distance_cm < 100:  # Only send for close objects
            self.arduino.send_alert(alert_type, distance_cm)
    
    def detect_with_yolo(self, frame):
        """Detect objects using YOLO"""
        detections = []
        height, width = frame.shape[:2]
        
        if self.model is None:
            return frame, detections
        
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
                    
                    # Estimate distance
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
                        'direction': direction,
                        'bbox': (x1, y1, x2, y2)
                    }
                    detections.append(detection)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {conf:.2f} ({distance}, {direction})"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Trigger alerts
                    if class_name == 'person':
                        if distance == "very close":
                            self.speak(f"Person {direction}, very close!", 'person')
                            self.send_arduino_alert('PERSON', distance_cm)
                        elif distance == "close":
                            self.speak(f"Person {direction}", 'person')
                            self.send_arduino_alert('PERSON', distance_cm)
                        self.db.save_detection('person', conf, distance, direction)
                        self.detection_count += 1
                        
                    elif class_name in ['car', 'truck', 'bus', 'bicycle']:
                        if distance in ["very close", "close"]:
                            self.speak(f"Vehicle {direction}, {distance}!", 'vehicle')
                            self.send_arduino_alert('VEHICLE', distance_cm)
                        self.db.save_detection('vehicle', conf, distance, direction)
                        self.detection_count += 1
        
        return frame, detections
    
    def generate_frames(self):
        """Generate video frames"""
        fps_start = time.time()
        frame_count = 0
        
        while True:
            if self.use_test_pattern or self.cap is None:
                # Create test pattern
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "SMART BLIND STICK SYSTEM", (150, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Waiting for camera...", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Connect a camera to see real-time detection", (150, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                detections = []
                
            else:
                ret, frame = self.cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera Error", (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - fps_start
                        self.fps = int(30 / elapsed) if elapsed > 0 else 30
                        fps_start = time.time()
                    
                    frame, detections = self.detect_with_yolo(frame)
            
            # Update counts
            self.person_count = sum(1 for d in detections if d['class'] == 'person')
            self.vehicle_count = sum(1 for d in detections if d['class'] in ['car', 'truck', 'bus', 'bicycle'])
            self.detected_objects = detections
            
            # Draw overlay information
            y_offset = 30
            cv2.putText(frame, "SMART BLIND STICK SYSTEM", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Arduino status
            if self.arduino.connected:
                cv2.putText(frame, "Arduino: ✅ Connected", (10, y_offset + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Arduino: ⚠️ Not Connected (Software Mode)", (10, y_offset + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            cv2.putText(frame, f"FPS: {self.fps} | Persons: {self.person_count} | Vehicles: {self.vehicle_count}", 
                       (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"Mobile Connected: {len(self.clients)}", (10, y_offset + 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"Location: {self.current_location['address'][:40]}", (10, y_offset + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            
            if self.emergency_mode:
                cv2.putText(frame, "EMERGENCY MODE ACTIVE", (10, y_offset + 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Prepare data for WebSocket
            self.current_data = {
                'detections': detections,
                'person_count': self.person_count,
                'vehicle_count': self.vehicle_count,
                'fps': self.fps,
                'emergency': self.emergency_mode,
                'location': self.current_location,
                'timestamp': datetime.now().isoformat(),
                'connected_clients': len(self.clients),
                'detection_count': self.detection_count,
                'arduino_connected': self.arduino.connected
            }
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    async def handle_client(self, websocket):
        """Handle WebSocket clients"""
        self.clients.add(websocket)
        print(f"📱 Mobile connected! Total: {len(self.clients)}")
        
        try:
            # Send initial data
            if self.current_data:
                await websocket.send(json.dumps(self.current_data))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'register':
                        await websocket.send(json.dumps({'type': 'registered', 'status': 'ok'}))
                    elif data.get('type') == 'request_location':
                        await websocket.send(json.dumps({
                            'type': 'location_update',
                            'location': self.current_location
                        }))
                except:
                    pass
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.clients.remove(websocket)
            print(f"📱 Mobile disconnected. Total: {len(self.clients)}")
    
    async def broadcast_updates(self):
        """Broadcast updates to all clients"""
        while True:
            if self.clients and self.current_data:
                dead = set()
                for client in self.clients:
                    try:
                        await client.send(json.dumps(self.current_data))
                    except:
                        dead.add(client)
                self.clients -= dead
            await asyncio.sleep(0.1)
    
    def run_websocket(self):
        """Run WebSocket server"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def server():
            async with websockets.serve(self.handle_client, '0.0.0.0', 8765):
                print("🔌 WebSocket server running on ws://0.0.0.0:8765")
                await asyncio.gather(self.broadcast_updates(), asyncio.Future())
        
        loop.run_until_complete(server())
    
    async def send_emergency(self, location, person_count):
        """Send emergency alert"""
        maps_url = f"https://www.google.com/maps?q={location['lat']},{location['lng']}"
        
        emergency_data = {
            'type': 'emergency',
            'title': '🚨 EMERGENCY ALERT! 🚨',
            'message': f'Emergency button pressed! Immediate assistance needed!',
            'location': location,
            'maps_url': maps_url,
            'person_count': person_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n{'='*60}")
        print("🚨 EMERGENCY ALERT SENT!")
        print(f"{'='*60}")
        print(f"📍 Location: {location['address']}")
        print(f"📍 Coordinates: {location['lat']}, {location['lng']}")
        print(f"📍 Google Maps: {maps_url}")
        print(f"👥 Persons detected: {person_count}")
        
        # Trigger Arduino emergency
        self.arduino.send_alert('EMERGENCY', 0)
        
        self.db.save_emergency(location, person_count, [])
        self.db.save_alert('emergency', f'Emergency alert', location, person_count)
        
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
        """Start the system"""
        ws_thread = threading.Thread(target=self.run_websocket, daemon=True)
        ws_thread.start()
        
        self.speak("Smart Blind Stick system started", "system")
        
        # Get local IP
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print("\n" + "="*60)
        print("🌐 SERVER RUNNING!")
        print("="*60)
        print(f"📱 Open on your MOBILE PHONE: http://{local_ip}:5000")
        print(f"💻 Open on this computer: http://127.0.0.1:5000")
        print(f"🔌 WebSocket: ws://{local_ip}:8765")
        print("\n💡 TIPS:")
        print("   • If Arduino not connected, system runs in software-only mode")
        print("   • Connect Arduino to USB for physical feedback (buzzer + vibration)")
        print("   • Press 'E' key on keyboard for emergency alert")
        print("   • The system will detect: persons, vehicles, bicycles")
        print("="*60 + "\n")
    
    def cleanup(self):
        """Clean up resources"""
        self.arduino.close()
        if self.cap:
            self.cap.release()

# HTML Template (simplified for testing)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🦯 Smart Blind Stick</h1>
        
        <div class="status">
            <span id="wsStatus" class="badge disconnected">🔴 Connecting...</span>
            <span id="arduinoStatus" class="badge disconnected">🔌 Arduino: Unknown</span>
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
            <h3>📍 Location</h3>
            <div id="locationInfo" class="location">Getting location...</div>
            <button onclick="openGoogleMaps()" style="width:100%; margin-top:10px; padding:10px; background:#4caf50; border:none; border-radius:8px; color:white; cursor:pointer;">🗺️ Open in Google Maps</button>
        </div>
    </div>
    
    <script>
        let ws = null;
        
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
                    arduinoStatus.className = 'badge disconnected';
                }
            }
            
            if (data.detections && data.detections.length > 0) {
                updateDetections(data.detections);
            }
            
            if (data.location) {
                const locText = `📍 ${data.location.address || 'Unknown location'}<br>📌 ${data.location.lat}, ${data.location.lng}`;
                document.getElementById('locationInfo').innerHTML = locText;
            }
            
            if (data.type === 'emergency') {
                handleEmergency(data);
            }
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
                const emoji = obj.class === 'person' ? '👤' : (obj.class === 'car' ? '🚗' : '📦');
                
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
                    if (data.location) {
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
        
        // Initial alert
        setTimeout(() => {
            addAlert('System', 'Smart Blind Stick ready!');
        }, 1000);
        
        // Update stats periodically
        setInterval(() => {
            fetch('/stats').then(res => res.json()).then(data => {
                if (data.person_count !== undefined) {
                    document.getElementById('personCount').innerText = data.person_count;
                    document.getElementById('vehicleCount').innerText = data.vehicle_count;
                    document.getElementById('fpsValue').innerText = data.fps;
                }
            }).catch(e => console.log(e));
        }, 2000);
    </script>
</body>
</html>
'''

blind_stick = None

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
            'arduino_connected': blind_stick.arduino.connected
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
            # Run async in new loop
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

if __name__ == "__main__":
    blind_stick = SmartBlindStick()
    blind_stick_thread = threading.Thread(target=blind_stick.run, daemon=True)
    blind_stick_thread.start()
    time.sleep(2)
    
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER...")
    print("="*60)
    print("📱 Open this URL on your MOBILE PHONE:")
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n   👉 http://{local_ip}:5000")
    print(f"   👉 http://127.0.0.1:5000 (same computer)")
    
    print("\n💡 If Arduino is not connected:")
    print("   • System will run in SOFTWARE-ONLY mode")
    print("   • Alerts will be visual and voice-only")
    print("   • Connect Arduino for buzzer/vibration feedback")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if blind_stick:
            blind_stick.cleanup()