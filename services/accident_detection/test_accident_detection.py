#!/usr/bin/env python3
"""
USB Camera Accident Detection Test Script
Tests accident detection on 4 USB cameras (lanes)
Optimized for Raspberry Pi 5

Features:
- Real-time system monitoring (CPU, RAM, Temperature)
- Support for both TFLite (optimized) and original Keras models
- Multi-lane accident detection
- Live video display with stats overlay
"""

import cv2
import numpy as np
import time
import threading
import os
import subprocess
import psutil
from datetime import datetime
from collections import deque

# Try TFLite first (optimized)
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

# Always import Keras for fallback
try:
    from keras.models import model_from_json
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# ==================== CONFIGURATION ====================

class Config:
    """
    System Configuration
    
    TO USE OPTIMIZED MODEL (Recommended):
        USE_TFLITE = True
        
    TO USE ORIGINAL MODEL (Slower):
        USE_TFLITE = False
    """
    
    # ========== MODEL SELECTION (CHANGE THIS!) ==========
    USE_TFLITE = False  # True = TFLite (fast), False = Keras (slow)
    # ====================================================
    
    # Model paths
    TFLITE_MODEL = "accident_model_quantized.tflite"  # Optimized model
    JSON_MODEL = "model.json"  # Original architecture
    WEIGHTS_MODEL = "model_weights.h5"  # Original weights
    
    # Camera settings
    MAX_CAMERAS = 6  # Scan up to 6 camera indices
    FRAME_WIDTH = 416
    FRAME_HEIGHT = 416
    CAMERA_FPS = 15
    
    # Detection settings
    ACCIDENT_THRESHOLD = 80  # % confidence to log
    EMERGENCY_THRESHOLD = 98  # % for emergency
    COOLDOWN_SECONDS = 3  # Seconds between same-lane logs
    
    # Display
    SHOW_VIDEO = True  # Set False for headless
    SHOW_STATS_OVERLAY = True  # Show system stats on video
    
    # System monitoring
    STATS_UPDATE_INTERVAL = 2  # Update system stats every N seconds
    
    # Logging
    MAX_LOGS = 50

config = Config()

# ==================== SYSTEM MONITOR ====================

class SystemMonitor:
    """Monitor Raspberry Pi system resources"""
    
    def __init__(self):
        self.temp = 0.0
        self.cpu = 0.0
        self.ram_mb = 0
        self.ram_pct = 0.0
        self.last_update = 0
        self.lock = threading.Lock()
        
        # Start monitoring thread
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def _get_cpu_temp(self):
        """Get CPU temperature"""
        try:
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True,
                text=True,
                timeout=1
            )
            temp_str = result.stdout.strip()
            temp = float(temp_str.split('=')[1].split("'")[0])
            return temp
        except:
            return 0.0
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                temp = self._get_cpu_temp()
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                
                with self.lock:
                    self.temp = temp
                    self.cpu = cpu
                    self.ram_mb = mem.used / (1024 * 1024)
                    self.ram_pct = mem.percent
                    self.last_update = time.time()
                
            except Exception as e:
                print(f"[MONITOR] Error: {e}")
            
            time.sleep(config.STATS_UPDATE_INTERVAL)
    
    def get_stats(self):
        """Get current system stats"""
        with self.lock:
            return {
                'temp': self.temp,
                'cpu': self.cpu,
                'ram_mb': self.ram_mb,
                'ram_pct': self.ram_pct
            }
    
    def get_status_string(self):
        """Get formatted status string"""
        stats = self.get_stats()
        return (f"Temp: {stats['temp']:.1f}°C | "
                f"CPU: {stats['cpu']:.1f}% | "
                f"RAM: {stats['ram_mb']:.0f}MB ({stats['ram_pct']:.1f}%)")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

# ==================== ACCIDENT DETECTOR ====================

class AccidentDetector:
    """Accident detection with model selection support"""
    
    class_names = ['Accident', 'No Accident']
    
    def __init__(self):
        self.using_tflite = False
        self.interpreter = None
        self.model = None
        self.model_name = "Unknown"
        self.model_size_mb = 0
        self.load_time = 0
        
        print(f"\n[DETECTOR] Model Selection: {'TFLite (Optimized)' if config.USE_TFLITE else 'Keras (Original)'}")
        
        start_time = time.time()
        
        # Try to load based on user preference
        if config.USE_TFLITE:
            # User wants TFLite
            if self._try_load_tflite():
                self.load_time = time.time() - start_time
                return
            else:
                print("[DETECTOR] ⚠️  TFLite not available, falling back to Keras")
        
        # Load Keras (either as preference or fallback)
        if self._try_load_keras():
            self.load_time = time.time() - start_time
            return
        
        # Nothing worked
        raise FileNotFoundError(
            "No model found!\n"
            "  For TFLite: Need accident_model_quantized.tflite\n"
            "  For Keras: Need model.json + model_weights.h5"
        )
    
    def _try_load_tflite(self):
        """Try to load TFLite model"""
        if not TFLITE_AVAILABLE:
            print("[DETECTOR] ✗ TFLite runtime not available")
            print("[DETECTOR]   Install with: pip install tflite-runtime")
            return False
        
        if not os.path.exists(config.TFLITE_MODEL):
            print(f"[DETECTOR] ✗ TFLite model not found: {config.TFLITE_MODEL}")
            print(f"[DETECTOR]   Run optimize_model.py to create it")
            return False
        
        try:
            print(f"[DETECTOR] Loading TFLite: {config.TFLITE_MODEL}")
            
            # Get file size
            self.model_size_mb = os.path.getsize(config.TFLITE_MODEL) / (1024 * 1024)
            
            # Load model
            self.interpreter = tflite.Interpreter(model_path=config.TFLITE_MODEL)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.is_quantized = self.input_details[0]['dtype'] == np.uint8
            
            self.using_tflite = True
            self.model_name = "TFLite INT8 Quantized" if self.is_quantized else "TFLite Float32"
            
            print(f"[DETECTOR] ✓ Loaded {self.model_name}")
            print(f"[DETECTOR]   Size: {self.model_size_mb:.1f} MB")
            print(f"[DETECTOR]   Quantized: {self.is_quantized}")
            
            return True
            
        except Exception as e:
            print(f"[DETECTOR] ✗ TFLite loading failed: {e}")
            return False
    
    def _try_load_keras(self):
        """Try to load Keras model"""
        if not KERAS_AVAILABLE:
            print("[DETECTOR] ✗ Keras not available")
            print("[DETECTOR]   Install with: pip install tensorflow")
            return False
        
        if not os.path.exists(config.JSON_MODEL):
            print(f"[DETECTOR] ✗ JSON model not found: {config.JSON_MODEL}")
            return False
        
        if not os.path.exists(config.WEIGHTS_MODEL):
            print(f"[DETECTOR] ✗ Weights not found: {config.WEIGHTS_MODEL}")
            return False
        
        try:
            print(f"[DETECTOR] Loading Keras: {config.JSON_MODEL} + {config.WEIGHTS_MODEL}")
            
            # Get file sizes
            json_size = os.path.getsize(config.JSON_MODEL) / (1024 * 1024)
            weights_size = os.path.getsize(config.WEIGHTS_MODEL) / (1024 * 1024)
            self.model_size_mb = json_size + weights_size
            
            # Load architecture
            with open(config.JSON_MODEL, "r") as f:
                model_json = f.read()
            
            self.model = model_from_json(model_json)
            self.model.load_weights(config.WEIGHTS_MODEL)
            self.model.make_predict_function()
            
            self.using_tflite = False
            self.model_name = "Keras Original"
            
            print(f"[DETECTOR] ✓ Loaded {self.model_name}")
            print(f"[DETECTOR]   Size: {self.model_size_mb:.1f} MB")
            print(f"[DETECTOR]   ⚠️  Consider optimizing with optimize_model.py")
            
            return True
            
        except Exception as e:
            print(f"[DETECTOR] ✗ Keras loading failed: {e}")
            return False
    
    def predict(self, frame):
        """
        Predict accident from frame
        
        Args:
            frame: BGR image (any size)
            
        Returns:
            prediction: "Accident" or "No Accident"
            probability: float (0-100)
        """
        # Resize to 250x250
        resized = cv2.resize(frame, (250, 250))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        if self.using_tflite:
            return self._predict_tflite(rgb)
        else:
            return self._predict_keras(rgb)
    
    def _predict_tflite(self, rgb_img):
        """TFLite inference"""
        # Prepare input
        if self.is_quantized:
            input_data = rgb_img.astype(np.uint8)
        else:
            input_data = (rgb_img / 255.0).astype(np.float32)
        
        input_data = np.expand_dims(input_data, axis=0)
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize if needed
        if self.is_quantized:
            scale, zero_point = self.output_details[0]['quantization']
            if scale != 0:
                output = scale * (output.astype(np.float32) - zero_point)
        
        # Softmax
        if output.max() > 1.0:
            exp_out = np.exp(output - np.max(output))
            output = exp_out / np.sum(exp_out)
        
        # Get prediction
        pred_idx = np.argmax(output[0])
        prediction = self.class_names[pred_idx]
        probability = float(output[0][pred_idx] * 100)
        
        return prediction, probability
    
    def _predict_keras(self, rgb_img):
        """Keras inference"""
        input_data = np.expand_dims(rgb_img, axis=0)
        output = self.model.predict(input_data, verbose=0)
        
        pred_idx = np.argmax(output[0])
        prediction = self.class_names[pred_idx]
        probability = float(output[0][pred_idx] * 100)
        
        return prediction, probability

# ==================== ACCIDENT LOGGER ====================

class AccidentLogger:
    """Thread-safe accident logging"""
    
    def __init__(self):
        self.logs = deque(maxlen=config.MAX_LOGS)
        self.last_log_time = {}
        self.lock = threading.Lock()
    
    def log(self, lane, probability):
        """Log accident with cooldown"""
        with self.lock:
            now = time.time()
            
            # Check cooldown
            if lane in self.last_log_time:
                if now - self.last_log_time[lane] < config.COOLDOWN_SECONDS:
                    return False
            
            self.last_log_time[lane] = now
            
            # Determine severity
            is_emergency = probability >= config.EMERGENCY_THRESHOLD
            
            log_entry = {
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'lane': lane,
                'probability': round(probability, 1),
                'severity': 'EMERGENCY' if is_emergency else ('high' if probability > 80 else 'medium'),
                'emergency': is_emergency
            }
            
            self.logs.append(log_entry)
            
            # Print alert
            if is_emergency:
                print(f"🆘 EMERGENCY! {lane}: {probability:.1f}%")
            else:
                print(f"⚠️  Accident {lane}: {probability:.1f}%")
            
            return True
    
    def get_recent(self, n=10):
        """Get recent logs"""
        with self.lock:
            return list(self.logs)[-n:]

# ==================== CAMERA WORKER ====================

class CameraWorker:
    """Background worker for camera processing"""
    
    def __init__(self, lane_name, camera_index, detector, logger):
        self.lane_name = lane_name
        self.camera_index = camera_index
        self.detector = detector
        self.logger = logger
        
        self.running = False
        self.thread = None
        self.camera = None
        
        # Stats
        self.frames_processed = 0
        self.last_fps_check = time.time()
        self.fps = 0
        
        # Latest frame for display
        self.latest_frame = None
        self.latest_prediction = "No Accident"
        self.latest_probability = 0.0
        self.lock = threading.Lock()
    
    def start(self):
        """Start processing"""
        # Open camera
        self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        
        if not self.camera.isOpened():
            print(f"[ERROR] Cannot open camera {self.camera_index} for {self.lane_name}")
            return False
        
        # Configure camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Start thread
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        print(f"[{self.lane_name}] Started on camera {self.camera_index}")
        return True
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            ret, frame = self.camera.read()
            
            if not ret:
                print(f"[{self.lane_name}] Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Detect accident
            try:
                prediction, probability = self.detector.predict(frame)
                
                # Log if accident detected
                if prediction == "Accident" and probability >= config.ACCIDENT_THRESHOLD:
                    self.logger.log(self.lane_name, probability)
                
                # Update display frame
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_prediction = prediction
                    self.latest_probability = probability
                
                self.frames_processed += 1
                
                # Calculate FPS
                now = time.time()
                if now - self.last_fps_check >= 1.0:
                    self.fps = self.frames_processed / (now - self.last_fps_check)
                    self.frames_processed = 0
                    self.last_fps_check = now
                
            except Exception as e:
                print(f"[{self.lane_name}] Detection error: {e}")
            
            time.sleep(0.01)  # Small delay
    
    def get_display_frame(self):
        """Get annotated frame for display with system stats"""
        with self.lock:
            if self.latest_frame is None:
                return None
            
            frame = self.latest_frame.copy()
            pred = self.latest_prediction
            prob = self.latest_probability
        
        # Determine color
        if pred == "Accident":
            if prob >= config.EMERGENCY_THRESHOLD:
                color = (0, 0, 255)  # Red - Emergency
            elif prob > 80:
                color = (0, 69, 255)  # Orange-red
            else:
                color = (0, 165, 255)  # Orange
            text_color = (255, 255, 255)
        else:
            color = (0, 255, 0)  # Green - No accident
            text_color = (0, 0, 0)
        
        # Draw main overlay
        cv2.rectangle(frame, (0, 0), (300, 80), color, -1)
        
        # Status text
        status_text = f"{pred} {prob:.1f}%"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # Lane name
        cv2.putText(frame, self.lane_name.upper(), (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # FPS counter (top right)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame.shape[1] - 120, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def stop(self):
        """Stop processing"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.camera:
            self.camera.release()

# ==================== CAMERA DISCOVERY ====================

def discover_cameras(max_index=6):
    """Discover USB cameras"""
    cameras = {}
    lane_id = 1
    
    print("\n[INFO] Scanning for USB cameras...")
    
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        
        if cap.isOpened():
            lane_name = f"lane{lane_id}"
            cameras[lane_name] = idx
            print(f"  ✓ Camera {idx} → {lane_name}")
            cap.release()
            lane_id += 1
        else:
            cap.release()
    
    print(f"[INFO] Found {len(cameras)} camera(s)\n")
    return cameras

# ==================== MAIN APPLICATION ====================

def add_system_stats_overlay(frame, system_monitor, detector):
    """Add system stats overlay to frame"""
    if not config.SHOW_STATS_OVERLAY:
        return frame
    
    stats = system_monitor.get_stats()
    
    # Create semi-transparent overlay at bottom
    overlay = frame.copy()
    height = frame.shape[0]
    cv2.rectangle(overlay, (0, height - 90), (frame.shape[1], height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # System stats text
    y_pos = height - 65
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    
    # Line 1: Temperature and CPU
    text1 = f"Temperature: {stats['temp']:.1f}C  |  CPU: {stats['cpu']:.1f}%"
    cv2.putText(frame, text1, (10, y_pos), font, font_scale, color, thickness)
    
    # Line 2: RAM and Model
    text2 = f"RAM: {stats['ram_mb']:.0f}MB ({stats['ram_pct']:.1f}%)  |  Model: {detector.model_name}"
    cv2.putText(frame, text2, (10, y_pos + 25), font, font_scale, color, thickness)
    
    # Line 3: Model size and load time
    text3 = f"Model Size: {detector.model_size_mb:.1f}MB  |  Load Time: {detector.load_time:.2f}s"
    cv2.putText(frame, text3, (10, y_pos + 50), font, font_scale, color, thickness)
    
    return frame

def main():
    print("\n" + "="*70)
    print("USB CAMERA ACCIDENT DETECTION TEST")
    print("Raspberry Pi 5 - Multi-Lane Detection with System Monitoring")
    print("="*70 + "\n")
    
    # Print configuration
    print("[CONFIG] Model Selection:")
    print(f"  USE_TFLITE: {config.USE_TFLITE}")
    print(f"  Using: {'TFLite (Optimized)' if config.USE_TFLITE else 'Keras (Original)'}\n")
    
    # Start system monitor
    print("[MONITOR] Starting system monitoring...")
    system_monitor = SystemMonitor()
    time.sleep(1)  # Let it collect first stats
    
    initial_stats = system_monitor.get_stats()
    print(f"[MONITOR] Initial: Temp={initial_stats['temp']:.1f}°C, "
          f"CPU={initial_stats['cpu']:.1f}%, RAM={initial_stats['ram_mb']:.0f}MB\n")
    
    # Initialize detector
    try:
        detector = AccidentDetector()
        print(f"[DETECTOR] Model loaded in {detector.load_time:.2f}s\n")
    except Exception as e:
        print(f"[ERROR] Failed to load detector: {e}")
        print("\nMake sure you have either:")
        print("  - accident_model_quantized.tflite (optimized)")
        print("  - model.json + model_weights.h5 (original)")
        print("\nTo create optimized model, run: python optimize_model.py")
        system_monitor.stop()
        return
    
    # Discover cameras
    cameras = discover_cameras(config.MAX_CAMERAS)
    
    if not cameras:
        print("[ERROR] No cameras detected!")
        print("\nTroubleshooting:")
        print("  1. Check USB cameras are connected")
        print("  2. Run: ls /dev/video*")
        print("  3. Try: v4l2-ctl --list-devices")
        system_monitor.stop()
        return
    
    # Create logger
    logger = AccidentLogger()
    
    # Create workers
    workers = {}
    for lane_name, camera_index in cameras.items():
        worker = CameraWorker(lane_name, camera_index, detector, logger)
        if worker.start():
            workers[lane_name] = worker
    
    if not workers:
        print("[ERROR] No workers started!")
        system_monitor.stop()
        return
    
    print(f"\n[INFO] Running accident detection on {len(workers)} cameras")
    print(f"[INFO] Model: {detector.model_name} ({detector.model_size_mb:.1f}MB)")
    print("[INFO] Press 'q' to quit\n")
    
    # Print system stats header
    print("="*70)
    print("SYSTEM MONITORING")
    print("="*70)
    
    last_stats_print = time.time()
    
    # Main display loop
    try:
        while True:
            # Print stats to console every 5 seconds
            if time.time() - last_stats_print >= 5:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {system_monitor.get_status_string()}")
                
                for lane_name in sorted(workers.keys()):
                    worker = workers[lane_name]
                    print(f"  {lane_name}: {worker.latest_prediction:12} "
                          f"({worker.latest_probability:5.1f}%)  FPS: {worker.fps:4.1f}")
                
                recent = logger.get_recent(5)
                if recent:
                    print(f"  Recent accidents: {len(recent)}")
                
                last_stats_print = time.time()
            
            if config.SHOW_VIDEO:
                # Display all camera feeds
                display_frames = []
                
                for lane_name in sorted(workers.keys()):
                    worker = workers[lane_name]
                    frame = worker.get_display_frame()
                    
                    if frame is not None:
                        # Resize for display
                        display_frame = cv2.resize(frame, (400, 300))
                        display_frames.append(display_frame)
                
                # Arrange in grid
                if len(display_frames) == 1:
                    combined = display_frames[0]
                elif len(display_frames) == 2:
                    combined = np.hstack(display_frames)
                elif len(display_frames) == 3:
                    top = np.hstack(display_frames[:2])
                    bottom = display_frames[2]
                    # Pad bottom to match width
                    pad_width = top.shape[1] - bottom.shape[1]
                    if pad_width > 0:
                        bottom = np.hstack([bottom, np.zeros((bottom.shape[0], pad_width, 3), dtype=np.uint8)])
                    combined = np.vstack([top, bottom])
                elif len(display_frames) >= 4:
                    top = np.hstack(display_frames[:2])
                    bottom = np.hstack(display_frames[2:4])
                    combined = np.vstack([top, bottom])
                else:
                    combined = display_frames[0] if display_frames else np.zeros((300, 400, 3), dtype=np.uint8)
                
                # Add system stats overlay
                combined = add_system_stats_overlay(combined, system_monitor, detector)
                
                cv2.imshow("Accident Detection - All Lanes", combined)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Headless mode - just sleep
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        print("\n[INFO] Shutting down...")
        
        for worker in workers.values():
            worker.stop()
        
        system_monitor.stop()
        cv2.destroyAllWindows()
        
        # Final stats
        final_stats = system_monitor.get_stats()
        
        print("\n" + "="*70)
        print("FINAL STATISTICS")
        print("="*70)
        
        print(f"\nSystem Stats:")
        print(f"  Final Temperature: {final_stats['temp']:.1f}°C")
        print(f"  Final CPU Usage: {final_stats['cpu']:.1f}%")
        print(f"  Final RAM Usage: {final_stats['ram_mb']:.0f}MB ({final_stats['ram_pct']:.1f}%)")
        
        print(f"\nModel Info:")
        print(f"  Type: {detector.model_name}")
        print(f"  Size: {detector.model_size_mb:.1f}MB")
        print(f"  Load Time: {detector.load_time:.2f}s")
        
        total_frames = sum(w.frames_processed for w in workers.values())
        total_accidents = len(logger.logs)
        
        print(f"\nProcessing Stats:")
        print(f"  Total frames processed: {total_frames}")
        print(f"  Total accidents detected: {total_accidents}")
        
        if total_accidents > 0:
            print(f"\nRecent Accidents:")
            for log in logger.get_recent(10):
                print(f"  {log['timestamp']} - {log['lane']}: "
                      f"{log['probability']}% ({log['severity']})")
        
        print("\n" + "="*70)
        print("[INFO] Shutdown complete\n")

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()
