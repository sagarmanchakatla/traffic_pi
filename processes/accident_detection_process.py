 
import subprocess
import psutil
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value
from ctypes import c_bool 
import cv2

from services.accident_detection.accident_detection import AccidentDetection

# ---------------- CONFIG ----------------
FRAME_WIDTH = 416
FRAME_HEIGHT = 416
TEMP_THRESHOLD = 75
CPU_THRESHOLD = 80
DISPLAY = False
MAX_CAMERA_INDEX = 6
ADAPTIVE_MODE = True
ENABLE_PHYSICAL_LIGHTS = True
 
ENABLE_ACCIDENT_DETECTION = True
ACCIDENT_CHECK_INTERVAL = 90  # seconds
 
ENABLE_SYSTEM_MONITOR = True
MONITOR_INTERVAL = 30  # seconds


def accident_detection_process(
    emergency_event: Event,
    status_queue: Queue,
    camera_indices: list,
    check_interval: int
):
    """
    Accident detection service process
    Runs periodic checks, completely isolated from traffic controller
    """
    process_name = "ACCIDENT"
    print(f"[{process_name}] Process started (PID: {mp.current_process().pid})")
    print(f"[{process_name}] Check interval: {check_interval}s")
    
    try:
        # Load accident detection model
        print(f"[{process_name}] Loading accident detection model...")
        detector = AccidentDetection()
        print(f"[{process_name}] ✓ Model loaded")
        
        # Initialize cameras (separate from traffic controller)
        print(f"[{process_name}] Initializing cameras...")
        cameras = {}
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                lane_name = f"lane{len(cameras) + 1}"
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cameras[lane_name] = cap
                print(f"[{process_name}]   ✓ {lane_name}")
            else:
                cap.release()
        
        if not cameras:
            print(f"[{process_name}] ERROR: No cameras detected!")
            return
        
        print(f"[{process_name}] Starting detection loop...")
        
        # Main detection loop
        while not emergency_event.is_set():
            time.sleep(check_interval)
            
            if emergency_event.is_set():
                break
            
            print(f"\n[{process_name}] {'='*60}")
            print(f"[{process_name}] RUNNING ACCIDENT CHECK")
            print(f"[{process_name}] {'='*60}")
            
            # Capture frames
            snapshots = {}
            for lane, cap in cameras.items():
                ret, frame = cap.read()
                if ret:
                    snapshots[lane] = frame
            
            # Convert to bytes
            snapshot_bytes = {}
            for lane, frame in snapshots.items():
                _, img_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                snapshot_bytes[lane] = img_bytes.tobytes()
            
            # Run detection
            detection_start = time.time()
            results = detector.detect_lanes(snapshot_bytes)
            detection_time = time.time() - detection_start
            
            print(f"[{process_name}] Detection time: {detection_time:.2f}s")
            
            # Check results
            accident_found = False
            for lane, result in results.items():
                if result == "ACCIDENT":
                    print(f"\n[{process_name}] 🚨 ACCIDENT DETECTED ON {lane.upper()}!")
                    accident_found = True
                    break
            
            if accident_found:
                # Trigger emergency
                print(f"[{process_name}] TRIGGERING EMERGENCY EVENT")
                emergency_event.set()
                
                # Send alert
                status_queue.put({
                    'process': 'accident',
                    'status': 'EMERGENCY',
                    'lane': lane
                })
                break
            else:
                print(f"[{process_name}] ✓ No accidents detected")
                print(f"[{process_name}] {'='*60}\n")
                
                status_queue.put({
                    'process': 'accident',
                    'status': 'ok',
                    'last_check': time.time()
                })
        
        # Cleanup
        for cap in cameras.values():
            cap.release()
        
        print(f"[{process_name}] Process stopped")
        
    except Exception as e:
        print(f"[{process_name}] ERROR: {e}")
        import traceback
        traceback.print_exc()
 
 