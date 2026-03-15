# Process 1 - Traffic signal controller

import subprocess
import psutil
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value
from ctypes import c_bool
import cv2
import time

from services.traffic.traffic_optimizer import TrafficTimingOptimizer
from services.yolo.yolo_detector import YOLODetector
from services.traffic.traffic_lights import TrafficLightService

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


def traffic_controller_process(
    emergency_event: Event,
    throttle_mode: Value,
    status_queue: Queue,
    camera_indices: list
):
    """
    Main traffic signal controller process
    Runs independently, only checks emergency_event
    """
    process_name = "TRAFFIC"
    print(f"[{process_name}] Process started (PID: {mp.current_process().pid})")
    
    try:
        # Initialize components
        print(f"[{process_name}] Loading YOLO model...")
        detector = YOLODetector()
        optimizer = TrafficTimingOptimizer()
        
        # Initialize cameras
        print(f"[{process_name}] Initializing cameras...")
        cameras = {}
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                lane_name = f"lane{len(cameras) + 1}"
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cameras[lane_name] = cap
                print(f"[{process_name}]   ✓ {lane_name}")
            else:
                cap.release()
        
        if not cameras:
            print(f"[{process_name}] ERROR: No cameras detected!")
            return
        
        # Initialize traffic lights
        lights = None
        if ENABLE_PHYSICAL_LIGHTS:
            try:
                lights = TrafficLightService(list(cameras.keys()))
                if lights.enabled:
                    # pass
                    lights.test_sequence()
                    print(f"[{process_name}] ✓ Physical lights ready")
            except Exception as e:
                print(f"[{process_name}] Light init failed: {e}")
                lights = None
        
        lane_config = {lane: {"hasLeft": False} for lane in cameras.keys()}
        
        # Traffic control state
        current_timings = None
        current_priority = []
        cycle_start_time = None
        current_cycle_duration = 0
        phase_index = 0
        phase_start_time = None
        last_yellow_check = 0
        last_red_check = 0
        
        print(f"[{process_name}] Starting traffic control loop...")
        print(f"[{process_name}] Emergency monitoring: ENABLED")
        
        # Main control loop
        while not emergency_event.is_set():
            
            # ===== CALCULATE NEW CYCLE =====
            if current_timings is None or time.time() - cycle_start_time >= current_cycle_duration:
                print(f"\n[{process_name}] {'='*60}")
                print(f"[{process_name}] CALCULATING NEW CYCLE")
                print(f"[{process_name}] {'='*60}")
                
                if lights:
                    lights.set_all_red()
                
                # Capture and detect
                snapshots = {}
                for lane, cap in cameras.items():
                    ret, frame = cap.read()
                    if ret:
                        if throttle_mode.value:
                            frame = cv2.resize(frame, (320, 320))
                        snapshots[lane] = frame
                
                # Count vehicles
                lane_counts = {}
                for lane, frame in snapshots.items():
                    _, img_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    result = detector.detect(img_bytes.tobytes())
                    
                    counts = {}
                    for det in result["detections"]:
                        name = det["class_name"]
                        counts[name] = counts.get(name, 0) + 1
                    lane_counts[lane] = counts
                    
                    total = sum(counts.values())
                    print(f"[{process_name}]   {lane}: {total} vehicles")
                
                # Optimize timings
                optimization = optimizer.optimize(lane_counts, lane_config)
                current_timings = optimization["timings"]
                current_priority = optimization["priority"]
                current_cycle_duration = optimization["totalTime"]
                
                print(f"[{process_name}] New cycle duration: {current_cycle_duration}s")
                
                cycle_start_time = time.time()
                phase_index = 0
                phase_start_time = time.time()
                
                # Start first phase
                if current_priority:
                    first_lane = current_priority[0]
                    timing = current_timings[first_lane]
                    print(f"[{process_name}] PHASE 1: {first_lane.upper()} GREEN for {timing['green']}s")
                    if lights:
                        lights.set_color(first_lane, 'green')
            
            # ===== PHASE MANAGEMENT =====
            current_time = time.time()
            
            if phase_index < len(current_priority):
                lane = current_priority[phase_index]
                timing = current_timings[lane]
                phase_elapsed = current_time - phase_start_time
                
                # Green → Yellow transition
                if phase_elapsed >= timing['green']:
                    if current_time - last_yellow_check > 0.5:
                        if lights and lights.current_state.get(lane) != 'yellow':
                            print(f"[{process_name}] {lane.upper()}: GREEN → YELLOW")
                            lights.set_color(lane, 'yellow')
                            last_yellow_check = current_time
                
                # Yellow → Red transition
                yellow_end = timing['green'] + timing['yellow']
                if phase_elapsed >= yellow_end:
                    if current_time - last_red_check > 0.5:
                        if lights and lights.current_state.get(lane) != 'red':
                            print(f"[{process_name}] {lane.upper()}: YELLOW → RED")
                            lights.set_color(lane, 'red')
                        
                        phase_index += 1
                        
                        if phase_index < len(current_priority):
                            next_lane = current_priority[phase_index]
                            next_timing = current_timings[next_lane]
                            print(f"[{process_name}] PHASE {phase_index + 1}: {next_lane.upper()} GREEN for {next_timing['green']}s")
                            if lights:
                                lights.set_color(next_lane, 'green')
                            phase_start_time = time.time()
                        
                        last_red_check = current_time
            
            # Send status update
            try:
                status_queue.put_nowait({
                    'process': 'traffic',
                    'status': 'running',
                    'cycle_progress': (time.time() - cycle_start_time) / current_cycle_duration if current_cycle_duration > 0 else 0
                })
            except:
                pass  # Queue full, skip
            
            # Short sleep
            time.sleep(0.05)
        
        # Emergency shutdown
        print(f"\n[{process_name}] EMERGENCY DETECTED - Setting all lights to RED")
        if lights:
            lights.set_all_red()
            lights.cleanup()
        
        for cap in cameras.values():
            cap.release()
        
        print(f"[{process_name}] Process stopped")
        
    except Exception as e:
        print(f"[{process_name}] ERROR: {e}")
        import traceback
        traceback.print_exc()
 
 