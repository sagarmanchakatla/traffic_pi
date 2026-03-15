"""
OPTIMAL MULTI-PROCESS ARCHITECTURE
===================================

Process 1 (Traffic): Captures frames → Sends to Queue → Continues immediately
Process 2 (Accident): Reads from Queue → Processes independently → No blocking!
Process 3 (Monitor): Independent monitoring

Key: Frames shared via Queue, processes run in PARALLEL
"""

import cv2
import time
import subprocess
import psutil
import multiprocessing as mp
from multiprocessing import Process, Event, Value, Queue
from ctypes import c_bool
import numpy as np
from services.yolo.yolo_detector import YOLODetector
from services.traffic.traffic_optimizer import TrafficTimingOptimizer
from services.traffic.traffic_lights import TrafficLightService
from services.accident_detection.accident_detection import AccidentDetection
from services.system_monitor.system_monitor import SystemMonitor

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
# ---------------------------------------


# ==================== PROCESS 1: TRAFFIC CONTROLLER ====================

def traffic_controller_process(
    emergency_event: Event,
    throttle_mode: Value,
    status_queue: Queue,
    frame_queue: Queue,  # ← NEW: Send frames to accident detector
    camera_indices: list
):
    """
    Traffic controller - owns cameras, sends frames to accident detector
    DOES NOT WAIT for accident detection - continues immediately!
    """
    process_name = "TRAFFIC"
    print(f"[{process_name}] Process started (PID: {mp.current_process().pid})")
    
    try:
        # Initialize
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
            print(f"[{process_name}] ERROR: No cameras!")
            return
        
        # Initialize lights
        lights = None
        if ENABLE_PHYSICAL_LIGHTS:
            try:
                lights = TrafficLightService(list(cameras.keys()))
                if lights.enabled:
                    lights.test_sequence()
                    print(f"[{process_name}] ✓ Lights ready")
            except Exception as e:
                print(f"[{process_name}] Light init failed: {e}")
        
        lane_config = {lane: {"hasLeft": False} for lane in cameras.keys()}
        
        # State
        current_timings = None
        current_priority = []
        cycle_start_time = None
        current_cycle_duration = 0
        phase_index = 0
        phase_start_time = None
        last_yellow_check = 0
        last_red_check = 0
        last_accident_frame_sent = 0
        
        print(f"[{process_name}] Starting traffic control...")
        
        # Main loop
        while not emergency_event.is_set():
            current_time = time.time()
            
            # ===== SEND FRAMES TO ACCIDENT DETECTOR (NON-BLOCKING!) =====
            if ENABLE_ACCIDENT_DETECTION and (current_time - last_accident_frame_sent >= ACCIDENT_CHECK_INTERVAL):
                print(f"\n[{process_name}] Capturing frames for accident detection...")
                
                # Capture frames
                snapshots = {}
                capture_start = time.time()
                for lane, cap in cameras.items():
                    ret, frame = cap.read()
                    if ret:
                        snapshots[lane] = frame
                
                capture_time = time.time() - capture_start
                print(f"[{process_name}] Frame capture: {capture_time:.3f}s")
                
                # Send to accident detector (NON-BLOCKING!)
                try:
                    frame_queue.put_nowait({
                        'timestamp': current_time,
                        'snapshots': snapshots
                    })
                    print(f"[{process_name}] ✓ Frames sent to accident detector")
                    last_accident_frame_sent = current_time
                except:
                    print(f"[{process_name}] ⚠️  Accident detector queue full, skipping")
                
                # ← IMPORTANT: We continue immediately! No waiting!
            # ================================================================
            
            # ===== CALCULATE NEW CYCLE =====
            if current_timings is None or time.time() - cycle_start_time >= current_cycle_duration:
                print(f"\n[{process_name}] {'='*60}")
                print(f"[{process_name}] CALCULATING NEW CYCLE")
                print(f"[{process_name}] {'='*60}")
                
                if lights:
                    lights.set_all_red()
                
                # Capture frames
                snapshots = {}
                for lane, cap in cameras.items():
                    ret, frame = cap.read()
                    if ret:
                        if throttle_mode.value:
                            frame = cv2.resize(frame, (320, 320))
                        snapshots[lane] = frame
                
                # Detect vehicles
                lane_counts = {}
                for lane, frame in snapshots.items():
                    _, img_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    result = detector.detect(img_bytes.tobytes())
                    
                    counts = {}
                    for det in result["detections"]:
                        counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1
                    lane_counts[lane] = counts
                    print(f"[{process_name}]   {lane}: {sum(counts.values())} vehicles")
                
                # Optimize
                optimization = optimizer.optimize(lane_counts, lane_config)
                current_timings = optimization["timings"]
                current_priority = optimization["priority"]
                current_cycle_duration = optimization["totalTime"]
                
                print(f"[{process_name}] Cycle duration: {current_cycle_duration}s")
                
                cycle_start_time = time.time()
                phase_index = 0
                phase_start_time = time.time()
                
                # Start first phase
                if current_priority:
                    first_lane = current_priority[0]
                    timing = current_timings[first_lane]
                    print(f"[{process_name}] PHASE 1: {first_lane.upper()} GREEN {timing['green']}s")
                    if lights:
                        lights.set_color(first_lane, 'green')
            
            # ===== PHASE MANAGEMENT =====
            if phase_index < len(current_priority):
                lane = current_priority[phase_index]
                timing = current_timings[lane]
                phase_elapsed = current_time - phase_start_time
                
                # Green → Yellow
                if phase_elapsed >= timing['green']:
                    if current_time - last_yellow_check > 0.5:
                        if lights and lights.current_state.get(lane) != 'yellow':
                            print(f"[{process_name}] {lane.upper()}: GREEN → YELLOW")
                            lights.set_color(lane, 'yellow')
                            last_yellow_check = current_time
                
                # Yellow → Red
                if phase_elapsed >= timing['green'] + timing['yellow']:
                    if current_time - last_red_check > 0.5:
                        if lights and lights.current_state.get(lane) != 'red':
                            print(f"[{process_name}] {lane.upper()}: YELLOW → RED")
                            lights.set_color(lane, 'red')
                        
                        phase_index += 1
                        
                        if phase_index < len(current_priority):
                            next_lane = current_priority[phase_index]
                            next_timing = current_timings[next_lane]
                            print(f"[{process_name}] PHASE {phase_index + 1}: {next_lane.upper()} GREEN {next_timing['green']}s")
                            if lights:
                                lights.set_color(next_lane, 'green')
                            phase_start_time = time.time()
                        
                        last_red_check = current_time
            
            # Status update
            try:
                status_queue.put_nowait({
                    'process': 'traffic',
                    'status': 'running'
                })
            except:
                pass
            
            time.sleep(0.05)  # ← Very short sleep, precise timing!
        
        # Cleanup
        print(f"\n[{process_name}] EMERGENCY - All lights RED")
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


# ==================== PROCESS 2: ACCIDENT DETECTION ====================

def accident_detection_process(
    emergency_event: Event,
    status_queue: Queue,
    frame_queue: Queue  # ← NEW: Receive frames from traffic controller
):
    """
    Accident detector - receives frames from traffic controller
    Processes independently WITHOUT blocking traffic!
    """
    process_name = "ACCIDENT"
    print(f"[{process_name}] Process started (PID: {mp.current_process().pid})")
    
    try:
        # Load model
        print(f"[{process_name}] Loading accident detection model...")
        detector = AccidentDetection()
        print(f"[{process_name}] ✓ Model loaded")
        print(f"[{process_name}] Waiting for frames from traffic controller...")
        
        # Main loop - wait for frames
        while not emergency_event.is_set():
            try:
                # Wait for frames (blocking, but only blocks THIS process!)
                frame_data = frame_queue.get(timeout=5)
                
                if frame_data is None:
                    continue
                
                timestamp = frame_data['timestamp']
                snapshots = frame_data['snapshots']
                
                print(f"\n[{process_name}] {'='*60}")
                print(f"[{process_name}] RECEIVED FRAMES - Running detection...")
                print(f"[{process_name}] {'='*60}")
                
                # Convert frames to bytes
                snapshot_bytes = {}
                for lane, frame in snapshots.items():
                    _, img_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    snapshot_bytes[lane] = img_bytes.tobytes()
                
                # Run detection (takes time, but doesn't block traffic!)
                detection_start = time.time()
                results = detector.detect_lanes(snapshot_bytes)
                detection_time = time.time() - detection_start
                
                print(f"[{process_name}] Detection time: {detection_time:.2f}s")
                print(f"[{process_name}] Latency: {time.time() - timestamp:.2f}s")
                
                # Check results
                accident_found = False
                for lane, result in results.items():
                    if result == "ACCIDENT":
                        print(f"\n[{process_name}] 🚨 ACCIDENT ON {lane.upper()}!")
                        accident_found = True
                        break
                
                if accident_found:
                    # Trigger emergency
                    print(f"[{process_name}] TRIGGERING EMERGENCY")
                    emergency_event.set()
                    
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
                
            except mp.queues.Empty:
                # Timeout - no frames received, continue waiting
                continue
            except Exception as e:
                print(f"[{process_name}] Detection error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[{process_name}] Process stopped")
        
    except Exception as e:
        print(f"[{process_name}] ERROR: {e}")
        import traceback
        traceback.print_exc()


# ==================== PROCESS 3: SYSTEM MONITOR ====================

def system_monitor_process(
    emergency_event: Event,
    throttle_mode: Value,
    status_queue: Queue,
    monitor_interval: int
):
    """System monitoring"""
    process_name = "MONITOR"
    print(f"[{process_name}] Process started (PID: {mp.current_process().pid})")
    
    while not emergency_event.is_set():
        time.sleep(monitor_interval)
        
        if emergency_event.is_set():
            break
        
        temp = SystemMonitor.get_cpu_temp()
        cpu = SystemMonitor.get_cpu_usage()
        memory = psutil.virtual_memory().percent
        
        should_throttle = SystemMonitor.should_throttle(temp, cpu)
        throttle_mode.value = should_throttle
        
        SystemMonitor.log_stats(temp, cpu, memory, process_name)
        
        status_queue.put({
            'process': 'monitor',
            'temp': temp,
            'cpu': cpu,
            'memory': memory
        })
    
    print(f"[{process_name}] Process stopped")


# ==================== MAIN ====================

def main():
    print("\n" + "="*70)
    print("RASPBERRY PI TRAFFIC MANAGEMENT SYSTEM")
    print("OPTIMAL 3-PROCESS ARCHITECTURE WITH FRAME SHARING")
    print("="*70 + "\n")
    
    # Initial check
    temp = SystemMonitor.get_cpu_temp()
    cpu = SystemMonitor.get_cpu_usage()
    memory = psutil.virtual_memory().percent
    
    print(f"[MAIN] Initial status:")
    SystemMonitor.log_stats(temp, cpu, memory, "MAIN")
    
    # Discover cameras
    print("\n[MAIN] Discovering cameras...")
    camera_indices = []
    for idx in range(MAX_CAMERA_INDEX):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            camera_indices.append(idx)
            print(f"[MAIN]   ✓ Camera {idx}")
            cap.release()
    
    if not camera_indices:
        print("[MAIN] ERROR: No cameras!")
        return
    
    print(f"[MAIN] Found {len(camera_indices)} camera(s)\n")
    
    # Shared state
    emergency_event = Event()
    throttle_mode = Value(c_bool, False)
    status_queue = Queue(maxsize=100)
    frame_queue = Queue(maxsize=2)  # ← NEW: Frame sharing queue (small buffer)
    
    # Create processes
    processes = []
    
    # Process 1: Traffic Controller
    p1 = Process(
        target=traffic_controller_process,
        args=(emergency_event, throttle_mode, status_queue, frame_queue, camera_indices),
        name="TrafficController"
    )
    processes.append(p1)
    
    # Process 2: Accident Detection
    if ENABLE_ACCIDENT_DETECTION:
        p2 = Process(
            target=accident_detection_process,
            args=(emergency_event, status_queue, frame_queue),
            name="AccidentDetection"
        )
        processes.append(p2)
    
    # Process 3: System Monitor
    if ENABLE_SYSTEM_MONITOR:
        p3 = Process(
            target=system_monitor_process,
            args=(emergency_event, throttle_mode, status_queue, MONITOR_INTERVAL),
            name="SystemMonitor"
        )
        processes.append(p3)
    
    # Start all
    print(f"[MAIN] Starting {len(processes)} processes...")
    for p in processes:
        p.start()
        print(f"[MAIN]   ✓ {p.name} (PID: {p.pid})")
    
    print(f"\n[MAIN] All processes running")
    print(f"[MAIN] Architecture: Traffic sends frames → Accident detects (parallel)")
    print(f"[MAIN] Press Ctrl+C to stop\n")
    
    try:
        while True:
            if emergency_event.is_set():
                print("\n[MAIN] 🚨 EMERGENCY DETECTED!")
                print("[MAIN] Shutting down...")
                break
            
            all_alive = all(p.is_alive() for p in processes)
            if not all_alive:
                print("\n[MAIN] ⚠️  Process died")
                for p in processes:
                    if not p.is_alive():
                        print(f"[MAIN]   ✗ {p.name} terminated")
                break
            
            # Process status updates
            try:
                while not status_queue.empty():
                    status = status_queue.get_nowait()
            except:
                pass
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C detected")
    
    finally:
        emergency_event.set()
        
        # Send poison pill to frame queue
        try:
            frame_queue.put(None, timeout=1)
        except:
            pass
        
        print("\n[MAIN] Waiting for processes...")
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"[MAIN]   Force terminating {p.name}...")
                p.terminate()
                p.join(timeout=2)
                if p.is_alive():
                    p.kill()
        
        temp = SystemMonitor.get_cpu_temp()
        print(f"\n[MAIN] Final temperature: {temp:.1f}°C")
        print("[MAIN] Shutdown complete\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()