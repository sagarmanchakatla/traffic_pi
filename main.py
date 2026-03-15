# Main script that runs all the processes and services

import cv2
import time
import subprocess
import psutil
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value
from ctypes import c_bool
# from services.yolo.yolo_detector import YOLODetector
# from services.traffic.traffic_optimizer import TrafficTimingOptimizer
# from services.traffic.traffic_lights import TrafficLightService
# from services.accident_detection.accident_detection import AccidentDetection
from services.system_monitor.system_monitor import SystemMonitor

from processes.accident_detection_process import accident_detection_process
from processes.system_monitor_process import system_monitor_process
from processes.traffic_signal_controller import traffic_controller_process

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


def main():
    print("\n" + "="*70)
    print("RASPBERRY PI TRAFFIC MANAGEMENT SYSTEM")
    print("MULTI-PROCESS ARCHITECTURE")
    print("="*70 + "\n")
    
    # Initial system check
    temp = SystemMonitor.get_cpu_temp()
    cpu = SystemMonitor.get_cpu_usage()
    memory = psutil.virtual_memory().percent
    
    print(f"[MAIN] Initial status:")
    SystemMonitor.log_stats(temp, cpu, memory)
    
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
        print("[MAIN] ERROR: No cameras detected!")
        return
    
    print(f"[MAIN] Found {len(camera_indices)} camera(s)\n")
    
    # Shared state between processes
    emergency_event = Event()  # Emergency shutdown signal
    throttle_mode = Value(c_bool, False)  # Adaptive throttling flag
    status_queue = Queue(maxsize=100)  # Status updates from processes
    
    # Create processes
    processes = []
    
    # Process 1: Traffic Controller
    p1 = Process(
        target=traffic_controller_process,
        args=(emergency_event, throttle_mode, status_queue, camera_indices),
        name="TrafficController"
    )
    processes.append(p1)
    
    # Process 2: Accident Detection
    if ENABLE_ACCIDENT_DETECTION:
        p2 = Process(
            target=accident_detection_process,
            args=(emergency_event, status_queue, camera_indices, ACCIDENT_CHECK_INTERVAL),
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
    
    # Start all processes
    print(f"[MAIN] Starting {len(processes)} processes...")
    for p in processes:
        p.start()
        print(f"[MAIN]   ✓ {p.name} (PID: {p.pid})")
    
    print(f"\n[MAIN] All processes running")
    print(f"[MAIN] Press Ctrl+C to stop\n")
    
    try:
        # Monitor processes
        while True:
            # Check if emergency triggered
            if emergency_event.is_set():
                print("\n[MAIN] 🚨 EMERGENCY EVENT DETECTED!")
                print("[MAIN] Shutting down all processes...")
                break
            
            # Check process health
            all_alive = all(p.is_alive() for p in processes)
            if not all_alive:
                print("\n[MAIN] ⚠️  Process died, shutting down...")
                for p in processes:
                    if not p.is_alive():
                        print(f"[MAIN]   ✗ {p.name} (PID: {p.pid}) terminated")
                break
            
            # Process status updates (non-blocking)
            try:
                while not status_queue.empty():
                    status = status_queue.get_nowait()
                    # You can log or process status updates here
                    # print(f"[MAIN] Status: {status}")
            except:
                pass
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C detected, shutting down...")
    
    finally:
        # Graceful shutdown
        print("\n[MAIN] Initiating graceful shutdown...")
        
        # Signal all processes to stop
        emergency_event.set()
        
        # Wait for processes to finish (with timeout)
        print("[MAIN] Waiting for processes to finish...")
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"[MAIN]   Force terminating {p.name}...")
                p.terminate()
                p.join(timeout=2)
                if p.is_alive():
                    p.kill()
        
        # Final stats
        temp = SystemMonitor.get_cpu_temp()
        print(f"\n[MAIN] Final temperature: {temp:.1f}°C")
        print("[MAIN] Shutdown complete\n")
 
 
if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()