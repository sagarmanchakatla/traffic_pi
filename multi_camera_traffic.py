import cv2
import time
import subprocess
import psutil
from services.yolo.yolo_detector import YOLODetector
from services.traffic.traffic_optimizer import TrafficTimingOptimizer
from services.traffic.traffic_lights import TrafficLightService
from services.accident_detection.accident_detection import AccidentDetection

# ---------------- CONFIG ----------------
FRAME_WIDTH = 416
FRAME_HEIGHT = 416
# DETECTION_INTERVAL = 10
DETECTION_RATE = 7
TEMP_THRESHOLD = 75
CPU_THRESHOLD = 80
DISPLAY = False
MAX_CAMERA_INDEX = 6
ADAPTIVE_MODE = True
ENABLE_PHYSICAL_LIGHTS = True  # ← Set False to disable LEDs
# ---------------------------------------

class SystemMonitor:
    """Monitor Pi temperature and performance"""
    
    @staticmethod
    def get_cpu_temp():
        try:
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True, timeout=1)
            temp_str = result.stdout.strip()
            temp = float(temp_str.split('=')[1].split("'")[0])
            return temp
        except:
            return 0.0
    
    @staticmethod
    def get_cpu_usage():
        return psutil.cpu_percent(interval=0.5)
    
    @staticmethod
    def should_throttle(temp, cpu):
        return temp > TEMP_THRESHOLD or cpu > CPU_THRESHOLD
    
    @staticmethod
    def log_stats(temp, cpu, memory):
        status = "⚠️  THROTTLING" if SystemMonitor.should_throttle(temp, cpu) else "✓ OK"
        print(f"[SYS] {status} | Temp: {temp:.1f}°C | CPU: {cpu:.1f}% | RAM: {memory:.1f}%")

class TrafficSignalController:
    def __init__(self, detector,accident_detector, optimizer, cameras, lane_config, enable_lights=True):
        self.detector = detector
        self.accident_detector = accident_detector
        self.optimizer = optimizer
        self.cameras = cameras
        self.lane_config = lane_config
        
        # Initialize physical lights
        self.lights = None
        if enable_lights:
            try:
                print("\n[INFO] Initializing physical traffic lights...")
                self.lights = TrafficLightService(list(cameras.keys()))
                
                if self.lights.enabled:
                    print("[INFO] Running light test sequence...")
                    self.lights.test_sequence()
                    print("[INFO] ✓ Physical lights ready\n")
                else:
                    print("[INFO] Running without physical lights\n")
                    self.lights = None
            except Exception as e:
                print(f"[ERROR] Light initialization failed: {e}")
                print("[INFO] Continuing without physical lights\n")
                self.lights = None
        
        # Current cycle state
        self.current_timings = None
        self.current_priority = []
        self.cycle_start_time = None
        self.current_cycle_duration = 0
        
        # Signal state
        self.current_phase_index = 0
        self.phase_start_time = None
        self.is_running = False
        
        # Performance tracking
        self.detection_count = 0
        self.last_stats_time = time.time()
        self.throttled_mode = False
        
        # Track current light state
        self.current_green_lane = None
        self.yellow_started = False
        
    def capture_snapshots(self):
        snapshots = {}
        cameras_to_process = self.cameras.items()
        
        if self.throttled_mode:
            print("[THROTTLE] Processing reduced camera set")
            cameras_to_process = list(self.cameras.items())[:2]
        
        for lane, cap in cameras_to_process:
            ret, frame = cap.read()
            if ret:
                if self.throttled_mode:
                    frame = cv2.resize(frame, (320, 320))
                snapshots[lane] = frame
            else:
                print(f"[WARN] Failed to capture from {lane}")
        
        return snapshots
    
    def detect_and_count(self, snapshots):
        lane_counts = {}
        annotated_frames = {}
        
        for lane, frame in snapshots.items():
            _, img_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            detection_start = time.time()
            result = self.detector.detect(img_bytes.tobytes())
            detection_time = time.time() - detection_start
            
            detections = result["detections"]
            
            counts = {}
            for det in detections:
                name = det["class_name"]
                counts[name] = counts.get(name, 0) + 1
            
            lane_counts[lane] = counts
            
            if DISPLAY and not self.throttled_mode:
                annotated = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, label, (x1, max(20, y1 - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                annotated_frames[lane] = annotated
            
            print(f"  [{lane}] Detection: {detection_time:.2f}s - Vehicles: {sum(counts.values())}")
        
        return lane_counts, annotated_frames
    
    def calculate_new_cycle(self):
        print("\n" + "="*70)
        print("CALCULATING NEW CYCLE")
        print("="*70)
        
        # Set all lights to red during calculation
        if self.lights:
            self.lights.set_all_red()
        
        temp = SystemMonitor.get_cpu_temp()
        cpu = SystemMonitor.get_cpu_usage()
        memory = psutil.virtual_memory().percent
        
        SystemMonitor.log_stats(temp, cpu, memory)
        
        if ADAPTIVE_MODE:
            self.throttled_mode = SystemMonitor.should_throttle(temp, cpu)
            if self.throttled_mode:
                print("[MODE] Adaptive throttling ENABLED")
                time.sleep(2)
        
        snapshot_start = time.time()
        snapshots = self.capture_snapshots()
        print(f"[TIME] Snapshot capture: {time.time() - snapshot_start:.2f}s")
        
        detect_start = time.time()
        lane_counts, annotated_frames = self.detect_and_count(snapshots)
        print(f"[TIME] Total detection: {time.time() - detect_start:.2f}s")
        
        print("\n=== VEHICLE COUNTS ===")
        total_vehicles = 0
        for lane, counts in lane_counts.items():
            lane_total = sum(counts.values())
            total_vehicles += lane_total
            print(f"{lane}: {counts} (Total: {lane_total})")
        print(f"SYSTEM TOTAL: {total_vehicles} vehicles")
        
        optimize_start = time.time()
        optimization = self.optimizer.optimize(lane_counts, self.lane_config)
        print(f"[TIME] Optimization: {time.time() - optimize_start:.2f}s")
        
        self.current_timings = optimization["timings"]
        self.current_priority = optimization["priority"]
        self.current_cycle_duration = optimization["totalTime"]
        
        print("\n=== NEW SIGNAL TIMINGS ===")
        for idx, lane in enumerate(self.current_priority, 1):
            timing = self.current_timings[lane]
            print(f"{idx}. {lane}: Green={timing['green']}s, Yellow={timing['yellow']}s")
            if timing['leftGreen'] > 0:
                print(f"   └─ Left: Green={timing['leftGreen']}s, Yellow={timing['leftYellow']}s")
        
        print(f"\nTotal Cycle Duration: {self.current_cycle_duration}s")
        print("="*70 + "\n")
        
        if DISPLAY and annotated_frames:
            for lane, frame in annotated_frames.items():
                cv2.imshow(f"{lane}_detection", frame)
            cv2.waitKey(1000)
        
        self.cycle_start_time = time.time()
        self.current_phase_index = 0
        self.phase_start_time = time.time()
        self.detection_count += 1
        self.current_green_lane = None
        self.yellow_started = False
        
        return optimization
    
    def get_phase_duration(self, lane_index):
        if lane_index >= len(self.current_priority):
            return 0
        
        lane = self.current_priority[lane_index]
        timing = self.current_timings[lane]
        
        total = 0
        if timing['leftGreen'] > 0:
            total += timing['leftGreen'] + timing['leftYellow']
        total += timing['green'] + timing['yellow']
        total += 2
        
        return total
    
    def run_cycle(self):
        self.is_running = True
        
        print("[INFO] Starting traffic signal controller...")
        print(f"[INFO] Physical lights: {'ENABLED' if self.lights else 'DISABLED'}")
        print(f"[INFO] Adaptive mode: {ADAPTIVE_MODE}")
        print(f"[INFO] Temperature threshold: {TEMP_THRESHOLD}°C\n")
        
        # Calculate initial cycle
        self.calculate_new_cycle()
        
        # All-red clearance before starting first phase
        # if self.lights:
        #     self.lights.set_all_red()
        #     time.sleep(2)
        
        # Start first phase
        if self.current_priority:
            first_lane = self.current_priority[0]
            timing = self.current_timings[first_lane]
            
            print(f"\n[PHASE 1/{len(self.current_priority)}] {first_lane.upper()}")
            print(f"  GREEN for {timing['green']}s")
            
            if self.lights:
                self.lights.set_color(first_lane, 'green')
            
            self.current_green_lane = first_lane
            self.yellow_started = False
            self.phase_start_time = time.time()
        
        frame_count = 0
        last_yellow_check = 0
        last_red_check = 0
        
        while self.is_running:
            current_time = time.time()
            
            if frame_count % DETECTION_RATE == 0:
                print(f"\n[ACCIDENT CHECK] Frame {frame_count}")

                curr_snapshots = self.capture_snapshots()
                
                snapshot_bytes = {}
                for lane, frame in curr_snapshots.items():
                    _, img_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    snapshot_bytes[lane] = img_bytes.tobytes()
                    
                accident_detections = self.accident_detector.detect_lanes(snapshot_bytes)
                
                for lane, result in accident_detections.items():
                    if result == "ACCIDENT":
                        print(f"  Accident detected on {lane.upper()}")
                        if self.lights:
                            self.lights.set_all_red()
                        self.is_running = False
                        break
                    else:
                        print("[INFO] No accidents detected")
            
            # Periodic system stats
            if current_time - self.last_stats_time >= 30:
                temp = SystemMonitor.get_cpu_temp()
                cpu = SystemMonitor.get_cpu_usage()
                memory = psutil.virtual_memory().percent
                SystemMonitor.log_stats(temp, cpu, memory)
                self.last_stats_time = current_time
            
            # Check if entire cycle is complete
            cycle_elapsed = current_time - self.cycle_start_time
            
            if cycle_elapsed >= self.current_cycle_duration:
                print(f"\n[CYCLE] Complete! (Duration: {cycle_elapsed:.1f}s)")
                
                # All-red clearance
                # if self.lights:
                #     self.lights.set_all_red()
                #     time.sleep(2)
                
                # Calculate new cycle
                self.calculate_new_cycle()
                
                # Start first phase of new cycle
                if self.current_priority:
                    first_lane = self.current_priority[0]
                    timing = self.current_timings[first_lane]
                    
                    print(f"\n[PHASE 1/{len(self.current_priority)}] {first_lane.upper()}")
                    print(f"  GREEN for {timing['green']}s")
                    
                    if self.lights:
                        self.lights.set_color(first_lane, 'green')
                    
                    self.current_green_lane = first_lane
                    self.yellow_started = False
                    self.phase_start_time = time.time()
                
                continue
            
            # Current phase management
            if self.current_phase_index < len(self.current_priority):
                lane = self.current_priority[self.current_phase_index]
                timing = self.current_timings[lane]
                phase_elapsed = current_time - self.phase_start_time
                
                # Transition: Green → Yellow
                if phase_elapsed >= timing['green'] and not self.yellow_started:
                    # Prevent multiple triggers
                    if current_time - last_yellow_check > 0.5:
                        print(f"[TRANSITION] {lane.upper()}: GREEN → YELLOW (after {timing['green']}s)")
                        print(f"  YELLOW for {timing['yellow']}s")
                        
                        if self.lights:
                            self.lights.set_color(lane, 'yellow')
                        
                        self.yellow_started = True
                        last_yellow_check = current_time
                
                # Transition: Yellow → Red (and move to next phase)
                yellow_end = timing['green'] + timing['yellow']
                all_red_end = yellow_end + 2  # Add all-red clearance
                
                if phase_elapsed >= all_red_end:
                    # Prevent multiple triggers
                    if current_time - last_red_check > 0.5:
                        # End of this phase - set to red
                        if self.lights:
                            if self.lights.current_state.get(lane) != 'red':
                                print(f"[TRANSITION] {lane.upper()}: YELLOW → RED")
                                self.lights.set_color(lane, 'red')
                        
                        # All-red clearance
                        # print(f"[CLEARANCE] All-red for 2s")
                        # time.sleep(2)
                        
                        # Move to next phase
                        self.current_phase_index += 1
                        
                        if self.current_phase_index < len(self.current_priority):
                            # Start next lane
                            next_lane = self.current_priority[self.current_phase_index]
                            next_timing = self.current_timings[next_lane]
                            
                            print(f"\n[PHASE {self.current_phase_index + 1}/{len(self.current_priority)}] {next_lane.upper()}")
                            print(f"  GREEN for {next_timing['green']}s")
                            
                            if self.lights:
                                self.lights.set_color(next_lane, 'green')
                            
                            self.current_green_lane = next_lane
                            self.yellow_started = False
                            self.phase_start_time = time.time()
                        else:
                            # All phases complete - wait for cycle to finish
                            print(f"[INFO] All phases complete, waiting for cycle end...")
                        
                        last_red_check = current_time
            
            if DISPLAY and frame_count % 3 == 0:
                for lane, cap in self.cameras.items():
                    ret, frame = cap.read()
                    if ret:
                        display_frame = cv2.resize(frame, (320, 240))
                        
                        if self.lights and lane in self.lights.current_state:
                            status = self.lights.current_state[lane].upper()
                            color = {
                                'GREEN': (0, 255, 0),
                                'YELLOW': (0, 255, 255),
                                'RED': (0, 0, 255)
                            }.get(status, (128, 128, 128))
                        else:
                            status = "N/A"
                            color = (128, 128, 128)
                        
                        cv2.putText(display_frame, f"{lane.upper()}", (10, 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(display_frame, status, (10, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        cv2.imshow(f"{lane}_live", display_frame)
            
            frame_count += 1
                        
            if DISPLAY and cv2.waitKey(100) & 0xFF == ord('q'):
                print("\n[INFO] Exit requested")
                self.is_running = False
                break
            
            sleep_time = 0.2 if self.throttled_mode else 0.1
            time.sleep(sleep_time)
    
    def stop(self):
        self.is_running = False
        if self.lights:
            self.lights.cleanup()
        print("[INFO] Controller stopped")

def discover_cameras(max_index=6):
    cameras = {}
    lane_id = 1
    
    print("[INFO] Scanning for cameras...")
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            lane_name = f"lane{lane_id}"
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            cameras[lane_name] = cap
            print(f"  ✓ Camera {idx} → {lane_name}")
            lane_id += 1
        else:
            cap.release()
    
    print(f"[INFO] Found {len(cameras)} camera(s)\n")
    return cameras

def main():
    print("\n" + "="*70)
    print("RASPBERRY PI TRAFFIC SIGNAL WITH PHYSICAL LIGHTS")
    print("="*70 + "\n")
    
    temp = SystemMonitor.get_cpu_temp()
    cpu = SystemMonitor.get_cpu_usage()
    memory = psutil.virtual_memory().percent
    
    print(f"[SYS] Initial status:")
    SystemMonitor.log_stats(temp, cpu, memory)
    
    if temp > TEMP_THRESHOLD:
        print(f"\n⚠️  WARNING: High temperature ({temp}°C)")
        time.sleep(3)
    
    print("[INFO] Loading YOLO model...")
    model_load_start = time.time()
    detector = YOLODetector("yolov8n.pt")
    print(f"[INFO] Model loaded in {time.time() - model_load_start:.2f}s\n")
    
    print("\n[INFO] Loading Accident Detection Model...")
    accident_detector = AccidentDetection()
    print("[INFO] Accident Detection Model Loaded\n")
    
    optimizer = TrafficTimingOptimizer()
    cameras = discover_cameras(MAX_CAMERA_INDEX)
    
    if not cameras:
        print("[ERROR] No cameras detected!")
        return
    
    lane_config = {lane: {"hasLeft": False} for lane in cameras.keys()}
    
    controller = TrafficSignalController(
        detector,accident_detector, optimizer, cameras, lane_config,
        enable_lights=ENABLE_PHYSICAL_LIGHTS
    )
    
    try:
        controller.run_cycle()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.stop()
        
        for cap in cameras.values():
            cap.release()
        cv2.destroyAllWindows()
        
        temp = SystemMonitor.get_cpu_temp()
        print(f"\n[SYS] Final temperature: {temp:.1f}°C")
        print("[INFO] Goodbye!\n")

if __name__ == "__main__":
    main()
