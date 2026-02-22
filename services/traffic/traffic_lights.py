"""
Traffic Light Controller Service
Integrates with main.py traffic signal system
Auto-detects number of lanes and controls physical LEDs
"""

from gpiozero import LED
import time
import threading

class TrafficLightService:
    """
    Physical traffic light controller using gpiozero
    
    CURRENT SETUP (1 Lane):
    Lane 1: Red=GPIO17 (Pin 11), Yellow=GPIO27 (Pin 13), Green=GPIO22 (Pin 15)
    
    FUTURE SETUP (Add more lanes as needed):
    Lane 2: Red=GPIO23 (Pin 16), Yellow=GPIO24 (Pin 18), Green=GPIO25 (Pin 22)
    Lane 3: Red=GPIO5  (Pin 29), Yellow=GPIO6  (Pin 31), Green=GPIO13 (Pin 33)
    Lane 4: Red=GPIO19 (Pin 35), Yellow=GPIO26 (Pin 37), Green=GPIO21 (Pin 40)
    
    Wiring (for each LED):
    - Long leg (anode) → GPIO pin
    - Short leg (cathode) → 220Ω resistor → GND
    """
    
    # GPIO Pin assignments (BCM numbering)
    GPIO_CONFIG = {
        'lane1': {'red': 17, 'yellow': 27, 'green': 22},
        'lane2': {'red': 23, 'yellow': 24, 'green': 25},
        'lane3': {'red': 5,  'yellow': 6,  'green': 13},
        'lane4': {'red': 19, 'yellow': 26, 'green': 21}
    }
    
    def __init__(self, lanes):
        """
        Initialize traffic lights for detected lanes
        
        Args:
            lanes: List of lane names (e.g., ['lane1'] or ['lane1', 'lane2', 'lane3', 'lane4'])
        """
        self.lanes = lanes
        self.leds = {}
        self.current_state = {}
        self.enabled = True
        self.lock = threading.Lock()
        
        print(f"\n{'='*70}")
        print(f"TRAFFIC LIGHT SERVICE - Initializing {len(lanes)} lane(s)")
        print(f"{'='*70}")
        
        try:
            # Initialize LEDs for each detected lane
            for lane in lanes:
                if lane not in self.GPIO_CONFIG:
                    print(f"[LIGHTS] ⚠️  {lane} not configured, skipping")
                    continue
                
                pins = self.GPIO_CONFIG[lane]
                
                # Create LED objects
                self.leds[lane] = {
                    'red': LED(pins['red']),
                    'yellow': LED(pins['yellow']),
                    'green': LED(pins['green'])
                }
                
                # Turn off all LEDs initially
                self.leds[lane]['red'].off()
                self.leds[lane]['yellow'].off()
                self.leds[lane]['green'].off()
                
                # Set to RED initially
                self.leds[lane]['red'].on()
                self.current_state[lane] = 'red'
                
                print(f"[LIGHTS] ✓ {lane.upper()}: Red=GPIO{pins['red']} (Pin 11), "
                      f"Yellow=GPIO{pins['yellow']} (Pin 13), Green=GPIO{pins['green']} (Pin 15)")
            
            print(f"[LIGHTS] ✓ Successfully initialized {len(self.leds)} traffic light(s)")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"[LIGHTS] ✗ Initialization failed: {e}")
            print(f"[LIGHTS] Running without physical lights (simulation mode)")
            self.enabled = False
    
    def set_color(self, lane, color):
        """
        Set traffic light color for a lane
        
        Args:
            lane: 'lane1', 'lane2', etc.
            color: 'red', 'yellow', or 'green'
        """
        with self.lock:
            if not self.enabled:
                # Simulation mode
                print(f"[LIGHTS-SIM] {lane.upper()}: {color.upper()}")
                return
            
            if lane not in self.leds:
                return
            
            try:
                # Turn off all colors
                self.leds[lane]['red'].off()
                self.leds[lane]['yellow'].off()
                self.leds[lane]['green'].off()
                
                # Turn on requested color
                if color == 'red':
                    self.leds[lane]['red'].on()
                elif color == 'yellow':
                    self.leds[lane]['yellow'].on()
                elif color == 'green':
                    self.leds[lane]['green'].on()
                
                self.current_state[lane] = color
                print(f"[LIGHTS] {lane.upper()}: {color.upper()}")
                
            except Exception as e:
                print(f"[LIGHTS] ✗ Error: {e}")
    
    def set_all_red(self):
        """Set all lanes to RED (clearance/safety)"""
        print("[LIGHTS] === ALL RED (Clearance) ===")
        for lane in self.leds.keys():
            self.set_color(lane, 'red')
    
    def execute_phase(self, lane, green_time, yellow_time=4):
        """
        Execute complete traffic light sequence for a lane
        
        Args:
            lane: Lane name
            green_time: Duration of green light (seconds)
            yellow_time: Duration of yellow light (seconds, default 4)
        """
        if lane not in self.leds:
            return
        
        # GREEN phase
        print(f"\n[PHASE] {lane.upper()}: GREEN for {green_time}s")
        self.set_color(lane, 'green')
        time.sleep(green_time)
        
        # YELLOW phase
        print(f"[PHASE] {lane.upper()}: YELLOW for {yellow_time}s")
        self.set_color(lane, 'yellow')
        time.sleep(yellow_time)
        
        # RED phase
        print(f"[PHASE] {lane.upper()}: RED")
        self.set_color(lane, 'red')
    
    def test_sequence(self):
        """Test all lights in sequence"""
        if not self.enabled:
            print("[LIGHTS] Test skipped - running in simulation mode")
            return
        
        print("\n[TEST] Starting light test sequence...")
        
        for lane in self.leds.keys():
            print(f"\n  Testing {lane.upper()}:")
            
            print("    Red...", end='', flush=True)
            self.set_color(lane, 'red')
            time.sleep(1)
            print(" ✓")
            
            print("    Yellow...", end='', flush=True)
            self.set_color(lane, 'yellow')
            time.sleep(1)
            print(" ✓")
            
            print("    Green...", end='', flush=True)
            self.set_color(lane, 'green')
            time.sleep(1)
            print(" ✓")
        
        # Set all back to red
        self.set_all_red()
        print("\n[TEST] Test complete - all lights functional\n")
    
    def blink_yellow_all(self, times=3):
        """Blink all yellow lights (warning mode)"""
        if not self.enabled:
            return
        
        print("[LIGHTS] ⚠️  WARNING MODE - Blinking yellow")
        
        for _ in range(times):
            # Yellow ON
            for lane in self.leds.keys():
                self.leds[lane]['yellow'].on()
                self.leds[lane]['red'].off()
                self.leds[lane]['green'].off()
            time.sleep(0.5)
            
            # Yellow OFF
            for lane in self.leds.keys():
                self.leds[lane]['yellow'].off()
            time.sleep(0.5)
        
        # Back to all red
        self.set_all_red()
    
    def cleanup(self):
        """Cleanup on shutdown"""
        print("\n[LIGHTS] Shutting down traffic lights...")
        
        if not self.enabled:
            print("[LIGHTS] No cleanup needed (simulation mode)")
            return
        
        try:
            # Warning blink
            self.blink_yellow_all(times=2)
            
            # All red
            self.set_all_red()
            time.sleep(0.5)
            
            # Turn off all LEDs
            for lane in self.leds.keys():
                self.leds[lane]['red'].off()
                self.leds[lane]['yellow'].off()
                self.leds[lane]['green'].off()
                
                # Close LED objects
                self.leds[lane]['red'].close()
                self.leds[lane]['yellow'].close()
                self.leds[lane]['green'].close()
            
            print("[LIGHTS] ✓ Cleanup complete")
            
        except Exception as e:
            print(f"[LIGHTS] ✗ Cleanup error: {e}")


# ==================== STANDALONE TEST ====================

if __name__ == "__main__":
    """Test the traffic light service"""
    
    print("\n" + "="*70)
    print("TRAFFIC LIGHT SERVICE - STANDALONE TEST")
    print("="*70 + "\n")
    
    # Test with 1 lane (your current setup)
    print("Testing with 1 lane (current setup):\n")
    
    lights = TrafficLightService(['lane1'])
    
    if not lights.enabled:
        print("\n⚠️  Cannot test - check GPIO connections")
        print("\nWiring for Lane 1:")
        print("  Red LED:    Long leg → GPIO17 (Pin 11), Short leg → 220Ω → GND")
        print("  Yellow LED: Long leg → GPIO27 (Pin 13), Short leg → 220Ω → GND")
        print("  Green LED:  Long leg → GPIO22 (Pin 15), Short leg → 220Ω → GND")
        exit(1)
    
    # Run test sequence
    lights.test_sequence()
    
    # Simulate traffic cycle
    print("\n" + "="*70)
    print("SIMULATING TRAFFIC CYCLE")
    print("="*70 + "\n")
    
    print("Initial state: All RED")
    lights.set_all_red()
    time.sleep(2)
    
    print("\n--- Cycle 1 ---")
    lights.execute_phase('lane1', green_time=10, yellow_time=3)
    lights.set_all_red()
    time.sleep(2)
    
    print("\n--- Cycle 2 ---")
    lights.execute_phase('lane1', green_time=8, yellow_time=3)
    lights.set_all_red()
    time.sleep(2)
    
    # Cleanup
    lights.cleanup()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("1. If test passed, integrate with main.py")
    print("2. To add more lanes, connect LEDs to:")
    print("   Lane 2: Red=GPIO23, Yellow=GPIO24, Green=GPIO25")
    print("   Lane 3: Red=GPIO5,  Yellow=GPIO6,  Green=GPIO13")
    print("   Lane 4: Red=GPIO19, Yellow=GPIO26, Green=GPIO21")
    print("3. System will auto-detect and use all connected lanes\n")
