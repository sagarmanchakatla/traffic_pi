# stop_all_lights.py
# Turns OFF all GPIO pins used by the traffic light controller

from gpiozero import LED
import time

# All GPIO pins used in the system
TRAFFIC_PINS = [
    17, 27, 22,  # Lane 1
    23, 24, 25,  # Lane 2
    5, 6, 13,    # Lane 3
    19, 26, 21   # Lane 4
]

leds = []

print("Stopping all traffic lights...")

# Initialize LEDs
for pin in TRAFFIC_PINS:
    led = LED(pin)
    leds.append(led)

# Turn everything OFF
for led in leds:
    led.off()

# Small delay to ensure state change
time.sleep(0.5)

# Release GPIO pins
for led in leds:
    led.close()

print("✅ All GPIO traffic lights are OFF and released.")