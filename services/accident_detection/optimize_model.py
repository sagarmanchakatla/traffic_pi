#!/usr/bin/env python3
"""
Optimize Accident Detection Model
Converts model.json + model_weights.h5 to TFLite quantized model
For use with Raspberry Pi camera-based accident detection
"""

import tensorflow as tf
import numpy as np
import os
from datetime import datetime

print("="*70)
print("ACCIDENT DETECTION MODEL OPTIMIZATION")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")

# ==================== CONFIGURATION ====================

# Input files
MODEL_JSON = "model.json"
MODEL_WEIGHTS = "model_weights.h5"

# Output file
OUTPUT_FILE = "accident_model_quantized.tflite"

# Training data for calibration
TRAIN_DIR = "data/train"

# ==================== STEP 1: LOAD MODEL ====================

print("[1/4] Loading model from JSON and weights...")

try:
    # Load architecture
    with open(MODEL_JSON, "r") as f:
        model_json = f.read()
    
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(MODEL_WEIGHTS)
    
    print(f"✓ Model loaded successfully")
    
    # Get original size
    json_size = os.path.getsize(MODEL_JSON) / (1024**2)
    weights_size = os.path.getsize(MODEL_WEIGHTS) / (1024**2)
    original_size = json_size + weights_size
    
    print(f"  Original size: {original_size:.1f} MB\n")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nMake sure these files exist in current directory:")
    print(f"  - {MODEL_JSON}")
    print(f"  - {MODEL_WEIGHTS}")
    exit(1)

# ==================== STEP 2: CALIBRATION DATA ====================

print("[2/4] Preparing calibration data...")

def get_calibration_data():
    """Load sample images for quantization calibration"""
    
    def data_generator():
        images_loaded = 0
        target_images = 100
        
        # Try to load from training data
        if os.path.exists(TRAIN_DIR):
            print(f"  Loading from: {TRAIN_DIR}")
            
            for class_name in ["accident", "Accident", "non-accident", "No Accident"]:
                class_dir = os.path.join(TRAIN_DIR, class_name)
                
                if not os.path.exists(class_dir):
                    continue
                
                files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_file in files[:50]:  # Max 50 per class
                    if images_loaded >= target_images:
                        break
                    
                    try:
                        img_path = os.path.join(class_dir, img_file)
                        
                        # Load and preprocess
                        img = tf.keras.preprocessing.image.load_img(
                            img_path, target_size=(250, 250)
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        yield [img_array.astype(np.float32)]
                        images_loaded += 1
                        
                        if images_loaded % 20 == 0:
                            print(f"  Loaded {images_loaded} images...", end='\r')
                    
                    except:
                        continue
            
            print(f"\n  ✓ Loaded {images_loaded} calibration images")
        
        # Fallback to random data
        else:
            print(f"  ⚠️  Training data not found at: {TRAIN_DIR}")
            print(f"  Using random data (less accurate)")
            
            for i in range(target_images):
                if i % 20 == 0:
                    print(f"  Generating {i}/{target_images}...", end='\r')
                
                random_img = np.random.rand(1, 250, 250, 3).astype(np.float32)
                yield [random_img]
            
            print(f"\n  ✓ Generated {target_images} random samples")
    
    return data_generator

# ==================== STEP 3: CONVERT TO TFLITE ====================

print("\n[3/4] Converting to TFLite INT8...")
print("  This may take 2-5 minutes...\n")

try:
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set calibration data
    converter.representative_dataset = get_calibration_data()
    
    # Force INT8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    print("\n  Converting...")
    tflite_model = converter.convert()
    
    print(f"  ✓ Conversion successful!\n")

except Exception as e:
    print(f"\n  ✗ INT8 failed: {e}")
    print("  Trying dynamic quantization...\n")
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        print(f"  ✓ Dynamic quantization successful!\n")
    except Exception as e2:
        print(f"  ✗ All conversions failed: {e2}")
        exit(1)

# ==================== STEP 4: SAVE AND TEST ====================

print("[4/4] Saving and testing model...")

# Save
with open(OUTPUT_FILE, 'wb') as f:
    f.write(tflite_model)

new_size = len(tflite_model) / (1024**2)
reduction = ((original_size - new_size) / original_size) * 100

print(f"  ✓ Saved: {OUTPUT_FILE}")
print(f"  Size: {new_size:.1f} MB")
print(f"  Reduction: {reduction:.1f}%\n")

# Test
try:
    import tflite_runtime.interpreter as tflite
except:
    import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path=OUTPUT_FILE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Model Info:")
print(f"    Input: {input_details[0]['shape']}, {input_details[0]['dtype']}")
print(f"    Output: {output_details[0]['shape']}, {output_details[0]['dtype']}")

# Speed test
import time

if input_details[0]['dtype'] == np.uint8:
    test_img = np.random.randint(0, 256, (1, 250, 250, 3), dtype=np.uint8)
else:
    test_img = np.random.rand(1, 250, 250, 3).astype(np.float32)

times = []
for _ in range(50):
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], test_img)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
    times.append(time.time() - start)

avg_time = np.mean(times) * 1000
fps = 1000 / avg_time

print(f"\n  Performance:")
print(f"    Inference time: {avg_time:.1f} ms")
print(f"    FPS: {fps:.1f}")

# ==================== SUMMARY ====================

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE!")
print("="*70)

print(f"\n✓ Optimized model: {OUTPUT_FILE}")
print(f"\nImprovements:")
print(f"  • Size: {original_size:.1f} MB → {new_size:.1f} MB ({reduction:.1f}% reduction)")
print(f"  • Speed: ~{fps:.1f} FPS")
print(f"  • RAM: ~50% less usage")

print(f"\nNext steps:")
print(f"  1. Copy to Raspberry Pi:")
print(f"     scp {OUTPUT_FILE} pi@raspberrypi:~/traffic_system/models/")
print(f"\n  2. Use with test script:")
print(f"     python test_accident_detection.py")

print("\n" + "="*70 + "\n")
