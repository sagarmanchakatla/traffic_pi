from ultralytics import YOLO
from huggingface_hub import hf_hub_download, list_repo_files
import os


REPO_ID = "Enos-123/traffic-accident-detection-yolo11x"
WEIGHTS_PATH = "weights/epoch61.pt"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# LIST FILES IN REPO
# =============================
print("Listing repository files...")
files = list_repo_files(REPO_ID)
print(files)

# =============================
# DOWNLOAD MODEL WEIGHTS
# =============================
print("\nDownloading model weights...")
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=WEIGHTS_PATH
)
print("Model downloaded at:", model_path)

# =============================
# LOAD YOLO MODEL
# =============================
print("\nLoading YOLO model...")
model = YOLO(model_path)
print("Model loaded successfully")

# =============================
# TEST WITH SAMPLE IMAGES
# =============================
print("\nRunning inference on sample images...")
test_images = []

for i in range(1, 5):
    img_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"testing/fig{i}.jpg"
    )
    test_images.append(img_path)

results = model(test_images)

total_detections = 0
confidences = []

for r in results:
    if r.boxes is not None:
        total_detections += len(r.boxes)
        confidences.extend([float(c) for c in r.boxes.conf])

print("Total Detections:", total_detections)
if confidences:
    print("Average Confidence:", sum(confidences) / len(confidences))
else:
    print("Average Confidence: N/A")

# =============================
# SINGLE IMAGE INFERENCE
# =============================
def run_image_inference(image_path):
    print(f"\nRunning inference on image: {image_path}")
    results = model(image_path)
    results[0].save(save_dir=OUTPUT_DIR)
    print("Saved annotated image to:", OUTPUT_DIR)

# Example usage:
# run_image_inference("your_image.jpg")

# =============================
# VIDEO INFERENCE
# =============================
def run_video_inference(video_path):
    print(f"\nRunning inference on video: {video_path}")
    results = model(video_path)
    results.save(save_dir=OUTPUT_DIR)
    print("Saved annotated video to:", OUTPUT_DIR)

# Example usage:
run_video_inference("test_video.mp4")

print("\nScript execution completed.")