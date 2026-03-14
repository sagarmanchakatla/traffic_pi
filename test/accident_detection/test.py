from ultralytics import YOLO
from huggingface_hub import hf_hub_download, list_repo_files
import os

REPO_ID = "Enos-123/traffic-accident-detection-yolo11x"
WEIGHTS_FILENAME = "epoch61.pt"
REMOTE_WEIGHTS_PATH = f"weights/{WEIGHTS_FILENAME}"

LOCAL_WEIGHTS_DIR = "weights"
LOCAL_WEIGHTS_PATH = os.path.join(LOCAL_WEIGHTS_DIR, WEIGHTS_FILENAME)

OUTPUT_DIR = "outputs"

os.makedirs(LOCAL_WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# LIST FILES IN REPO
# =============================
print("Listing repository files...")
files = list_repo_files(REPO_ID)
print(files)

# =============================
# DOWNLOAD MODEL WEIGHTS LOCALLY
# =============================
if not os.path.exists(LOCAL_WEIGHTS_PATH):
    print("\nDownloading model weights to local folder...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=REMOTE_WEIGHTS_PATH,
        local_dir=".",                 # Download to current project
        local_dir_use_symlinks=False   # Ensure real file copy
    )
    print("Model downloaded at:", LOCAL_WEIGHTS_PATH)
else:
    print("\nModel already exists locally:", LOCAL_WEIGHTS_PATH)

# =============================
# LOAD YOLO MODEL
# =============================
print("\nLoading YOLO model...")
model = YOLO(LOCAL_WEIGHTS_PATH)
print("Model loaded successfully")

# =============================
# TEST WITH SAMPLE IMAGES
# =============================
print("\nRunning inference on sample images...")
test_images = []

for i in range(1, 5):
    img_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"testing/fig{i}.jpg",
        local_dir=".",
        local_dir_use_symlinks=False
    )
    test_images.append(img_path)

results = model(test_images, verbose=False)

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
    results = model(image_path, verbose=False)
    results[0].save(save_dir=OUTPUT_DIR)
    print("Saved annotated image to:", OUTPUT_DIR)

# =============================
# VIDEO INFERENCE
# =============================
def run_video_inference(video_path):
    print(f"\nRunning inference on video: {video_path}")
    results = model(video_path, verbose=False)
    results.save(save_dir=OUTPUT_DIR)
    print("Saved annotated video to:", OUTPUT_DIR)

print("\nScript execution completed.")