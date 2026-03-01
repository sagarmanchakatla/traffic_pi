import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download, list_repo_files
import os

model_path = "epoch61.pt"

print("\nLoading YOLO model...")
model = YOLO(model_path)
print("Model loaded successfully")


def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = model(frame, stream=True)
            # print(results)
            for r in results:
                print(r)
                annotated_frame = r.plot()
                
                # 5. Display the frame using OpenCV
                cv2.imshow("YOLO11 Accident Detection", annotated_frame)
            
            # 6. Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    video_path = "test_video.mp4"
    run_video(video_path)