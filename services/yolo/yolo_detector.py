import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load the YOLOv8 model
        # It will download 'yolov8n.pt' automatically if not present
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        logger.info(f"Loaded YOLO model: {model_path}")
        
        # Define vehicle classes we are interested in
        # COCO dataset class indices: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.vehicle_classes = [2, 3, 5, 7] 

    def detect(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Detect vehicles in an image byte stream.
        Returns detections and annotated image as base64.
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image")
                return {"detections": [], "annotated_image": None}

            # Run inference
            results = self.model(
                img,
                conf=0.25, # Confidence threshold
                classes=self.vehicle_classes,
                verbose=False
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]
                        
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "class_id": class_id,
                            "class_name": class_name
                        })
            
            # Draw boxes
            annotated_img = img.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                label = f"{det['class_name']} {det['confidence']:.2f}"
                color = (0, 255, 0) # Green
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 5)
                cv2.putText(annotated_img, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "detections": detections,
                "annotated_image": img_base64
            }
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"detections": [], "annotated_image": None}

    def get_vehicle_counts(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Helper to get simple counts by class from an image, plus the annotated image.
        """
        result = self.detect(image_bytes)
        detections = result["detections"]
        counts = {}
        for det in detections:
            name = det["class_name"]
            counts[name] = counts.get(name, 0) + 1
            
        return {
            "counts": counts,
            "annotated_image": result["annotated_image"]
        }
