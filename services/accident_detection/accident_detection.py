from ultralytics import YOLO
import cv2
import numpy as np
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEIGHT_PATH = "weights/epoch61.pt"
TEST_IMG = "fig2.jpg"

class AccidentDetection:
    def __init__(self):
        try:
            self.model = YOLO(WEIGHT_PATH)
        except Exception as e:
            logger.error(f"Failed to load Model: {e}")
            
    def detect(self, image_bytes: bytes) -> str:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("Failed to decode image")
                return "NOT"

            results = self.model(img, verbose=False)

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return "NOT"

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if cls_id == 0 and confidence > 0.6:
                    logger.info(f"Accident detected with confidence: {confidence:.2f}")
                    return "ACCIDENT"

            return "NOT"

        except Exception as e:
            logger.error(f"Error occurred during accident detection: {e}")
            return "NOT" 
        
    def detect_lanes(self, snapshots: Dict[str,bytes]):
        results = {}
        try:
            for lane, frame in snapshots.items():
                results[lane] = self.detect(frame)
            return results
        except Exception as e:
            logger.error(f"Error occurred during accident detection for all Lanes: {e}")
            return 
                
        
if __name__ == "__main__":
    detector = AccidentDetection()
    with open(TEST_IMG, "rb") as f:
        image_bytes = f.read()
    
    results = detector.detect(image_bytes)
    print("Resutl is", results)
    # for r in results:
    #     annotated_frame = r.plot()

    #     cv2.imshow("YOLO11 Accident Detection", annotated_frame)

    #     # Keep window open until 'q' is pressed
    #     while True:
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    cv2.destroyAllWindows()