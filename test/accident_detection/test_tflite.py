from ultralytics import YOLO

# Load your model
model = YOLO("weights/epoch61.pt")

# Export to TFLite INT8
model.export(format="tflite", int8=True)
# Creates: epoch61_int8.tflite (~30MB!)

