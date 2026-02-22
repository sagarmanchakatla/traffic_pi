import cv2
import glob

devices = sorted(glob.glob("/dev/video*"))

if not devices:
    print("❌ No video devices found")
    exit(1)

print(f"🔍 Found {len(devices)} video devices\n")

for dev in devices:
    index = int(dev.replace("/dev/video", ""))
    print(f"Testing {dev} ... ", end="")

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("❌ Cannot open")
        continue

    ret, frame = cap.read()
    if ret:
        print(f"✅ OK ({frame.shape[1]}x{frame.shape[0]})")
    else:
        print("❌ Opened but no frame")

    cap.release()
