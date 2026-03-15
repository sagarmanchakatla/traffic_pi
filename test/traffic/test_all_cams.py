import cv2

MAX_CAMERAS = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cameras = {}

print("Scanning for cameras...")

# Discover cameras
for idx in range(MAX_CAMERAS):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cameras[idx] = cap
        print(f"? Camera found at index {idx}")
    else:
        cap.release()

if not cameras:
    print("No cameras detected")
    exit()

print(f"\nTotal cameras detected: {len(cameras)}")
print("Press 'q' to quit\n")

try:
    while True:

        for idx, cap in cameras.items():

            ret, frame = cap.read()

            if not ret:
                print(f"Failed to read camera {idx}")
                continue

            cv2.putText(
                frame,
                f"Camera {idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(f"Camera_{idx}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping camera viewer...")

finally:
    for cap in cameras.values():
        cap.release()

    cv2.destroyAllWindows()

print("All cameras closed")
