import cv2
import torch
from ultralytics import YOLO

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Running YOLO on device: {device.upper()}")

# 1. UPGRADE: Loading the latest YOLO26 Small model
model = YOLO("yolo26s.pt").to(device)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Webcam started! Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2. OPTIMIZE: Lower the confidence ceiling so hard-to-detect objects appear
    results = model(frame, stream=True, conf=0.15)

    for result in results:
        annotated_frame = result.plot()

    cv2.imshow("YOLO Live Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()