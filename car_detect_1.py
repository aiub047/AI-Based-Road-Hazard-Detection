import cv2
import torch
from ultralytics import YOLO

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Running YOLO on device: {device.upper()}")

# Load the model
model = YOLO("yolo26s.pt").to(device)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Webcam started! Filtering for CARS ONLY. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FILTER: classes=[2] isolates cars. Lower conf=0.15 helps capture toy cars.
    results = model(frame, stream=True, conf=0.15, classes=[2])

    for result in results:
        annotated_frame = result.plot()

    cv2.imshow("YOLO - Cars Only", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()