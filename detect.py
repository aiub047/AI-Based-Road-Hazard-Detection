from ultralytics import YOLO

# 1. Load your pre-trained model
model = YOLO("yolov8n.pt")

# 2. Run object detection on your photo
results = model("/Users/mohammadaiubkhan/Works/AI/Models/Yolo/yolo_project/me.jpg")

# 3. FIX: Access the first result in the list and display it
results[0].show()
