from ultralytics import YOLO

# Load a pretrained YOLOv8s model
model = YOLO("yolov8s.pt")

# Train on your dataset
model.train(
    data="C:/new yolo/YOLO_dataset/data.yaml",  # path to dataset yaml
    epochs=50,       # number of epochs
    imgsz=640,        # image size
)
