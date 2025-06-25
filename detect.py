from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train3/weights/best.pt")  # Update path if needed

# Run prediction on an image or folder
results = model.predict(
    source=r"c:\Users\HP\Downloads\100017d.JPG",  # or folder: "C:/new yolo/test_images/"
    imgsz=640,         
           
)
results[0].show()
results[0].save(filename="D-1.jpg")