from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load model
model = YOLO('runs/detect/train3/weights/best.pt')

# 2. Load image
image_path ="images op/100010d.JPG"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
img_height, img_width = img.shape[:2]

# 3. Predict
results = model.predict(source=image_path, imgsz=640, conf=0.25)

# 4. Create blank heatmap
heatmap = np.zeros((img_height, img_width), dtype=np.float32)

# 5. Add boxes to heatmap
for result in results:
    if result.boxes is None:
        continue
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = max(0, min(x1, img_width - 1))
        x2 = max(0, min(x2, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        y2 = max(0, min(y2, img_height - 1))

        # Create mask for Gaussian blob
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0:
            blob = np.zeros((h, w), dtype=np.float32)
            cv2.circle(blob, (w // 2, h // 2), min(w, h) // 3, float(score), -1)
            blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=w/6, sigmaY=h/6)

            heatmap[y1:y2, x1:x2] += blob

# 6. Normalize and colorize
heatmap = np.clip(heatmap, 0, None)  # remove negatives
heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap_uint8 = heatmap_norm.astype(np.uint8)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# 7. Overlay
overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# 8. Plot
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title('Improved Detection Heatmap Overlay')
plt.axis('off')
plt.tight_layout()
plt.show()