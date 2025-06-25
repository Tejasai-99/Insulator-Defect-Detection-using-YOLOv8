import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load model
model = YOLO("runs/detect/train3/weights/best.pt")

# Load original image
img_path = "images op/170713d.JPG"
original_img = cv2.imread(img_path)
h_orig, w_orig = original_img.shape[:2]

# Resize for model input
img_resized = cv2.resize(original_img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

# Register forward hook to extract feature map
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.model.model[-2].register_forward_hook(get_activation("featmap"))

# Run model prediction
results = model(img_tensor)

# Generate feature-based heatmap
featmap = activation["featmap"].squeeze(0).mean(0).cpu().numpy()
heatmap = (featmap - featmap.min()) / (featmap.max() - featmap.min())
heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig))
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

# Overlay heatmap on original image
overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

# Draw detection boxes + labels on overlay
names = model.model.names  # class name map
scale_x, scale_y = w_orig / 640, h_orig / 640

for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
    x1, y1, x2, y2 = map(float, box[:4])
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    label = names[int(cls)]
    score = float(conf)

    # Draw box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Label text
    text = f"{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
    cv2.putText(overlay, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Save and display final output
cv2.imwrite("final_combined_detection_heatmap.jpg", overlay)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("YOLO Detection + Heatmap Overlay with Labels")
plt.axis('off')
plt.show()
