import os
import json
import cv2

# Paths
json_path = "C:/new yolo/labels_v1.2.json"
image_folder = "C:/new yolo/Images"
output_folder = "C:/new yolo/yolo_labels"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Load JSON data
with open(json_path, "r") as f:
    data_list = json.load(f)

# Function to convert bbox to YOLO format
def convert_bbox(img_w, img_h, x, y, w, h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h

# Map condition to class ID: 0 = Good, 1 = Broken, 2 = Flashover
def get_class_id(obj):
    conditions = obj.get("conditions", {})
    all_conditions = " ".join(str(val).lower() for val in conditions.values())

    if "broken" in all_conditions:
        return 1  # Broken
    elif "flashover" in all_conditions:
        return 2  # Flashover
    else:
        return 0  # Good

# Counters for summary
count_good = 0
count_broken = 0
count_flashover = 0
missing_images = []

# Iterate through each image's annotations
for item in data_list:
    image_name = item["filename"]
    image_path = os.path.join(image_folder, image_name)

    # Read image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        print(f" Image not found: {image_name}")
        missing_images.append(image_name)
        continue

    height, width = image.shape[:2]

    # Create .txt filename for YOLO
    base_name, _ = os.path.splitext(image_name)
    txt_filename = base_name + ".txt"
    txt_path = os.path.join(output_folder, txt_filename)

    with open(txt_path, "w") as txt_file:
        for obj in item["Labels"]["objects"]:
            if obj["name"].lower() == "insulator" and "bbox" in obj:
                x, y, bw, bh = obj["bbox"]

                # Get class ID
                class_id = get_class_id(obj)

                # Increment count
                if class_id == 0:
                    count_good += 1
                elif class_id == 1:
                    count_broken += 1
                elif class_id == 2:
                    count_flashover += 1

                # Convert to YOLO format
                x_c, y_c, w_n, h_n = convert_bbox(width, height, x, y, bw, bh)

                # Write to .txt
                txt_file.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

# Summary output
print(" YOLO annotation conversion complete.")
print(f"🔹 Good insulators: {count_good}")
print(f"🔹 Broken insulators: {count_broken}")
print(f"🔹 Flashover insulators: {count_flashover}")
if missing_images:
    print(f" Missing images: {len(missing_images)}")
    for img in missing_images:
        print(f"   - {img}")
