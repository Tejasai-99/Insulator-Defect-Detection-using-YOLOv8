import os
import shutil
import random

# Set your paths
images_path = "C:/new yolo/Images"
labels_path = "C:/new yolo/yolo_labels"
output_base = "C:/new yolo/YOLO_dataset"

# Train/val split ratio
split_ratio = 0.8

# Create required folders
folders = [
    "images/train", "images/val",
    "labels/train", "labels/val"
]
for folder in folders:
    os.makedirs(os.path.join(output_base, folder), exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle and split
random.shuffle(image_files)
split_index = int(len(image_files) * split_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def move_files(file_list, subset):
    for img_file in file_list:
        # Move image
        src_img = os.path.join(images_path, img_file)
        dst_img = os.path.join(output_base, f"images/{subset}", img_file)
        shutil.copy2(src_img, dst_img)

        # Move label
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(labels_path, label_file)
        dst_label = os.path.join(output_base, f"labels/{subset}", label_file)

        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"⚠️ Label not found for: {img_file}")

# Move files
move_files(train_files, "train")
move_files(val_files, "val")

print("Dataset successfully split and organized.")
print(f"🔹 Train images: {len(train_files)}")
print(f"🔹 Val images: {len(val_files)}")
