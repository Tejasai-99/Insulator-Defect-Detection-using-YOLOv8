# Insulator-Defect-Detection-using-YOLOv8


This project involves detecting defects in high-voltage transmission line insulators using the IDID dataset. The model is trained using YOLOv8, a state-of-the-art object detection algorithm, to classify insulator conditions such as **good**, **broken**, and **flashover**.

## Dataset: IDID (IEEE)

- The dataset used is **IDID (Insulator Defect Identification Dataset)**, published on IEEE DataPort.
- It includes images of insulators captured in real-world transmission line environments with annotation labels for different defect types.

> **Note:** Due to licensing restrictions, the dataset is not included in this repository. You can download it directly from [IEEE DataPort](https://ieee-dataport.org).

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Framework:** PyTorch, Ultralytics YOLOv8  
- **Libraries:** OpenCV, NumPy, Matplotlib  
- **Annotation Format:** Converted to YOLO format from COCO-style JSON
