# 🚗 Car Object Detection using YOLOv8

This repository covers the complete pipeline of developing a computer vision model to detect cars on the road. The project was built in the Kaggle environment using the **YOLOv8** architecture by Ultralytics.

---

## 🎯 Project Objective
Develop and train a lightweight object detection model to accurately and rapidly detect cars in road traffic frames.

---

## 🚀 Metrics and Training Results
The model was successfully trained on a Kaggle **Tesla T4 GPU** for **10 epochs**. It reached a plateau quickly and delivered excellent results:

* **mAP50**: `0.987` (98.7% accuracy in object detection)
* **mAP50-95**: `0.621` (high bounding box localization accuracy)
* **Inference Speed**: Only `5.4 - 5.8 ms` per image!

### 📊 Model Inference Example
Here is how the trained model performs on a test image:

<img width="609" height="358" alt="image" src="https://github.com/user-attachments/assets/12f1175d-999c-4d83-b1ba-39abb5d1c2b4" />


---

## 📁 Dataset Structure and Preprocessing
The original dataset contained road images and a CSV file with absolute coordinates `[xmin, ymin, xmax, ymax]`.

The data was preprocessed as follows:
1. **Normalization**: Bounding box coordinates were converted to a relative scale of `[0, 1]` based on image dimensions.
2. **Formatting**: Generated `.txt` annotation files in the standard YOLO format: `<class_id> <x_center> <y_center> <width> <height>`.
3. **Data Splitting**: The dataset was randomly split using `scikit-learn` with an **80 / 20** ratio:
   * **Train**: 284 images
   * **Val**: 71 images

Final dataset folder structure:
```text
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## 🛠️ Tech Stack
* **Language**: Python 3.12
* **CV Framework**: Ultralytics YOLOv8 (`yolov8n.pt` base model)
* **Data Engineering**: Pandas, NumPy, Scikit-learn
* **Visualization**: OpenCV, Matplotlib

---

## 💻 How to Run This Project Locally

### 1. Install Dependencies
```bash
pip install ultralytics opencv-python matplotlib
```

### ⏬ Download Model Weights
Before running the script, download the trained weights file:
* **[Download best.pt from Google Drive](https://drive.google.com/file/d/1nHpAfTq8GzmQ0BLxElW9svHHGD8_ogMH/view?usp=drive_link)**

Place the `best.pt` file into your local project folder.

```python
from ultralytics import YOLO
import matplotlib.pyplot as plt


model = YOLO('best.pt')


results = model.predict(source='path_to_your_image.jpg', save=True, conf=0.4)


result_img = plt.imread('runs/detect/predict/path_to_your_image.jpg')
plt.figure(figsize=(10, 7))
plt.imshow(result_img)
plt.axis('off')
plt.show()
```

---


