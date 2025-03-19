# 🚀 Traffic Signs Detection using YOLOv8

## 📌 **Project Overview**
This project implements **YOLOv8** for detecting traffic signs from images and videos. The dataset consists of annotated traffic signs, and the model is trained using **various augmentation techniques and anchor box optimizations** to improve small object detection.

---

## ⚙️ **Setup Instructions**
### **1️⃣ Install Dependencies**
Make sure you have Python installed, then run:
```bash
pip install ultralytics opencv-python numpy torch torchvision torchaudio
```

### **Train YOLOv8 Model**
Run the following command to train YOLO on your dataset:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model
model.train(data="data.yaml", epochs=50, imgsz=1280, batch=16, augment=True)
```

---

## 🧪 **Testing & Inference**
### **Run YOLOv8 on an Image**
```python
results = model("test_image.jpg", conf=0.3, iou=0.5)
results[0].show()
```

### **Run YOLOv8 on a Video**
```python
results = model("test_video.mp4", save=True)
```

### **Filter & Save High Confidence Detections**
```python
import cv2
cap = cv2.VideoCapture("test_video.mp4")
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 640))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for box in results[0].boxes:
        if box.conf > 0.5:
            frame = results[0].plot()
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
```

---
##**Video Frames**

https://drive.google.com/drive/folders/1BAVWYtH8G2sDp2wPw9ammDvTQL2R2Ijv?usp=sharing

---

##**Video Output**

https://drive.google.com/drive/folders/1huw_ccwy3_Bhu4xHsJZ_u82ujai7oIYQ?usp=sharing

---

## 🛠 **Future Improvements**
✅ Use **YOLOv8m or YOLOv8l** for better detection.
✅ Apply **super-resolution preprocessing** using ESRGAN.
✅ Fine-tune **focal loss** for handling class imbalance.
✅ Implement **real-time detection** on Raspberry Pi / Jetson Nano.

