# 🧠 YOLO26 Training & Export Guide

คู่มือการเตรียม Dataset, เทรนโมเดล และการแปลงไฟล์เพื่อนำไปใช้บน Raspberry Pi 5

🛠 Prerequisites

- Environment Manager: 'uv'
- Machine: Mac (M-Series), PC (with NVIDIA GPU), หรือ Google Colab
- Target: YOLO26 (Ultralytics)

---

## 1. Environment Setup

ใช้ 'uv' เพื่อสร้างสภาพแวดล้อมจำลองที่สะอาดและรวดเร็ว:

```bash
# สร้างโปรเจคสำหรับเทรน
mkdir training-module && cd training-module

# สร้าง Virtual Environment โดยระบุ Python 3.14
uv venv --python 3.14
source .venv/bin/activate
# setup uv (ถ้ายังไม่เคยใช้) และเริ่มต้นโปรเจค
uv init

# ติดตั้ง Library ที่จำเป็น (สำหรับเทรนและ Export opencv-python เพื่อความเสถียรในการดึงภาพจากกล้อง ncnn pnnx เพื่อให้ export(format="ncnn") ทำงานได้สมบูรณ์ และรวม Roboflow สำหรับดึง Dataset)
uv add ultralytics roboflow onnx onnxslim ncnn pnnx opencv-python
```

---

## 2. Dataset Preparation (ดึงจาก Roboflow)

จัดโครงสร้างโฟลเดอร์ตามมาตรฐานของ Ultralytics:

/datasets
└── pet_feeder_data
├── train/ # 80%
├── val/ # 10%
└── test/ # 10%

Dataset Config (data.yaml):

```yaml
path: ../datasets/pet_feeder_data
train: train/images
val: val/images
test: test/images

names:
  0: dog_yuri # ตัวที่ 1
  1: dog_makham # ตัวที่ 2
  2: unknown
  3: food_full
  4: food_low
  5: food_empty
```

สร้างไฟล์ชื่อ 'download_data.py' เพื่อดึง Dataset ที่คุณทำไว้ใน Roboflow ลงมาที่เครื่อง:

```python
from roboflow import Roboflow

# ใส่ API Key และชื่อโปรเจคของคุณจากหน้าเว็บ Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("your-workspace").project("pet-feeder-project")
version = project.version(1) # ระบุเวอร์ชันที่ต้องการ

# Download ในรูปแบบ YOLO26
dataset = version.download("yolo26")
```

---

## 3. Training Process (YOLO26)

รันคำสั่งเทรนโดยเน้น Augmentation เพื่อป้องกันโมเดลยึดติดกับสีอาหารหรือสภาพแสง:

```Python
from ultralytics import YOLO
import torch

def train_model():
    # 1. Load Model (เลือก Nano เพื่อให้รันบน Pi 5 ได้ลื่นๆ)
    model = YOLO("yolo26n.pt")

    # 2. เช็ค Device
    # ถ้าเป็น Mac จะใช้ "mps", ถ้าเป็น PC มีการ์ดจอจะใช้ "cuda" (0)
    device = "mps" if torch.backends.mps.is_available() else "0"

    # 3. เริ่มเทรน
    model.train(
        data="/home/awesome/training-module/Dog-face-detection-4/data.yaml", # Path จากโฟลเดอร์ที่โหลด Roboflow มา
        epochs=300,
        imgsz=640,
        device=device,
        batch=-1,
        patience=50,       # Early Stopping: ถ้าเทรนไป 50 รอบแล้วไม่ดีขึ้น ให้หยุดเทรนเพื่อกัน Overfitting
        augment=True,      # เปิด Augmentation (ช่วยให้โมเดลเรียนรู้ได้ดีขึ้น)
        optimizer='AdamW', # มาตรฐานปี 2026 ให้ความแม่นยำสูงกว่า SGD ในงาน Detect สัตว์เลี้ยง
        hsv_h=0.015,       # ปรับจูนสี (กันเรื่องแสงเปลี่ยนในจานข้าว)
        hsv_s=0.7,
        hsv_v=0.4,
        plots=True         # สร้างกราฟไว้ใส่เล่มโปรเจค
    )

if __name__ == "__main__":
    train_model()
```

---

## 4. Export to NCNN

เพื่อให้ใช้งานบน Pi 5 ได้โดยไม่ต้องลง PyTorch เราต้องแปลงไฟล์ตามลำดับดังนี้:

แบบ Python:

```python
from ultralytics import YOLO

# Load โมเดลที่คุณเทรนเสร็จแล้ว
model = YOLO("runs/detect/train/weights/best.pt")

# Export เป็น NCNN format (จะได้โฟลเดอร์ชื่อ 'best_ncnn_model')
model.export(format="ncnn", half=True, device="0")
```

แบบ CLI:

```bash
yolo export model=runs/detect/train/weights/best.pt format=ncnn half=True device=0
```

(หมายเหตุ: การใส่ half=True จะช่วยทำ FP16 quantization ให้โมเดลเล็กลงครึ่งหนึ่งแต่ยังแม่นยำอยู่)

## 5. Validation & Reporting

หลังเทรนเสร็จ อย่าลืมเก็บค่าเหล่านี้ไว้เขียนเล่มโปรเจค:

- mAP50 / mAP50-95: เพื่อยืนยันความแม่นยำ
- Confusion Matrix: ดูว่าโมเดลสับสนระหว่าง "อาหารน้อย" กับ "ถาดเปล่า" หรือไม่
- Inference Speed: ทดสอบความเร็วบนเครื่องเทรนเพื่อเปรียบเทียบ

---

สรุปสิ่งที่ต้องย้ายไปที่ Pi 5:

1. ไฟล์ '.param' และ '.bin'
2. ไฟล์ 'data.yaml' (ไว้ใช้อ้างอิง Class names)
