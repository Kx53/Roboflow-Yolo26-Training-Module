# 🧠 YOLO26 Training & Export Guide (Pet Feeder Project)

คู่มือการเตรียม Dataset, เทรนโมเดล และการแปลงไฟล์เพื่อนำไปใช้บน Raspberry Pi 5 สำหรับโปรเจคตรวจจับใบหน้าสุนัขและสถานะชามอาหาร

## 🚀 Quick Start

โปรเจคนี้ใช้ `uv` สำหรับจัดการ Python Environment และ Dependencies เพื่อความรวดเร็วและแม่นยำ

```bash
# ติดตั้ง dependencies ทั้งหมด
uv sync
```

---

## 1. Environment Setup

ใช้ Python 3.14 และ Library:

- **Core**: `ultralytics` (YOLO26)
- **Data**: `roboflow`, `python-dotenv`
- **Export (ONNX – แนะนำสำหรับ Pi 5)**: `onnx`, `onnxslim`, `onnxruntime-gpu`
- **Export (NCNN – fallback)**: `ncnn`, `pnnx` *(ไม่รองรับ end2end ของ YOLO26 ดูหัวข้อที่ 5)*

### การตั้งค่า Environment Variables

สร้างไฟล์ `.env` จาก `.env.example`:

```bash
cp .env.example .env
```

จากนั้นใส่ค่าของคุณ:

- `ROBOFLOW_API_KEY`: API Key จาก Roboflow Settings
- `ROBOFLOW_DATASET_VERSION`: เวอร์ชันของ Dataset ที่ต้องการเทรน

---

## 2. Dataset Preparation

ดาวน์โหลด Dataset จาก Roboflow โดยใช้สคริปต์ที่เตรียมไว้:

```bash
# รันเพื่อโหลดข้อมูลลงเครื่อง
python download_data.py
```

สคริปต์จะดึงข้อมูลจากโปรเจค `dog-face-detection-0uxs8` ในรูปแบบ YOLO26 และบันทึกลงในโฟลเดอร์ `Dog-face-detection-X` โดยอัตโนมัติ

---

## 3. Training Process

เราใช้โมเดล **YOLO26n (Nano)** เพื่อความลื่นไหลสูงสุดบน Raspberry Pi 5

### สคริปต์การเทรน (`train.py`)

รันการเทรนด้วยคำสั่ง:

```bash
python train.py
```

**การตั้งค่าสำคัญ:**

- `epochs`: 150 (พร้อม Early Stopping `patience=50`)
- `imgsz`: 640 (ตรงกับ ONNX input `[1, 3, 640, 640]`)
- `optimizer`: `auto` → เลือก **MuSGD** (YOLO26 native) หรือ **AdamW** อัตโนมัติตามขนาด run
- `lr0`: 0.001 (Fine-tuning learning rate ต่ำเพื่อรักษา pretrained weights)
- `cos_lr`: เปิด Cosine LR schedule ให้ converge นุ่มนวล
- `close_mosaic`: 10 (ปิด mosaic 10 epoch สุดท้าย — YOLO26 default recipe)
- `seed=0`, `deterministic=True`: เทรนซ้ำได้ผลเดิม (สำคัญต่อการเขียนรายงาน)
- HSV augmentation: ปรับสี/แสงเพื่อกัน overfitting จากสภาพแสงในจานอาหาร

---

## 4. Latest Training Results (6 พ.ค. 2569)

ผลลัพธ์ล่าสุดจากการเทรนเวอร์ชัน `yolo26n_v1`:

- **mAP50**: 0.977 (ความแม่นยำสูงมาก)
- **mAP50-95**: 0.882
- **Performance by Class**:
  - `bowl_empty`: 0.995 mAP50
  - `bowl_full`: 0.995 mAP50
  - `yuri-dog`: 0.941 mAP50
- **Inference Speed**: ~1.0ms (บน RTX 5060 Ti)

---

## 5. Export to ONNX (NMS-free สำหรับ Raspberry Pi 5)

YOLO26 รองรับ **end-to-end NMS-free inference** เป็น default — output ของ ONNX จะถูกกรองและจัดเรียงเรียบร้อยมาแล้วในรูป `(1, 300, 6)` โดยไม่ต้องเขียน Non-Maximum Suppression เอง

### 5.1 รัน Export Script

มีฟังก์ชัน `export_to_onnx()` เตรียมไว้ใน `train.py` แล้ว:

```bash
# Export โมเดล best.pt → best.onnx
python -c "from train import export_to_onnx; export_to_onnx()"
```

หรือเรียกตรงๆ:

```python
from ultralytics import YOLO

model = YOLO("runs/pet_feeder/yolo26n_v1/weights/best.pt")
model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    end2end=True,    # NMS-free one-to-one head (default ของ YOLO26)
    simplify=True,
    opset=19,
    dynamic=False,
    half=False,      # หมายเหตุ: half=True → output0 ยังคงเป็น FP32 (เก็บ class_id)
)
```

### 5.2 ONNX Output Specification

ผลลัพธ์ที่ได้ตรงตามมาตรฐาน YOLO26 end2end:

| Property      | Value                                                         |
| ------------- | ------------------------------------------------------------- |
| Format        | ONNX                                                          |
| Input         | `[1, 3, 640, 640]` float32 (RGB, normalized 0-1)              |
| Output        | `[1, 300, 6]` float32                                         |
| Layout        | `[x1, y1, x2, y2, confidence, class_id]`                      |
| Box format    | `xyxy` (มุมซ้ายบน + มุมขวาล่าง)                                |
| NMS           | ✅ Built-in — ไม่ต้อง post-process เอง                          |
| Max detection | 300 ต่อภาพ                                                    |

### 5.3 Inference บน Raspberry Pi 5 ด้วย ONNX Runtime

ติดตั้งบน Pi 5 (CPU only):

```bash
pip install onnxruntime opencv-python numpy
```

ตัวอย่างโค้ด inference (รองรับภาพไม่ใช่จัตุรัสอย่างถูกต้อง):

```python
import onnxruntime as ort
import numpy as np
import cv2

CLASSES = ["dog_yuri", "dog_makham", "unknown",
           "food_full", "food_low", "food_empty"]
CONF_THRESHOLD = 0.25
INPUT_SIZE = 640


def letterbox(img, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    """Resize + pad ภาพให้รักษา aspect ratio (ตรงกับตอน Ultralytics เทรน)
    คืนค่า (img, ratio, (pad_w, pad_h)) เพื่อใช้ scale bbox กลับ
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    pad_w, pad_h = (new_shape - new_w) / 2, (new_shape - new_h) / 2

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (pad_w, pad_h)


# โหลดโมเดล
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Preprocess: letterbox → RGB → CHW → normalize → batch
img_orig = cv2.imread("test.jpg")
img_lb, r, (pad_w, pad_h) = letterbox(img_orig)
img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
input_tensor = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

# Inference — output shape: (1, 300, 6)
output = session.run(None, {input_name: input_tensor})[0]
detections = output[0]  # (300, 6)

# Filter ด้วย confidence อย่างเดียว — ไม่ต้องทำ NMS!
detections = detections[detections[:, 4] > CONF_THRESHOLD]

for det in detections:
    x1, y1, x2, y2, conf, cls_id = det

    # Scale พิกัดจาก 640×640 (มี pad) กลับสู่ภาพต้นฉบับ
    x1 = (x1 - pad_w) / r
    y1 = (y1 - pad_h) / r
    x2 = (x2 - pad_w) / r
    y2 = (y2 - pad_h) / r

    label = CLASSES[int(cls_id)]
    print(f"{label}: {conf:.2f} @ ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

    # วาดกล่องบนภาพต้นฉบับ
    cv2.rectangle(img_orig, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img_orig, f"{label} {conf:.2f}", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
```

> 💡 **ทำไมต้อง letterbox?** Ultralytics เทรนด้วย letterbox preprocessing (resize keep aspect ratio + pad) — ถ้าใช้ `cv2.resize` ตรงๆ บนภาพไม่ใช่จัตุรัส (เช่นจากกล้อง 1920×1080) input distribution จะต่างจากตอนเทรน → mAP อาจตก 2-5%

### 5.4 ⚠️ ทำไมไม่ใช้ NCNN?

NCNN และ format อื่นบางตัว (RKNN, PaddlePaddle, ExecuTorch, IMX, Edge TPU) **ไม่รองรับ end2end output** ของ YOLO26 — ระบบจะ fallback ไปใช้ one-to-many head อัตโนมัติ ทำให้ output กลายเป็น `(1, nc + 4, 8400)` แบบ YOLO11 และต้องเขียน NMS เอง

ใช้ **ONNX Runtime** บน Pi 5 จะได้ pipeline ที่สะอาดและเขียน post-processing น้อยกว่ามาก หากต้องการความเร็วสูงสุดและยอมเขียน NMS เอง สามารถใช้ NCNN ได้ผ่าน `model.export(format="ncnn")` แต่ **output จะไม่ตรงตามสเปก `(1, 300, 6)` ที่กล่าวไว้ข้างต้น**

---

## 📂 Project Structure

- `train.py`: สคริปต์หลัก — มี `train_model()` และ `export_to_onnx()`
- `download_data.py`: สคริปต์ดึง Dataset จาก Roboflow
- `yolo26n.pt`: Pre-trained weight เริ่มต้น
- `runs/pet_feeder/yolo26n_v1/`: ผลลัพธ์การเทรน (weights, charts, confusion matrix)
  - `weights/best.pt`: Best checkpoint สำหรับ export
  - `weights/best.onnx`: ONNX end2end (สร้างจาก `export_to_onnx()`)
- `.env`: ไฟล์เก็บความลับ (API Keys)
