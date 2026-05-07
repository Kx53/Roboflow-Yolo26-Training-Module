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
- **Export**: `ncnn`, `pnnx`, `onnxslim`, `onnx`

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
- `epochs`: 150 (พร้อม Early Stopping 50 รอบ)
- `imgsz`: 640
- `optimizer`: Auto (AdamW/SGD ขึ้นอยู่กับระบบ)
- `augment`: เปิดใช้งานเพื่อป้องกัน Overfitting จากสภาพแสง
- `close_mosaic`: 10 (เทคนิค YOLO26 เพื่อความแม่นยำช่วงท้าย)

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

## 5. Export to NCNN (for Raspberry Pi 5)

เพื่อให้รันบน Pi 5 ได้โดยไม่ต้องใช้ PyTorch ให้แปลงไฟล์เป็น NCNN:

```python
from ultralytics import YOLO

# Load โมเดลที่เทรนเสร็จแล้ว
model = YOLO("runs/detect/runs/pet_feeder/yolo26n_v1/weights/best.pt")

# Export เป็น NCNN format (จะได้โฟลเดอร์ชื่อ 'best_ncnn_model')
model.export(format="ncnn", half=True)
```

---

## 📂 Project Structure

- `train.py`: สคริปต์หลักสำหรับการเทรน
- `download_data.py`: สคริปต์ดึง Dataset จาก Roboflow
- `yolo26n.pt`: Pre-trained weight เริ่มต้น
- `runs/`: ผลลัพธ์การเทรน (Weights, Charts, Confusion Matrix)
- `.env`: ไฟล์เก็บความลับ (API Keys)
