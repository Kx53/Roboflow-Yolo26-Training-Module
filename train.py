from ultralytics import YOLO
import torch

def train_model():
    # Load Model (เลือก Nano เพื่อให้รันบน Pi 5)
    model = YOLO("yolo26n.pt")

    # เช็ค Device
    # ถ้าเป็น Mac จะใช้ "mps", ถ้าเป็น PC มีการ์ดจอจะใช้ "cuda" (0)
    device = "mps" if torch.backends.mps.is_available() else "0"

    # เทรนเลย ใช้ template จาก Ultralytics Docs เอา
    model.train(
        data="/home/awesome/training-module/Dog-face-detection-7/data.yaml", # Path จากโฟลเดอร์ที่โหลด Roboflow มา
        epochs=150,
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