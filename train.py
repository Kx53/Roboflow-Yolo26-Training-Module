from ultralytics import YOLO
import torch


def get_device():
    """เลือก device ที่เหมาะสมโดยอัตโนมัติ
    - Mac (M-series) → MPS
    - PC ที่มี NVIDIA GPU → CUDA (device 0)
    - อื่นๆ → CPU (fallback)
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def train_model():
    # Load Model (เลือก Nano เพื่อให้รันบน Pi 5)
    model = YOLO("yolo26n.pt")

    device = get_device()
    # batch=-1 (auto-tune 60% VRAM) ใช้ได้บน CUDA เท่านั้น
    # MPS / CPU ตั้งค่าตายตัวเพื่อความเสถียร
    batch = -1 if device == 0 else 16

    # เทรนตาม YOLO26 fine-tuning recipe (พร้อมส่งต่อไป ONNX end2end export)
    model.train(
        data="/home/awesome/training-module/Dog-face-detection-8/data.yaml",  # Path จากโฟลเดอร์ที่โหลด Roboflow มา
        epochs=150,
        imgsz=640,           # ต้องตรงกับ ONNX input [1, 3, 640, 640]
        device=device,
        batch=batch,
        patience=50,         # Early Stopping: ถ้าเทรนไป 50 รอบแล้วไม่ดีขึ้น ให้หยุดเทรนเพื่อกัน Overfitting
        optimizer='auto',    # auto เลือก MuSGD (YOLO26 native) หรือ AdamW ตามขนาด run
        lr0=0.001,           # lr ต่ำสำหรับ fine-tune (ป้องกันทำลาย pretrained weights)
        cos_lr=True,         # Cosine LR schedule
        close_mosaic=10,     # ปิด mosaic 10 epoch สุดท้าย (YOLO26 default)
        hsv_h=0.015,         # ปรับจูนสี (กันเรื่องแสงเปลี่ยนในจานข้าว)
        hsv_s=0.7,
        hsv_v=0.4,
        seed=0,              # reproducible สำหรับเขียนเล่มโปรเจค
        deterministic=True,
        plots=True,          # สร้างกราฟไว้ใส่เล่มโปรเจค
        project='runs/pet_feeder',
        name='yolo26n_v1',
    )


def export_to_onnx(
    weights_path: str = "runs/pet_feeder/yolo26n_v1/weights/best.pt",
    imgsz: int = 640,
    device: str = "cpu",
):
    """Export โมเดลที่เทรนแล้วเป็น ONNX แบบ end2end NMS-free

    YOLO26 มี dual-head architecture ระหว่างเทรน — ตอน export
    one-to-one head จะถูกใช้เป็น default ทำให้ไม่ต้องทำ NMS เพิ่ม

    Args:
        weights_path: path ไป best.pt ที่เทรนเสร็จ
        imgsz: input image size (default 640)
        device: device ตอน export — แนะนำ "cpu" เพื่อให้ได้ ONNX graph ที่
                portable ที่สุด (load บน Pi 5 ได้โดยไม่ฝัง CUDA op)
                ถ้าต้องการความเร็วในการ export ให้ใช้ "0" (CUDA) หรือ "mps"

    ONNX Output Spec:
        Input:  [1, 3, 640, 640]   float32   (RGB, normalized 0-1)
        Output: [1, 300, 6]        float32   → [x1, y1, x2, y2, conf, class_id]
        Box format: xyxy (top-left + bottom-right)

    Returns:
        path ของไฟล์ .onnx ที่ export แล้ว
    """
    model = YOLO(weights_path)
    return model.export(
        format="onnx",
        imgsz=imgsz,
        batch=1,             # batch fix ที่ 1 ตาม spec [1, 3, 640, 640]
        end2end=True,        # NMS-free one-to-one head (default ของ YOLO26)
        simplify=True,       # ใช้ onnxslim simplify graph
        opset=19,            # ONNX opset ที่ stable กับ runtime ส่วนใหญ่
        dynamic=False,       # input shape คงที่ (Pi 5 ไม่ต้องการ dynamic batch)
        half=False,          # หมายเหตุ: ถ้า half=True, output0 ยังเป็น FP32 (เก็บ class_id)
        device=device,
    )


if __name__ == "__main__":
    train_model()
    # เมื่อเทรนเสร็จแล้ว uncomment บรรทัดล่างเพื่อ export เป็น ONNX อัตโนมัติ:
    # export_to_onnx()
