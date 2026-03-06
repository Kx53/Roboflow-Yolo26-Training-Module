from dotenv import load_dotenv
from roboflow import Roboflow
import os

# โหลดตัวแปรจากไฟล์ .env  
load_dotenv()

# ใส่ API Key และชื่อโปรเจคของคุณจากหน้าเว็บ Roboflow
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("computer-vision-o4dbt").project("dog-face-detection-0uxs8")
version = project.version(int(os.getenv("ROBOFLOW_DATASET_VERSION"))) # ระบุเวอร์ชันที่ต้องการ

# Download ในรูปแบบ YOLO26
dataset = version.download("yolo26")