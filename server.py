from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

# YOLO 모델 로드
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # YOLO 추론 (GPU 사용: device=0)
    results = model.predict(image, device=0)  # 👈 여기서 device=0 추가!

    output = []
    for box in results[0].boxes.xyxy:
        output.append({
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
        })

    return {"result": output}



# 추가: DataGrid에 보낼 더미 JSON 데이터
@app.get("/get_data")
def get_data():
    dummy_data = [
        {"id": 1, "name": "Item A", "value": 123},
        {"id": 2, "name": "Item B", "value": 456},
        {"id": 3, "name": "Item C", "value": 789},
    ]
    return dummy_data

# venv\Scripts\activate
# uvicorn server:app --reload --host 0.0.0.0 --port 8000