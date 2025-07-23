from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import sqlite3
import numpy as np

app = FastAPI()

# YOLO 모델 로드
model = YOLO("best_2.pt")
def to_float(val):
    # 넘파이 배열 또는 텐서면 값 하나 꺼내서 float 변환
    if isinstance(val, (np.ndarray,)):
        if val.size == 1:
            return float(val.item())
        else:
            return float(val.flat[0])
    return float(val)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    width, height = image.size

    results = model.predict(image)
    obb = results[0].obb

    output = []

    if obb is not None and obb.xyxyxyxy is not None:
        xyxyxyxy_tensor = obb.xyxyxyxy
        xyxyxyxy_cpu = xyxyxyxy_tensor.cpu()
        xyxyxyxy_np = xyxyxyxy_cpu.numpy()

        # 차원 줄이기 (필요 시)
        if xyxyxyxy_np.ndim == 3 and xyxyxyxy_np.shape[2] == 1:
            xyxyxyxy_np = xyxyxyxy_np.squeeze(axis=2)

        for box in xyxyxyxy_np:
            points = []
            # 8개 좌표 (x1,y1,x2,y2,x3,y3,x4,y4)
            for i in range(0, 8, 2):
                points.append([float(box[i]), float(box[i + 1])])
            output.append({"points": points})

    return {
        "result": output,
        "image_width": width,
        "image_height": height
    }

@app.get("/get_data")
def get_data():
    conn = sqlite3.connect("vision.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            h.part_code AS parent_code, 
            b.part_code, 
            b.part_name, 
            b.useage, 
            b.part_seq
        FROM bomh h 
        LEFT JOIN bom b ON b.pskey = h.skey
        ORDER BY b.part_seq
    """)

    rows = cur.fetchall()
    result = []

    for row in rows:
        result.append({
            "parent_code": row["parent_code"],
            "part_code": row["part_code"],
            "part_name": row["part_name"],
            "useage": row["useage"],
            "part_seq": row["part_seq"]
        })

    conn.close()
    return result
