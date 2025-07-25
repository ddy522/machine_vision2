import cv2
import base64
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi import Request
from ultralytics import YOLO
import sqlite3

app = FastAPI()

model = YOLO("best_last2.pt")  # OBB 지원 모델이면 변경

templates = Jinja2Templates(directory="template2")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index3.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # 0번 웹캠

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # 1️⃣ YOLO 실행
            results = model(frame)

            # 2️⃣ OBB 결과 가공
            detections = []  # ✅ 이 리스트만 바꿔서 points 구조로!
            obb = results[0].obb

            if obb is not None and obb.xyxyxyxy is not None:
                xyxyxyxy_np = obb.xyxyxyxy.cpu().numpy()
                cls_ids = obb.data[:, -1].cpu().numpy().tolist()

                if xyxyxyxy_np.ndim == 4:
                    xyxyxyxy_np = xyxyxyxy_np.squeeze(-1)

                for box, cls_id in zip(xyxyxyxy_np, cls_ids):
                    points = [[float(x), float(y)] for x, y in box]
                    detections.append({
                        "points": points,
                        "class_id": int(cls_id),
                        "class_name": model.names[int(cls_id)]
                    })

            # 3️⃣ 이미지 인코딩
            annotated_frame = results[0].plot()
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            frame_bytes = base64.b64encode(buffer).decode("utf-8")

            # 4️⃣ JSON 통합 송신 — ⚡ 그대로!
            await websocket.send_text(json.dumps({
                "image": frame_bytes,    # ✅ 이미지 그대로!
                "detections": detections # ✅ 새 구조!
            }))

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("클라이언트 연결 끊김")
    finally:
        cap.release()


@app.get("/work")
def get_data2():
    conn = sqlite3.connect("vision.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
       SELECT work_seq, work_task, bom_code, skey
       FROM work
    """)
    rows = cur.fetchall()
    result = []
    for row in rows:
        result.append({
            "work_seq": row["work_seq"],
            "work_task": row["work_task"],
            "bom_code": row["bom_code"]
        })
    conn.close()
    return result
