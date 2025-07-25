import cv2
import base64
import asyncio
import json
import torch
import sqlite3
from collections import Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from ultralytics import YOLO

app = FastAPI()

def center_crop_square(frame):
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

def process_frame_to_640(frame):
    cropped = center_crop_square(frame)
    resized = cv2.resize(cropped, (640, 640))
    return resized

# --- 모델 로드 ---
model1 = YOLO("best_2.pt")
model1.to('cuda:0')

model2 = YOLO("best_last2.pt")

# --- 웹캠 2개 캡처 ---
cap1 = cv2.VideoCapture(0)  # 카메라 0
cap2 = cv2.VideoCapture(1)  # 카메라 1

# --- 탐지 결과 전역 저장 ---
latest_results_model1 = []
latest_results_model2 = []

# --- 모델1 감지 루프 (카메라0) ---
async def detect_loop_model1():
    global latest_results_model1
    while True:
        if not cap1.isOpened():
            cap1.open(0)
        ret, frame = cap1.read()
        if not ret:
            await asyncio.sleep(0.05)
            continue

        frame_640 = process_frame_to_640(frame)
        with torch.no_grad():
            results1 = model1.predict(frame_640, imgsz=320, conf=0.4, device='cuda:0')

        output1 = []
        obb1 = results1[0].obb
        if obb1 is not None and obb1.xyxyxyxy is not None:
            xyxyxyxy_np = obb1.xyxyxyxy.cpu().numpy()
            cls_ids = obb1.data[:, -1].cpu().numpy().tolist()
            if xyxyxyxy_np.ndim == 3 and xyxyxyxy_np.shape[2] == 1:
                xyxyxyxy_np = xyxyxyxy_np.squeeze(axis=2)
            for box, cls_id in zip(xyxyxyxy_np, cls_ids):
                points = [[float(p[0]), float(p[1])] for p in box]
                output1.append({
                    "points": points,
                    "class_id": int(cls_id),
                    "class_name": model1.names[int(cls_id)]
                })
        latest_results_model1 = output1

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        await asyncio.sleep(0.05)

# --- 모델2 감지 루프 (카메라1) ---
async def detect_loop_model2():
    global latest_results_model2
    while True:
        if not cap2.isOpened():
            cap2.open(1)
        ret, frame = cap2.read()
        if not ret:
            await asyncio.sleep(0.05)
            continue


        frame_640_2 = process_frame_to_640(frame)
        with torch.no_grad():
            results2 = model2.predict(frame_640_2, imgsz=320, conf=0.4, device='cuda:0')

        output2 = []
        obb2 = results2[0].obb
        if obb2 is not None and obb2.xyxyxyxy is not None:
            xyxyxyxy_np = obb2.xyxyxyxy.cpu().numpy()
            cls_ids = obb2.data[:, -1].cpu().numpy().tolist()
            if xyxyxyxy_np.ndim == 3 and xyxyxyxy_np.shape[2] == 1:
                xyxyxyxy_np = xyxyxyxy_np.squeeze(axis=2)
            for box, cls_id in zip(xyxyxyxy_np, cls_ids):
                points = [[float(p[0]), float(p[1])] for p in box]
                output2.append({
                    "points": points,
                    "class_id": int(cls_id),
                    "class_name": model2.names[int(cls_id)]
                })
        latest_results_model2 = output2

        await asyncio.sleep(0.03)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(detect_loop_model1())
    asyncio.create_task(detect_loop_model2())

# --- WebSocket 모델1 ---
@app.websocket("/ws")
async def websocket_endpoint_model1(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps({
                "model1": latest_results_model1
            }))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print("model1 클라이언트 연결 종료")

# --- WebSocket 모델2 ---
@app.websocket("/ws2")
async def websocket_endpoint_model2(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps({
                "model2": latest_results_model2
            }))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print("model2 클라이언트 연결 종료")

# --- 템플릿 세팅 ---
templates = Jinja2Templates(directory="template2")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})

# --- DB API: BOM ---
@app.get("/bom")
def get_bom_data():
    conn = sqlite3.connect("vision.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            h.part_code AS parent_code, 
            b.part_code, 
            b.part_name, 
            b.useage, 
            b.part_seq,
            b.cls_no
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
            "part_seq": row["part_seq"],
            "cls_no": row["cls_no"]
        })
    conn.close()
    return result

# --- DB API: work ---
@app.get("/work")
def get_work_data():
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


# ✅ MJPEG 스트리밍 제너레이터 (카메라0)
def mjpeg_generator_camera0():
    while True:
        if not cap1.isOpened():
            cap1.open(0)
        ret, frame = cap1.read()
        if not ret:
            continue
        frame_640 = process_frame_to_640(frame)
        ret, jpeg = cv2.imencode('.jpg', frame_640)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ✅ MJPEG 스트리밍 제너레이터 (카메라1)
def mjpeg_generator_camera1():
    while True:
        if not cap2.isOpened():
            cap2.open(1)
        ret, frame = cap2.read()
        if not ret:
            continue
        frame_640 = process_frame_to_640(frame)
        ret, jpeg = cv2.imencode('.jpg', frame_640)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ✅ MJPEG HTTP 엔드포인트 (카메라0)
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(mjpeg_generator_camera0(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

# ✅ MJPEG HTTP 엔드포인트 (카메라1)
@app.get("/video_feed_camera1")
async def video_feed_camera1():
    return StreamingResponse(mjpeg_generator_camera1(),
                             media_type='multipart/x-mixed-replace; boundary=frame')
