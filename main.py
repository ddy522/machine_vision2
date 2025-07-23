# main.py (FastAPI 서버)

import cv2
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

# YOLO 모델 로드 (GPU)
model = YOLO("best_2.pt")
model.to('cuda:0')

# 웹캠 캡처
cap = cv2.VideoCapture(1)

# ROI/CLASS 모드 선택
MODE = 'class'  # 'roi' or 'class'

# ROI 좌표 설정
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 100, 100, 500, 400

# 전역 상태
latest_results = []
prev_gray_roi = None
prev_counts = None

# YOLO 추론 루프
async def detect_loop():
    global latest_results, prev_gray_roi, prev_counts

    while True:
        if not cap.isOpened():
            cap.open(0)

        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue

        run_yolo = True

        if MODE == 'roi':
            roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if prev_gray_roi is not None:
                frame_diff = cv2.absdiff(prev_gray_roi, gray_roi)
                diff_score = frame_diff.mean()
                if diff_score < 10:
                    run_yolo = False
                else:
                    print(f"📸 ROI 변화 감지됨 (score={diff_score:.1f}) → YOLO 실행")

            prev_gray_roi = gray_roi

        elif MODE == 'class':
            with torch.no_grad():
                temp_results = model.predict(
                    frame,
                    imgsz=320,
                    conf=0.4,
                    device='cuda:0'
                )
            obb = temp_results[0].obb
            current_counts = {}

            if obb is not None and obb.cls is not None:
                cls_ids = obb.cls.cpu().numpy().astype(int)
                current_counts = dict(Counter(cls_ids))

            if prev_counts == current_counts:
                run_yolo = False
            else:
                print(f"🆕 객체 구성 변화 감지 → YOLO 실행")

            prev_counts = current_counts.copy()

        if run_yolo:
            with torch.no_grad():
                results = model.predict(
                    frame,
                    imgsz=320,
                    conf=0.4,
                    device='cuda:0'
                )

            obb = results[0].obb
            output = []
            if obb is not None and obb.xyxyxyxy is not None:
                xyxyxyxy_np = obb.xyxyxyxy.cpu().numpy()
                cls_ids = obb.data[:, -1].cpu().numpy().tolist()

                if xyxyxyxy_np.ndim == 3 and xyxyxyxy_np.shape[2] == 1:
                    xyxyxyxy_np = xyxyxyxy_np.squeeze(axis=2)

                for box, cls_id in zip(xyxyxyxy_np, cls_ids):
                    points = [[float(p[0]), float(p[1])] for p in box]
                    output.append({
                        "points": points,
                        "class_id": int(cls_id),
                        "class_name": model.names[int(cls_id)]
                    })

            latest_results = output

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        await asyncio.sleep(0.05)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(detect_loop())

def gen_frames():
    while True:
        if not cap.isOpened():
            cap.open(0)

        success, frame = cap.read()
        if not success:
            continue

        if MODE == 'roi':
            cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

templates = Jinja2Templates(directory="template2")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                msg_task = asyncio.create_task(websocket.receive_text())
                done, pending = await asyncio.wait({msg_task}, timeout=5)
                if msg_task in done:
                    msg = msg_task.result()
                    if msg == "ping":
                        await websocket.send_text(json.dumps({"status": "pong"}))
                    else:
                        await websocket.send_text(json.dumps(latest_results))
                else:
                    await websocket.send_text(json.dumps({"type": "ping"}))
            except Exception as e:
                print(f"WebSocket receive/send error: {e}")
                break
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        print("❌ 클라이언트 연결 종료")
    except Exception as e:
        print(f"❌ WebSocket 오류: {e}")
    finally:
        await websocket.close()

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


@app.get("/get_data2")
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
