# main.py (FastAPI 서버) - model2 제거 버전

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

# ✅ 중앙 크롭 및 640x640 리사이즈 함수
def center_crop_square(frame):
    """프레임을 정사각형으로 중앙 크롭"""
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    return cropped

def process_frame_to_640(frame):
    """프레임을 중앙 크롭 후 640x640으로 리사이즈"""
    cropped = center_crop_square(frame)
    resized = cv2.resize(cropped, (640, 640))
    return resized

# YOLO 모델 1 로드
model = YOLO("best_2.pt")
model.to('cuda:0')  # GPU

# 웹캠 캡처
cap = cv2.VideoCapture(0)

# ROI/CLASS 모드 선택
MODE = 'class'  # 'roi' or 'class'

# ROI 좌표 설정
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 100, 100, 500, 400

# 전역 상태
latest_results_model1 = []
prev_gray_roi = None
prev_counts = None
workInProgress = False  # 작업중 상태 변수


# YOLO 추론 루프
async def detect_loop():
    global latest_results_model1, prev_gray_roi, prev_counts, workInProgress

    while True:
        if not cap.isOpened():
            cap.open(0)

        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue

        run_yolo = True

        # ✅ 640x640 크기로 변환 (중앙 크롭 후 리사이즈)
        frame_640 = process_frame_to_640(frame)

        if MODE == 'roi':
            roi = frame_640[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
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
                temp_results = model.predict(frame_640, imgsz=320, conf=0.4, device='cuda:0')
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
            # 모델1 추론
            with torch.no_grad():
                results1 = model.predict(frame_640, imgsz=320, conf=0.4, device='cuda:0')
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
                        "class_name": model.names[int(cls_id)]
                    })
            latest_results_model1 = output1

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
        frame_640 = process_frame_to_640(frame)
        if MODE == 'roi':
            cv2.rectangle(frame_640, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame_640)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

templates = Jinja2Templates(directory="template2")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/video_feed_camera0")
def video_feed_camera0():
    def gen():
        cap0 = cv2.VideoCapture(0)
        while True:
            ret, frame = cap0.read()
            if not ret:
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global workInProgress  # ✅ 전역변수!
    await websocket.accept()
    try:
        while True:
            try:
                msg_task = asyncio.create_task(websocket.receive_text())
                done, pending = await asyncio.wait({msg_task}, timeout=5)
                if msg_task in done:
                    msg = msg_task.result()
                    try:
                        msg_json = json.loads(msg)
                        if msg_json.get("type") == "work_status":
                            workInProgress = bool(msg_json.get("value", False))
                            print(f"✅ [WebSocket] workInProgress → {workInProgress}")
                        else:
                            combined = {
                                "model1": latest_results_model1
                            }
                            await websocket.send_text(json.dumps(combined))
                    except json.JSONDecodeError:
                        if msg == "ping":
                            await websocket.send_text(json.dumps({"status": "pong"}))
                        else:
                            combined = {
                                "model1": latest_results_model1
                            }
                            await websocket.send_text(json.dumps(combined))
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

@app.get("/bom")
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

