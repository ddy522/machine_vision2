import cv2
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from ultralytics import YOLO

app = FastAPI()

# YOLO 모델 로드
model = YOLO("best_2.pt")

# 웹캠 캡처 초기화
cap = cv2.VideoCapture(0)

latest_results = []

# 백그라운드 YOLO 탐지 루프
async def detect_loop():
    global latest_results
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = model.predict(frame, imgsz=640, conf=0.3, device='cpu')
        obb = results[0].obb
        
        print(model.names)

        output = []
        if obb is not None and obb.xyxyxyxy is not None:
            xyxyxyxy_np = obb.xyxyxyxy.cpu().numpy()
            cls_ids = obb.data[:, -1].cpu().numpy().tolist()

            if xyxyxyxy_np.ndim == 3 and xyxyxyxy_np.shape[2] == 1:
                xyxyxyxy_np = xyxyxyxy_np.squeeze(axis=2)
            class_names = model.names 
            for box, cls_id in zip(xyxyxyxy_np, cls_ids):
                points = [[float(p[0]), float(p[1])] for p in box]
                output.append({
                    "points": points,
                    "class_id": int(cls_id),
                    "class_name": class_names[int(cls_id)]  # 이름 추가
                })

        latest_results = output
        await asyncio.sleep(0.03)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(detect_loop())

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
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
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()  # 클라이언트가 아무 메시지 보내면 처리
            await websocket.send_text(json.dumps(latest_results))
            await asyncio.sleep(0.03)
    except Exception:
        await websocket.close()

