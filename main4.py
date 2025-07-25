# ======================================================================================
# 실시간 조립 공정 검증 시스템 (FastAPI + YOLOv8-OBB) - 로컬 웹캠 버전
# ======================================================================================

# --- 1. 필수 라이브러리 임포트 ---
import cv2
import torch
import numpy as np
import asyncio
import base64
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from contextlib import asynccontextmanager

# --- 2. 시스템 전역 설정 (Configuration) ---
MODEL_PATH = "best_last2.pt"  # 학습된 YOLOv8-OBB 모델 파일의 경로
CAMERA_INDEX = 0  # 사용할 기본 카메라 인덱스 (0: 내장, 1: 첫 번째 외장 등)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 연산 장치 설정
CONF_THRESHOLD = 0.8  # 신뢰도 임계값

# --- 3. 공정 상태 관리 (State Management) ---
ASSEMBLY_STATE = {
    "current_step": 1,
    "steps_info": {
        1: {"name": "빨강+초록", "class": "조립1", "message": "1단계: 빨강 블럭과 초록 블럭을 조립하세요."},
        2: {"name": "빨강+초록+파랑1", "class": "조립2", "message": "2단계: 파란색 삼각 블럭 하나를 추가하세요."},
        3: {"name": "빨강+초록+파랑2", "class": "조립3", "message": "3단계: 마지막 파란색 삼각 블럭을 추가하세요."}
    },
    "completion_message": "🎉 조립이 완료되었습니다! 🎉",
    "error_message": "잘못된 조립입니다. 현재 단계에 맞게 다시 조립해주세요."
}

# --- 4. FastAPI 애플리케이션 초기 설정 ---
templates = Jinja2Templates(directory="template2")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"연산 장치: {DEVICE}")
    print("YOLOv8-OBB 모델을 로딩합니다...")
    app.state.model = YOLO(MODEL_PATH)
    
    # 로컬 웹캠 연결 로직
    print(f"{CAMERA_INDEX}번 카메라에 연결을 시도합니다...")
    # cv2.CAP_DSHOW는 Windows에서 카메라 초기화 속도를 높여주고 안정성을 더해줍니다.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) 
    
    if not cap.isOpened():
        print(f"{CAMERA_INDEX}번 카메라를 찾을 수 없습니다. 0번 카메라로 다시 시도합니다...")
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("오류: 사용 가능한 카메라를 찾을 수 없습니다.")
        app.state.cap = None
    else:
        print("카메라에 성공적으로 연결되었습니다.")
        app.state.cap = cap
    
    yield
    
    print("애플리케이션을 종료합니다.")
    if app.state.cap:
        app.state.cap.release()
        print("카메라 리소스가 해제되었습니다.")

app = FastAPI(lifespan=lifespan)

# --- 5. 보조 함수 정의 ---
def center_crop_square(frame: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x, start_y = (w - min_dim) // 2, (h - min_dim) // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def update_assembly_state(detected_classes: list):
    current_step = ASSEMBLY_STATE["current_step"]
    
    if current_step > 3:
        return ASSEMBLY_STATE["completion_message"]

    expected_class = ASSEMBLY_STATE["steps_info"][current_step]["class"]
    
    if expected_class in detected_classes:
        if current_step < 3:
            ASSEMBLY_STATE["current_step"] += 1
            next_step_info = ASSEMBLY_STATE["steps_info"][current_step + 1]
            return f"성공! 다음 단계: {next_step_info['message']}"
        else:
            ASSEMBLY_STATE["current_step"] += 1
            return ASSEMBLY_STATE["completion_message"]
    elif detected_classes:
        return ASSEMBLY_STATE["error_message"]
    else:
        return ASSEMBLY_STATE["steps_info"][current_step]["message"]

# --- 6. FastAPI 엔드포인트(API 경로) 정의 ---
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index4.html", {"request": request})

@app.post("/reset")
async def reset_state():
    ASSEMBLY_STATE["current_step"] = 1
    print("공정 상태가 1단계로 초기화되었습니다.")
    return {"message": "State reset successfully"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = websocket.app.state.cap
    model = websocket.app.state.model

    if not cap:
        await websocket.close(code=1011, reason="Camera not available")
        return

    try:
        frame_counter = 0
        while True:
            success, frame = cap.read()
            if not success:
                print("프레임 읽기 실패. 카메라 연결을 확인하세요.")
                break
            
            frame_counter += 1

            # 이미지 처리 및 추론
            cropped = center_crop_square(frame)
            resized = cv2.resize(cropped, (640, 640))
            results = model(resized, device=DEVICE, verbose=False, conf=CONF_THRESHOLD)

            # 클래스 이름 추출
            detected_classes = []
            result = results[0]
            
            if result.obb and len(result.obb) > 0:
                if frame_counter % 30 == 0:
                    print(f"[Frame {frame_counter}] OBBs found: {len(result.obb)}")
                
                class_indices_tensor = result.obb.cls
                if class_indices_tensor is not None:
                    class_indices = class_indices_tensor.cpu().numpy().astype(int).tolist()
                    
                    if frame_counter % 30 == 0:
                        print(f"  - 감지된 클래스 인덱스: {class_indices}")
                        print(f"  - 모델의 전체 클래스 이름: {result.names}")
                    
                    try:
                        detected_classes = sorted(list(set(result.names[i] for i in class_indices)))
                        if detected_classes and frame_counter % 30 == 0:
                            print(f"  - [성공] 최종 감지된 클래스 이름: {detected_classes}")
                    except Exception as e:
                        print(f"  - [오류] 클래스 이름 변환 중 문제 발생: {e}")
            
            # 상태 업데이트 및 데이터 전송
            status_message = update_assembly_state(detected_classes)
            annotated_frame = result.plot()
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_json({
                "image": jpg_as_text,
                "current_step": ASSEMBLY_STATE["current_step"],
                "message": status_message,
                "detected_classes": detected_classes
            })
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("클라이언트 연결이 끊어졌습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        await websocket.close()