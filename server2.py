from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from contextlib import asynccontextmanager
import cv2
import threading
import time
from ultralytics import YOLO
import sqlite3
from collections import Counter
import json # JSON 응답을 위해 추가

# ... (기존 전역 변수들은 동일) ...
# 카메라 객체들을 전역 변수로 관리
cameras = {}
camera_locks = {}
yolo_models = {}
bom_data = []  # BOM 데이터를 전역 변수로 저장

# 카메라 전환 제어 변수
active_camera = 0  # 0: 0번 카메라 사용, 1: 1번 카메라 사용
camera_switch_lock = threading.Lock()  # 스레드 안전성을 위한 락

# 프로세스 단계 관리 변수 (확장된 단계)
process_step = "waiting_for_match"  # waiting_for_match, step1_remove_3, ..., completed
step_lock = threading.Lock()

# 0번 인덱스 초기 개수 추적
initial_count_0 = 0
expected_count_0_first = 0  # 첫 번째 0번 감소 후 예상 개수
expected_count_0_second = 0  # 두 번째 0번 감소 후 예상 개수

# 현재 감지된 객체 정보 저장
current_detections = {}
detections_lock = threading.Lock()

# 검출 개수 저장
detection_counts = {}
detection_counts_lock = threading.Lock()

# 프로세스 완료 여부 플래그 추가
is_process_completed_flag = False
completed_flag_lock = threading.Lock()

# 깜빡임 효과를 위한 변수
blink_frame_counter = 0
blink_state = True # True: 그리기, False: 그리지 않기
BLINK_INTERVAL = 10 # 10프레임마다 깜빡임 상태 변경 (약 0.33초)


# ... (get_data, load_bom_data 함수는 동일) ...
def get_data():
    """BOM 데이터베이스에서 데이터 조회"""
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

def load_bom_data():
    """BOM 데이터 로드 및 출력"""
    global bom_data
    try:
        bom_data = get_data()
        print("\n=== BOM 부품 정보 로드 완료 ===")
        for item in bom_data:
            if item["part_name"] and item["useage"]:
                print(f"부품명: {item['part_name']}, 사용량: {item['useage']}, 클래스번호: {item['cls_no']}")
        print("==============================\n")
    except Exception as e:
        print(f"BOM 데이터 조회 오류: {e}")
        bom_data = []

# ... (check_bom_match, update_current_detections 함수는 동일) ...
def check_bom_match(detected_classes):
    """검출된 클래스와 BOM 데이터 일치 여부 확인 및 검출 개수 저장"""
    global detection_counts
    
    # detected_classes를 정수로 변환
    detected_classes_int = [int(cls) for cls in detected_classes]
    detected_counts = Counter(detected_classes_int)
    
    # BOM에서 cls_no별 사용량 정보 추출
    bom_requirements = {}
    for item in bom_data:
        if item["cls_no"] is not None and item["useage"] is not None:
            bom_requirements[int(item["cls_no"])] = int(item["useage"])
    
    # detected_counts에 누락된 키들을 0으로 추가
    complete_detected_counts = {}
    for cls in bom_requirements.keys():
        complete_detected_counts[cls] = detected_counts.get(cls, 0)
    
    # 검출 개수를 전역 변수에 저장
    with detection_counts_lock:
        detection_counts = complete_detected_counts.copy()
        detection_counts["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n--- 객체 검출 결과 비교 ---")
    print(f"현재 검출된 인덱스: {detected_classes_int}")
    print(f"검출된 개수: {dict(complete_detected_counts)}")
    print(f"BOM 요구사항: {bom_requirements}")
    
    # 각 클래스별 상세 비교
    all_classes = set(complete_detected_counts.keys()) | set(bom_requirements.keys())
    for cls in sorted(all_classes):
        detected = complete_detected_counts.get(cls, 0)
        required = bom_requirements.get(cls, 0)
        status = "✅" if detected == required else "❌"
        print(f"  클래스 {cls}: 검출 {detected}개 / 필요 {required}개 {status}")
    
    # 완전 일치 확인
    if complete_detected_counts == bom_requirements:
        print("🎯 전체 결과: BOM 요구사항과 완전 일치!")
        print("---------------------------\n")
        return True
    else:
        print("⚠️  전체 결과: BOM 요구사항 불일치")
        print("---------------------------\n")
        return False

def update_current_detections(detected_classes, camera_id):
    """현재 감지된 객체 정보 업데이트"""
    global current_detections
    
    
    # 현재 활성 카메라의 감지 정보만 업데이트
    with camera_switch_lock:
        current_active = active_camera
    
    if camera_id == current_active:
        # detected_classes를 정수로 변환
        detected_classes_int = [int(cls) for cls in detected_classes]
        detected_counts = Counter(detected_classes_int)
        
        with detections_lock:
            # 모든 감지 정보 초기화 (기존 데이터 클리어)
            current_detections.clear()
            
            # BOM에 있는 모든 클래스에 대해 개수 업데이트
            for item in bom_data:
                if item["cls_no"] is not None:
                    cls_no = int(item["cls_no"])
                    current_detections[cls_no] = detected_counts.get(cls_no, 0)
            
            # 현재 시간 추가
            current_detections["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 디버깅 정보 출력
            print(f"[감지 업데이트] 카메라 {camera_id}: {dict(detected_counts)}")
    else:
        # 비활성 카메라인 경우에도 빈 상태로 업데이트
        with detections_lock:
            current_detections.clear()
            for item in bom_data:
                if item["cls_no"] is not None:
                    cls_no = int(item["cls_no"])
                    current_detections[cls_no] = 0
            current_detections["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

def check_process_step(detected_classes, camera_id):
    """프로세스 단계별 확인 (확장된 알고리즘)"""
    global process_step, active_camera, initial_count_0, expected_count_0_first, expected_count_0_second, is_process_completed_flag
    
    # 현재 감지된 객체 정보 업데이트
    update_current_detections(detected_classes, camera_id)
    
    # detected_classes를 정수로 변환
    detected_classes_int = [int(cls) for cls in detected_classes]
    detected_counts = Counter(detected_classes_int)
    
    # BOM 요구사항 추출
    bom_requirements = {}
    for item in bom_data:
        if item["cls_no"] is not None and item["useage"] is not None:
            bom_requirements[int(item["cls_no"])] = int(item["useage"])
    
    with step_lock:
        current_step = process_step

        # 프로세스가 완료된 상태이면 더 이상 단계를 진행하지 않습니다.
        if current_step == "completed":
            return False # 카메라 전환도 하지 않음
        
        if camera_id == 0:  # best_2 모델 (0번 카메라)
            if current_step == "waiting_for_match":
                # 초기 BOM 일치 확인
                if check_bom_match(detected_classes):
                    process_step = "step1_remove_3"
                    # 0번 인덱스 초기 개수 저장
                    initial_count_0 = detected_counts.get(0, 0)
                    expected_count_0_first = initial_count_0 - 1
                    expected_count_0_second = initial_count_0 - 2
                    print(f"🚀 1단계 시작: 3번 인덱스 제거를 기다립니다.")
                    print(f"📊 0번 인덱스 초기 개수: {initial_count_0}")
                    return False
                
            elif current_step == "step1_remove_3":
                # 3번 인덱스가 없어졌는지 확인
                expected_counts = bom_requirements.copy()
                expected_counts[3] = 0  # 3번은 없어져야 함
                
                complete_detected_counts = {}
                for cls in expected_counts.keys():
                    complete_detected_counts[cls] = detected_counts.get(cls, 0)
                
                print(f"[1단계] detected_classes_int: {detected_classes_int}")
                print(f"[1단계] 일치 여부: {complete_detected_counts == expected_counts}")
                
                if complete_detected_counts == expected_counts:
                    process_step = "step2_remove_2"
                    print("✅ 1단계 완료: 3번 인덱스 제거 확인")
                    print("🚀 2단계 시작: 2번 인덱스 제거를 기다립니다.")
                    return False
                elif 3 not in detected_counts or detected_counts[3] == 0:
                    print(f"⚠️ 경고: 3번 인덱스는 제거되었지만 다른 부품에 변화가 있습니다!")
                    return False
                else:
                    print(f"[1단계] 3번이 아직 남아있음: {detected_counts.get(3, 0)}개")
                    return False
                
            elif current_step == "step2_remove_2":
                # 2번 인덱스가 없어졌는지 확인
                expected_counts = bom_requirements.copy()
                expected_counts[3] = 0  # 3번은 이미 없음
                expected_counts[2] = 0  # 2번도 없어져야 함
                
                complete_detected_counts = {}
                for cls in expected_counts.keys():
                    complete_detected_counts[cls] = detected_counts.get(cls, 0)
                
                print(f"[2단계] detected_classes_int: {detected_classes_int}")
                print(f"[2단계] 일치 여부: {complete_detected_counts == expected_counts}")
                
                if complete_detected_counts == expected_counts:
                    process_step = "step3_check_3"
                    print("✅ 2단계 완료: 2번 인덱스 제거 확인")
                    print("🚀 3단계 시작: best3 모델에서 3번 인덱스 확인을 기다립니다.")
                    return True  # best3 모델로 전환
                elif 2 not in detected_counts or detected_counts[2] == 0:
                    print(f"⚠️ 경고: 2번 인덱스는 제거되었지만 다른 부품에 변화가 있습니다!")
                    return False
                else:
                    print(f"[2단계] 2번이 아직 남아있음: {detected_counts.get(2, 0)}개")
                    return False
                
            elif current_step == "step4_check_0_first":
                # 0번 인덱스가 한 개 줄었는지 확인
                current_count_0 = detected_counts.get(0, 0)
                print(f"[4단계] 0번 인덱스 현재 개수: {current_count_0}, 예상 개수: {expected_count_0_first}")
                
                if current_count_0 == expected_count_0_first:
                    process_step = "step5_check_4"
                    print("✅ 4단계 완료: 0번 인덱스 1개 감소 확인")
                    print("🚀 5단계 시작: best3 모델에서 4번 인덱스 확인을 기다립니다.")
                    return True  # best3 모델로 전환
                else:
                    print(f"⚠️ 4단계 대기: 0번 인덱스가 {expected_count_0_first}개가 되기를 기다립니다. (현재: {current_count_0}개)")
                    return False
                
            elif current_step == "step6_check_0_second":
                # 0번 인덱스가 한 개 더 줄었는지 확인 (화면에 아무것도 없어야 함)
                current_count_0 = detected_counts.get(0, 0)
                total_detected = len(detected_classes_int)  # 전체 검출된 객체 수
                
                print(f"[6단계] 0번 인덱스 현재 개수: {current_count_0}, 예상 개수: {expected_count_0_second}")
                print(f"[6단계] 전체 검출된 객체 수: {total_detected}")
                print(f"[6단계] 검출된 인덱스: {detected_classes_int}")
                
                # 0번 인덱스가 예상 개수와 일치하고, 추가로 화면에 아무것도 없는지 확인
                if current_count_0 == expected_count_0_second:
                    if expected_count_0_second == 0:  # 0번 인덱스가 완전히 없어져야 하는 경우
                        if total_detected == 0:  # 화면에 아무것도 없음
                            process_step = "step7_check_5"
                            print("✅ 6단계 완료: 화면에 아무것도 없음 확인 (0번 인덱스 완전 제거)")
                            print("🚀 7단계 시작: best3 모델에서 5번 인덱스 확인을 기다립니다.")
                            return True  # best3 모델로 전환
                        else:
                            print(f"⚠️ 6단계 대기: 0번 인덱스는 없지만 다른 객체가 감지됩니다. (감지된 객체: {detected_classes_int})")
                            return False
                    else:  # 0번 인덱스가 일부 남아있어야 하는 경우
                        # 0번 인덱스 외에 다른 인덱스는 없어야 함
                        other_indices = [idx for idx in detected_classes_int if idx != 0]
                        if len(other_indices) == 0:
                            process_step = "step7_check_5"
                            print("✅ 6단계 완료: 0번 인덱스만 남아있음 확인")
                            print("🚀 7단계 시작: best3 모델에서 5번 인덱스 확인을 기다립니다.")
                            return True  # best3 모델로 전환
                        else:
                            print(f"⚠️ 6단계 대기: 0번 인덱스 외에 다른 객체가 감지됩니다. (다른 객체: {other_indices})")
                            return False
                else:
                    print(f"⚠️ 6단계 대기: 0번 인덱스가 {expected_count_0_second}개가 되기를 기다립니다. (현재: {current_count_0}개)")
                    return False
                
        elif camera_id == 1:  # best3 모델 (1번 카메라)
            if current_step == "step3_check_3":
                # 3번 인덱스가 인식되는지 확인
                if 3 in detected_counts and detected_counts[3] > 0:
                    process_step = "step4_check_0_first"
                    print("✅ 3단계 완료: 3번 인덱스 감지!")
                    print("🚀 4단계 시작: best_2 모델에서 0번 인덱스 1개 감소 확인을 기다립니다.")
                    return True  # best_2 모델로 전환
                elif len(detected_classes) > 0:
                    detected_indexes = list(set(detected_classes))
                    print(f"⚠️ 3단계 대기: 3번 인덱스를 기다립니다. (현재 감지: {detected_indexes})")
                    return False
                
            elif current_step == "step5_check_4":
                # 4번 인덱스가 인식되는지 확인
                if 4 in detected_counts and detected_counts[4] > 0:
                    process_step = "step6_check_0_second"
                    print("✅ 5단계 완료: 4번 인덱스 감지!")
                    print("🚀 6단계 시작: best_2 모델에서 0번 인덱스 추가 1개 감소 확인을 기다립니다.")
                    return True  # best_2 모델로 전환
                elif len(detected_classes) > 0:
                    detected_indexes = list(set(detected_classes))
                    print(f"⚠️ 5단계 대기: 4번 인덱스를 기다립니다. (현재 감지: {detected_indexes})")
                    return False
                
            elif current_step == "step7_check_5":
                # 5번 인덱스가 인식되는지 확인
                if 5 in detected_counts and detected_counts[5] > 0:
                    process_step = "completed"
                    # is_process_completed_flag를 True로 설정
                    with completed_flag_lock:
                        is_process_completed_flag = True
                    print("🎉🎉🎉 제품 완성! 🎉🎉🎉")
                    print("✅ 7단계 완료: 5번 인덱스 감지!")
                    print("--- 초기화 대기 중 (사용자 확인 필요) ---\n")
                    # 여기서는 바로 return False; 카메라 전환을 하지 않고 현재 카메라에 머물게 됩니다.
                    # 프론트엔드에서 /reset_process 호출이 있을 때까지 이 상태를 유지합니다.
                    return False 
                elif len(detected_classes) > 0:
                    detected_indexes = list(set(detected_classes))
                    print(f"⚠️ 7단계 대기: 5번 인덱스를 기다립니다. (현재 감지: {detected_indexes})")
                    return False
        
        return False

# ... (initialize_cameras, initialize_yolo_models, cleanup_cameras, crop_center_square 함수는 동일) ...
def initialize_cameras():
    """카메라 초기화"""
    for cam_id in [0, 1]:
        try:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                # 카메라 설정
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cameras[cam_id] = cap
                camera_locks[cam_id] = threading.Lock()
                print(f"카메라 {cam_id} 초기화 성공")
            else:
                print(f"카메라 {cam_id} 초기화 실패")
        except Exception as e:
            print(f"카메라 {cam_id} 오류: {e}")

def initialize_yolo_models():
    """YOLO 모델 초기화"""
    try:
        yolo_models['best_2'] = YOLO('best_2.pt')
        print("YOLO 모델(best_2.pt) 로드 성공")
    except Exception as e:
        print(f"best_2.pt 모델 로드 실패: {e}")
    
    try:
        yolo_models['best3'] = YOLO('best3.pt')
        print("YOLO 모델(best3.pt) 로드 성공")
    except Exception as e:
        print(f"best3.pt 모델 로드 실패: {e}")

def cleanup_cameras():
    """카메라 리소스 해제"""
    for cam_id, cap in cameras.items():
        cap.release()
    cv2.destroyAllWindows()

def crop_center_square(frame, target_size=640):
    """이미지를 중앙에서 정사각형으로 크롭하고 리사이즈"""
    h, w = frame.shape[:2]
    
    # 더 작은 차원을 기준으로 정사각형 크기 결정
    min_dim = min(h, w)
    
    # 중앙 좌표 계산
    center_x, center_y = w // 2, h // 2
    
    # 크롭 영역 계산
    half_size = min_dim // 2
    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size
    
    # 크롭 실행
    cropped = frame[y1:y2, x1:x2]
    
    # 640x640으로 리사이즈
    resized = cv2.resize(cropped, (target_size, target_size))
    
    return resized

def run_yolo_inference(frame, camera_id):
    """YOLO OBB 모델로 추론 실행 및 결과 시각화"""
    global active_camera, is_process_completed_flag, blink_frame_counter, blink_state, now

    with completed_flag_lock:
        if is_process_completed_flag:
            return frame

    with camera_switch_lock:
        current_active = active_camera

    if camera_id != current_active:
        return frame

    blink_frame_counter += 1
    if blink_frame_counter % BLINK_INTERVAL == 0:
        blink_state = not blink_state
        blink_frame_counter = 0

    model_name = None
    if camera_id == 0:
        model_name = 'best_2'
    elif camera_id == 1:
        model_name = 'best3'

    if model_name not in yolo_models:
        print(f"오류: {model_name} 모델을 찾을 수 없습니다.")
        return frame

    detected_classes = []
    try:
        results = yolo_models[model_name](frame, conf=0.7, verbose=False)
        class_names = yolo_models[model_name].names  # {0: 'bolt', 1: 'nut', ...}
        annotated_frame = frame.copy()

        for r in results:
            if hasattr(r, 'obb') and r.obb is not None:
                boxes = r.obb.xyxy.cpu().numpy()
                classes = r.obb.cls.cpu().numpy().astype(int)
                confidences = r.obb.conf.cpu().numpy()
            elif hasattr(r, 'boxes') and r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                confidences = r.boxes.conf.cpu().numpy()
            else:
                continue

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                cls = classes[i]
                conf = confidences[i]

                detected_classes.append(cls)

                should_blink = False
                blink_class = -1

                with step_lock:
                    current_step = process_step
                    if camera_id == 0:
                        if current_step == "step1_remove_3":
                            blink_class = 3
                        elif current_step == "step2_remove_2":
                            blink_class = 2
                        elif current_step == "step4_check_0_first":
                            blink_class = 0
                        elif current_step == "step6_check_0_second":
                            blink_class = 0
                    elif camera_id == 1:
                        if current_step == "step3_check_3":
                            blink_class = 3
                        elif current_step == "step5_check_4":
                            blink_class = 4
                        elif current_step == "step7_check_5":
                            blink_class = 5
                    now = class_names[cls]
                    if cls == blink_class and not blink_state:
                        should_blink = True

                if not should_blink:
                    # 깜빡임 대상이면 노란색에 두껍게
                    if cls == blink_class:
                        thickness = 6
                        color = (0, 255, 255)  # 노란색 (BGR)
                    else:
                        thickness = 2
                        # 기본 색상
                        if cls == 3:
                            color = (0, 0, 255)
                        elif cls == 2:
                            color = (255, 0, 0)
                        elif cls == 0:
                            color = (255, 255, 0)
                        elif cls == 4:
                            color = (0, 255, 255)
                        elif cls == 5:
                            color = (255, 0, 255)
                        else:
                            color = (0, 255, 0)  # 기본 초록

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # 클래스 이름과 신뢰도 표시
                    text = class_names[cls]
                    if(class_names[cls] == "불량1"):
                        text = "NG1"
                    elif(class_names[cls] == "불량2"):
                        text = "NG2"
                    elif(class_names[cls] == "불량3"):
                        text = "NG3"    
                    elif(class_names[cls] == "조립1"):
                        text = "Assemble1"
                    elif(class_names[cls] == "조립2"):
                        text = "Assemble1"
                    elif(class_names[cls] == "조립3"):
                        text = "NG6"

                    label = f"{text} {conf:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5),
                                  (x1 + text_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        print(f"도이 : {class_names[cls]}")
        if check_process_step(detected_classes, camera_id):
            with camera_switch_lock:
                active_camera = 1 if camera_id == 0 else 0
            print(f"🔄 카메라 {camera_id}에서 {active_camera}로 전환")

        return annotated_frame

    except Exception as e:
        print(f"{model_name} 모델 추론 오류 (카메라 {camera_id}): {e}")
        import traceback
        traceback.print_exc()
        return frame


# ... (generate_frames, lifespan, app 라우팅은 동일) ...
def generate_frames(camera_id):
    """특정 카메라의 프레임을 생성하는 제너레이터"""
    if camera_id not in cameras:
        return
    
    while True:
        try:
            with camera_locks[camera_id]:
                success, frame = cameras[camera_id].read()
            
            if not success:
                # 카메라가 끊겼을 때 재시도 로직 추가 (선택 사항)
                print(f"카메라 {camera_id} 연결 끊김, 재시도...")
                time.sleep(1)
                continue # 다음 루프에서 다시 시도
            
            # 중앙 크롭 및 640x640 리사이즈
            processed_frame = crop_center_square(frame, 640)
            
            # YOLO 추론 실행
            yolo_frame = run_yolo_inference(processed_frame, camera_id)
            
            # 프레임을 JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', yolo_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # 멀티파트 스트림 형식으로 반환
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # 약 30fps
            
        except Exception as e:
            print(f"카메라 {camera_id} 프레임 생성 오류: {e}")
            break

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작시 실행
    load_bom_data()  # BOM 데이터 로드 및 출력
    initialize_cameras()
    initialize_yolo_models()
    yield
    # 종료시 실행
    cleanup_cameras()

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def index():
    """메인 페이지 - index3.html 파일을 서빙"""
    try:
        with open("index_server.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return HTMLResponse("index3.html 파일을 찾을 수 없습니다.", status_code=404)

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: int):
    """특정 카메라의 비디오 스트림을 제공"""
    if camera_id not in cameras:
        return Response("카메라를 찾을 수 없습니다", status_code=404)
    
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/process_status")
async def process_status():
    """현재 프로세스 단계 상태 확인"""
    global initial_count_0, expected_count_0_first, expected_count_0_second, is_process_completed_flag
    
    with step_lock:
        current_step = process_step
    
    with completed_flag_lock:
        completed = is_process_completed_flag

    step_descriptions = {
        "waiting_for_match": "BOM 요구사항 일치 대기 중",
        "step1_remove_3": "1단계: 3번 인덱스 제거 대기 중",
        "step2_remove_2": "2단계: 2번 인덱스 제거 대기 중",
        "step3_check_3": "3단계: best3 모델에서 3번 인덱스 감지 대기 중",
        "step4_check_0_first": f"4단계: 0번 인덱스 1개 감소 대기 중 (목표: {expected_count_0_first}개)",
        "step5_check_4": "5단계: best3 모델에서 4번 인덱스 감지 대기 중",
        "step6_check_0_second": f"6단계: 0번 인덱스 추가 1개 감소 대기 중 (목표: {expected_count_0_second}개)",
        "step7_check_5": "7단계: best3 모델에서 5번 인덱스 감지 대기 중",
        "completed": "완료: 제품 완성!"
    }
    
    return {
        "current_step": current_step,
        "description": step_descriptions.get(current_step, "알 수 없는 단계"),
        "active_camera": active_camera,
        "initial_count_0": initial_count_0,
        "expected_count_0_first": expected_count_0_first,
        "expected_count_0_second": expected_count_0_second,
        "is_completed": completed # 새로운 필드 추가
    }

@app.get("/bom_data")
async def get_bom_data():
    """현재 로드된 BOM 데이터 반환"""
    return {"bom_data": bom_data, "total_parts": len(bom_data)}

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

@app.get("/current_detections")
async def get_current_detections():
    """현재 활성 카메라에서 감지된 객체 정보 반환"""
    try:
        with detections_lock:
            detections_copy = current_detections.copy()
        
        # timestamp 제거하여 별도로 반환
        timestamp = detections_copy.pop("timestamp", None)
        
        with camera_switch_lock:
            current_camera = active_camera
        
        # 디버깅 정보 출력
        print(f"[API] 현재 감지 정보 요청: {detections_copy}")
        
        return {
            "detections": detections_copy,
            "active_camera": current_camera,
            "timestamp": timestamp,
            "total_detected": sum(detections_copy.values()) if detections_copy else 0
        }
    except Exception as e:
        print(f"감지 정보 API 오류: {e}")
        return {
            "detections": {},
            "active_camera": 0,
            "timestamp": None,
            "total_detected": 0
        }

@app.get("/detection_counts")
async def get_detection_counts():
    """현재 검출된 개수 반환"""
    try:
        with detection_counts_lock:
            counts_copy = detection_counts.copy()
        
        # timestamp 제거하여 별도로 반환
        timestamp = counts_copy.pop("timestamp", None)
        
        return {
            "counts": counts_copy,
            "timestamp": timestamp,
            "total": sum(counts_copy.values()) if counts_copy else 0
        }
    except Exception as e:
        print(f"검출 개수 API 오류: {e}")
        return {
            "counts": {},
            "timestamp": None,
            "total": 0
        }

@app.get("/camera_switch_status")
async def camera_switch_status():
    """현재 활성 카메라 및 모델 상태 확인"""
    with camera_switch_lock:
        current_camera = active_camera
    
    model_info = {
        0: {"model": "best_2", "description": "0번 카메라에서 best_2 모델 실행 중"},
        1: {"model": "best3", "description": "1번 카메라에서 best3 모델 실행 중"}
    }
    
    return {
        "active_camera": current_camera,
        "current_model": model_info[current_camera]["model"],
        "description": model_info[current_camera]["description"]
    }

@app.get("/camera_status")
async def camera_status():
    """카메라 상태 확인"""
    status = {}
    for cam_id in [0, 1]:
        if cam_id in cameras:
            status[f"camera_{cam_id}"] = "연결됨" if cameras[cam_id].isOpened() else "연결 끊김"
        else:
            status[f"camera_{cam_id}"] = "초기화되지 않음"
    return status

@app.get("/full_status")
async def get_full_status():
    """모든 상태 정보를 한 번에 반환하는 통합 API"""
    try:
        # 프로세스 상태
        with step_lock:
            current_step = process_step
        
        # 현재 감지 정보
        with detections_lock:
            detections_copy = current_detections.copy()
        timestamp = detections_copy.pop("timestamp", None)
        
        # 검출 개수
        with detection_counts_lock:
            counts_copy = detection_counts.copy()
        counts_timestamp = counts_copy.pop("timestamp", None)
        
        # 활성 카메라
        with camera_switch_lock:
            current_camera = active_camera

        # 프로세스 완료 플래그
        with completed_flag_lock:
            completed_flag = is_process_completed_flag
        
        step_descriptions = {
            "waiting_for_match": "BOM 요구사항 일치 대기 중",
            "step1_remove_3": "1단계: 3번 인덱스 제거 대기 중",
            "step2_remove_2": "2단계: 2번 인덱스 제거 대기 중",
            "step3_check_3": "3단계: best3 모델에서 3번 인덱스 감지 대기 중",
            "step4_check_0_first": f"4단계: 0번 인덱스 1개 감소 대기 중 (목표: {expected_count_0_first}개)",
            "step5_check_4": "5단계: best3 모델에서 4번 인덱스 감지 대기 중",
            "step6_check_0_second": f"6단계: 0번 인덱스 추가 1개 감소 대기 중 (목표: {expected_count_0_second}개)",
            "step7_check_5": "7단계: best3 모델에서 5번 인덱스 감지 대기 중",
            "completed": "완료: 제품 완성!"
        }
        
        return {
            "process_status": {
                "current_step": current_step,
                "description": step_descriptions.get(current_step, "알 수 없는 단계"),
                "initial_count_0": initial_count_0,
                "expected_count_0_first": expected_count_0_first,
                "expected_count_0_second": expected_count_0_second,
                "is_completed": completed_flag # 새로운 필드 추가
            },
            "current_detections": {
                "detections": detections_copy,
                "timestamp": timestamp,
                "total_detected": sum(detections_copy.values()) if detections_copy else 0
            },
            "detection_counts": {
                "counts": counts_copy,
                "timestamp": counts_timestamp,
                "total": sum(counts_copy.values()) if counts_copy else 0
            },
            "camera_status": {
                "active_camera": current_camera,
                "current_model": "best_2" if current_camera == 0 else "best3"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "now" : now
        }
    except Exception as e:
        print(f"통합 상태 API 오류: {e}")
        return {
            "error": f"상태 정보 조회 중 오류 발생: {str(e)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

@app.post("/reset_process") # POST 요청으로 변경하여 명확성 높임
async def reset_process():
    """프로세스 상태를 초기화하고 다음 작업을 시작할 준비를 함"""
    global process_step, initial_count_0, expected_count_0_first, expected_count_0_second, active_camera, is_process_completed_flag
    
    with step_lock:
        process_step = "waiting_for_match"
        print("🔄 프로세스 상태 초기화: waiting_for_match")

    with completed_flag_lock:
        is_process_completed_flag = False
        print("✅ 완료 플래그 초기화")
    
    # 0번 인덱스 관련 값들도 초기화
    initial_count_0 = 0
    expected_count_0_first = 0
    expected_count_0_second = 0
    
    # 카메라를 0번으로 초기화 (필요에 따라 0번이 아니라 시작할 카메라로 설정)
    with camera_switch_lock:
        active_camera = 0 
    print("🔄 활성 카메라 0번으로 초기화")

    # 감지 정보도 초기화 (필요하다면)
    with detections_lock:
        current_detections.clear()
        for item in bom_data: # BOM 데이터를 기반으로 0으로 초기화
            if item["cls_no"] is not None:
                current_detections[int(item["cls_no"])] = 0
        current_detections["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with detection_counts_lock:
        detection_counts.clear()
        detection_counts["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return {"message": "프로세스 상태가 초기화되었습니다. 새로운 작업을 시작할 준비가 되었습니다."}

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)