from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import cv2

model = YOLO("best_2.pt")
img = Image.open("test2.jpg").convert("RGB")

results = model.predict(img)

# plot()은 OpenCV BGR 이미지 numpy 배열 반환
annotated_img = results[0].plot()

# BGR → RGB 변환
annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

# numpy 배열 → PIL 이미지 변환
annotated_pil_img = Image.fromarray(annotated_img)

# matplotlib로 화면에 띄우기
plt.imshow(annotated_pil_img)
plt.axis('off')  # 축 제거
plt.show()

# venv\Scripts\activate
# uvicorn server:app --reload --host 0.0.0.0 --port 8000