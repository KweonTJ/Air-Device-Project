# gesture_training.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# 데이터 경로 설정
data_path = "gesture_data/"  # 제스처 데이터셋 폴더 경로
categories = ["gesture_1", "gesture_2", "gesture_3"]  # 각 제스처 클래스 이름
img_size = 64
data = []
labels = []

# # 데이터 준비 함수
# def load_data():
#     data = []
#     labels = []
#     for i, category in enumerate(categories):
#         folder_path = os.path.join(data_path, category)
#         for img_name in os.listdir(folder_path):
#             try:
#                 img_path = os.path.join(folder_path, img_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 img = cv2.resize(img, (img_size, img_size))
#                 data.append(img)
#                 labels.append(i)
#             except Exception as e:
#                 pass
#     data = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
#     labels = np.array(labels)
#     return data, labels

# # 데이터 로드 및 모델 학습
# data, labels = load_data()

# 데이터 수집 함수
def collect_data(label):
    cap = cv2.VideoCapture(0)  # 기본 카메라 사용
    collected_samples = 0

    print(f"Press 's' to start capturing data for {categories[label]}. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # CUDA로 GPU에서 그레이스케일 및 리사이즈 작업
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # 그레이스케일 변환
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        
        # 이미지 리사이즈
        gpu_resized = cv2.cuda.resize(gpu_gray, (img_size, img_size))

        # GPU에서 CPU로 다운로드
        resized = gpu_resized.download()

        # 사용자 입력으로 데이터 수집 제어
        cv2.imshow("Capture Gesture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # 's'를 누르면 이미지 수집 시작
            data.append(resized)
            labels.append(label)
            collected_samples += 1
            print(f"Collected samples: {collected_samples}")
        elif key == ord("q"):  # 'q'를 누르면 데이터 수집 종료
            break

    cap.release()
    cv2.destroyAllWindows()

# 모든 제스처 데이터 수집
for i, category in enumerate(categories):
    collect_data(i)

# 데이터 전처리
data = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
labels = np.array(labels)


# 모델 설계
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(categories), activation="softmax")
])

# 모델 컴파일 및 학습
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(data, labels, epochs=10, validation_split=0.1)

# 모델 저장
model.save("gesture_model")  # TensorFlow SavedModel 형식으로 저장
print("Model trained and saved as 'gesture_model'")
