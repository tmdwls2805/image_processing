import cv2
import os
import numpy as np
import urllib.request
import time

def download_yunet_model():
    """YuNet 모델 다운로드"""
    model_file = "face_detection_yunet_2023mar.onnx"
    
    if not os.path.exists(model_file):
        print("YuNet 모델 다운로드 중...")
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        urllib.request.urlretrieve(url, model_file)
        print("YuNet 모델 다운로드 완료")
    
    return model_file

def detect_faces_yunet(image_path, score_threshold=0.7):
    """YuNet을 사용한 얼굴 검출 (최신 모델, 작은 얼굴도 잘 검출)"""
    model_file = download_yunet_model()
    
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 읽을 수 없습니다.")
        return None, 0
    
    h, w = img.shape[:2]
    
    # YuNet 모델 초기화
    detector = cv2.FaceDetectorYN.create(
        model_file,
        "",
        (w, h),
        score_threshold,
        0.3,  # NMS threshold
        5000  # top_k
    )
    
    # 얼굴 검출
    _, faces = detector.detect(img)
    
    face_count = 0
    img_result = img.copy()
    
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            confidence = face[14]
            
            face_count += 1
            
            # 얼굴 사각형
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # 신뢰도 표시
            text = f"YuNet: {confidence:.2f}"
            cv2.putText(img_result, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 랜드마크 표시 (눈, 코, 입)
            landmarks = face[4:14].reshape(5, 2).astype(int)
            for landmark in landmarks:
                cv2.circle(img_result, tuple(landmark), 2, (255, 0, 0), -1)
    
    return img_result, face_count

def detect_faces_dnn_improved(image_path, confidence_threshold=0.5):
    """개선된 DNN 얼굴 검출 (작은 이미지 최적화)"""
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    
    # 모델 파일 체크
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        print("DNN 모델 파일이 없습니다. image_recognition_dnn.py를 먼저 실행하세요.")
        return None, 0
    
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    
    img = cv2.imread(image_path)
    if img is None:
        return None, 0
    
    h, w = img.shape[:2]
    
    # 작은 이미지는 업스케일링
    scale = 1.0
    if w < 300 or h < 300:
        scale = max(300/w, 300/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_scaled = cv2.resize(img, (new_w, new_h))
    else:
        img_scaled = img
        new_w, new_h = w, h
    
    # 이미지 전처리
    blob = cv2.dnn.blobFromImage(img_scaled, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    face_count = 0
    img_result = img.copy()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            face_count += 1
            
            # 원본 이미지 크기로 변환
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            text = f"DNN: {confidence:.2f}"
            cv2.putText(img_result, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_result, face_count

def compare_all_methods(image_path):
    """모든 얼굴 검출 방법 비교"""
    
    print(f"\n테스트 이미지: {image_path}")
    
    # 이미지 정보 출력
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 읽을 수 없습니다.")
        return
    
    h, w = img.shape[:2]
    print(f"이미지 크기: {w}x{h}")
    print("-" * 50)
    
    results = []
    
    # 1. Haar Cascade
    print("\n1. Haar Cascade 방법")
    start_time = time.time()
    
    img_haar = img.copy()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_haar, cv2.COLOR_BGR2GRAY)
    faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    
    for (x, y, w, h) in faces_haar:
        cv2.rectangle(img_haar, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_haar, "Haar", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    haar_time = time.time() - start_time
    print(f"   검출된 얼굴: {len(faces_haar)}개")
    print(f"   처리 시간: {haar_time:.3f}초")
    results.append(("Haar Cascade", img_haar, len(faces_haar), haar_time))
    
    # 2. DNN (개선된 버전)
    print("\n2. DNN 방법 (개선된 버전)")
    start_time = time.time()
    
    img_dnn, face_count_dnn = detect_faces_dnn_improved(image_path, confidence_threshold=0.4)
    
    dnn_time = time.time() - start_time
    if img_dnn is not None:
        print(f"   검출된 얼굴: {face_count_dnn}개")
        print(f"   처리 시간: {dnn_time:.3f}초")
        results.append(("DNN Improved", img_dnn, face_count_dnn, dnn_time))
    
    # 3. YuNet (최신 모델)
    print("\n3. YuNet 방법 (2023년 모델)")
    start_time = time.time()
    
    img_yunet, face_count_yunet = detect_faces_yunet(image_path, score_threshold=0.6)
    
    yunet_time = time.time() - start_time
    if img_yunet is not None:
        print(f"   검출된 얼굴: {face_count_yunet}개")
        print(f"   처리 시간: {yunet_time:.3f}초")
        results.append(("YuNet", img_yunet, face_count_yunet, yunet_time))
    
    # 결과 표시
    print("\n" + "=" * 50)
    print("성능 요약:")
    print("-" * 50)
    
    best_accuracy = max(results, key=lambda x: x[2])
    best_speed = min(results, key=lambda x: x[3])
    
    print(f"가장 많은 얼굴 검출: {best_accuracy[0]} ({best_accuracy[2]}개)")
    print(f"가장 빠른 속도: {best_speed[0]} ({best_speed[3]:.3f}초)")
    
    # 이미지 표시
    if len(results) > 0:
        # 모든 결과를 하나의 이미지로 합치기
        images = [r[1] for r in results]
        
        # 가로로 배치
        if len(images) == 3:
            combined = np.hstack(images)
        elif len(images) == 2:
            combined = np.hstack(images)
        else:
            combined = images[0]
        
        # 크기 조정
        height, width = combined.shape[:2]
        max_width = 1600
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            combined = cv2.resize(combined, (new_width, new_height))
        
        cv2.imshow("Face Detection Comparison (Blue:Haar | Green:DNN | Yellow:YuNet)", combined)
        print("\n아무 키나 누르면 창이 닫힙니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 메인 실행
if __name__ == "__main__":
    # 테스트할 이미지 경로
    test_images = [
        "C:/develop/vision-test/file/blur_image.webp",
        # "C:/develop/vision-test/file/image1.jpg",
        # "C:/develop/vision-test/file/bg_image.webp",
        # "C:/develop/vision-test/file/old_image.webp"
    ]
    
    # 사용 가능한 이미지 찾기
    for path in test_images:
        if os.path.exists(path):
            compare_all_methods(path)
            break
    else:
        print("이미지 파일을 찾을 수 없습니다.")