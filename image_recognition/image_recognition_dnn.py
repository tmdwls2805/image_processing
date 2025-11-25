import cv2
import os
import numpy as np
import urllib.request

def download_model_files():
    """필요한 모델 파일 다운로드"""
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    
    # 모델 파일이 없으면 다운로드
    if not os.path.exists(model_file):
        print("모델 파일 다운로드 중...")
        url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        urllib.request.urlretrieve(url, model_file)
        print("모델 다운로드 완료")
    
    # 설정 파일이 없으면 다운로드
    if not os.path.exists(config_file):
        print("설정 파일 다운로드 중...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        urllib.request.urlretrieve(url, config_file)
        print("설정 다운로드 완료")
    
    return model_file, config_file

def detect_faces_dnn(image_path, confidence_threshold=0.5):
    """DNN을 사용한 얼굴 검출"""
    # 모델 파일 다운로드
    model_file, config_file = download_model_files()
    
    # DNN 모델 로드
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 읽을 수 없습니다.")
        return None, 0
    
    h, w = img.shape[:2]
    
    # 이미지 전처리
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    
    # 얼굴 검출
    net.setInput(blob)
    detections = net.forward()
    
    face_count = 0
    
    # 검출된 얼굴 처리
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            face_count += 1
            
            # 얼굴 위치 계산
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            # 범위 체크
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 사각형 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 신뢰도 표시
            text = f"{confidence:.2f}"
            cv2.putText(img, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    return img, face_count

def compare_methods(image_path):
    """Haar Cascade와 DNN 방법 비교"""
    import time
    
    img_original = cv2.imread(image_path)
    if img_original is None:
        print("이미지를 읽을 수 없습니다.")
        return
    
    # 1. Haar Cascade 방법
    print("\n=== Haar Cascade 방법 ===")
    start_time = time.time()
    
    img_haar = img_original.copy()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_haar, cv2.COLOR_BGR2GRAY)
    faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    for (x, y, w, h) in faces_haar:
        cv2.rectangle(img_haar, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    haar_time = time.time() - start_time
    print(f"검출된 얼굴 수: {len(faces_haar)}")
    print(f"처리 시간: {haar_time:.3f}초")
    
    # 2. DNN 방법
    print("\n=== DNN 방법 ===")
    start_time = time.time()
    
    img_dnn, face_count_dnn = detect_faces_dnn(image_path, confidence_threshold=0.5)
    
    dnn_time = time.time() - start_time
    print(f"검출된 얼굴 수: {face_count_dnn}")
    print(f"처리 시간: {dnn_time:.3f}초")
    
    # 결과 표시
    if img_dnn is not None:
        # 두 이미지를 나란히 표시
        combined = np.hstack([img_haar, img_dnn])
        
        # 크기 조정 (너무 크면 화면에 맞춤)
        height, width = combined.shape[:2]
        if width > 1400:
            scale = 1400 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            combined = cv2.resize(combined, (new_width, new_height))
        
        # 라벨 추가
        cv2.putText(combined, "Haar Cascade (Blue)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(combined, "DNN (Green)", (combined.shape[1]//2 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Face Detection Comparison", combined)
        print("\n아무 키나 누르면 창이 닫힙니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 메인 실행
if __name__ == "__main__":
    # 테스트할 이미지 경로들
    test_images = [
        # "C:/develop/vision-test/file/image1.jpg",
        # "C:/develop/vision-test/file/bg_image.webp",
        # "C:/develop/vision-test/file/old_image.webp",
        "C:/develop/vision-test/file/blur_image.webp"
    ]
    
    # 사용 가능한 이미지 찾기
    image_path = None
    for path in test_images:
        if os.path.exists(path):
            image_path = path
            print(f"테스트 이미지: {path}")
            break
    
    if image_path:
        compare_methods(image_path)
    else:
        print("이미지 파일을 찾을 수 없습니다.")