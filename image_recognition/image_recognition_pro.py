import cv2
import os
import numpy as np
import urllib.request
import time

def download_yunet_model():
    """YuNet 모델 다운로드 (2023년 버전)"""
    model_file = "face_detection_yunet_2023mar.onnx"
    
    if not os.path.exists(model_file):
        print("YuNet 모델 다운로드 중...")
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        urllib.request.urlretrieve(url, model_file)
        print("YuNet 모델 다운로드 완료")
    
    return model_file

def download_yunet2022_model():
    """YuNet 2022년 모델 다운로드 (더 안정적)"""
    model_file = "face_detection_yunet_2022oct.onnx"
    
    if not os.path.exists(model_file):
        print("YuNet 2022 모델 다운로드 중...")
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2022oct.onnx"
        try:
            urllib.request.urlretrieve(url, model_file)
            print("YuNet 2022 모델 다운로드 완료")
        except Exception as e:
            print(f"모델 다운로드 실패: {e}")
            return None
    
    return model_file

def download_sface_model():
    """SFace 모델 다운로드 (얼굴 인식용)"""
    model_file = "face_recognition_sface_2021dec.onnx"
    
    if not os.path.exists(model_file):
        print("SFace 모델 다운로드 중...")
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        try:
            urllib.request.urlretrieve(url, model_file)
            print("SFace 모델 다운로드 완료")
        except Exception as e:
            print(f"SFace 모델 다운로드 실패: {e}")
            return None
    
    return model_file

def detect_faces_opencv_dnn_caffe_fallback(image_path, confidence_threshold=0.5):
    """Caffe 모델을 사용한 대안 DNN 얼굴 검출"""
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    
    # Caffe 모델 다운로드
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        print("Caffe 얼굴 검출 모델 다운로드 중...")
        try:
            # Caffe 모델 URL들
            caffe_model_urls = [
                "https://github.com/opencv/opencv_3rdparty/raw/19512b876f505face22c4b3e3c4e5cb7a7ddd2eb/res10_300x300_ssd_iter_140000.caffemodel",
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/19512b876f505face22c4b3e3c4e5cb7a7ddd2eb/res10_300x300_ssd_iter_140000.caffemodel"
            ]
            caffe_config_urls = [
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt"
            ]
            
            # Caffe 모델 다운로드
            model_downloaded = False
            for url in caffe_model_urls:
                try:
                    urllib.request.urlretrieve(url, model_file)
                    print(f"Caffe 모델 다운로드 성공: {url}")
                    model_downloaded = True
                    break
                except:
                    continue
            
            # Caffe 설정 파일 다운로드
            config_downloaded = False  
            for url in caffe_config_urls:
                try:
                    urllib.request.urlretrieve(url, config_file)
                    print(f"Caffe 설정 파일 다운로드 성공: {url}")
                    config_downloaded = True
                    break
                except:
                    continue
            
            if not model_downloaded or not config_downloaded:
                print("모든 모델 다운로드 실패")
                return None, 0
                
        except Exception as e:
            print(f"Caffe 모델 다운로드 실패: {e}")
            return None, 0
    
    # Caffe 모델로 얼굴 검출
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, 0
        
        h, w = img.shape[:2]
        img_result = img.copy()
        
        # Caffe 네트워크 로드
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        
        # 모델명 표시 (왼쪽 위)
        cv2.putText(img_result, "Caffe-DNN", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 128), 2)
        
        # 이미지 전처리
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        face_count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                face_count += 1
                
                # 좌표 계산
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # 경계 확인
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # 얼굴 사각형 그리기 (보라색 - Caffe DNN)
                cv2.rectangle(img_result, (x1, y1), (x2, y2), (255, 0, 128), 2)
                
                # 신뢰도 표시
                text = f"Caffe-DNN: {confidence:.2f}"
                cv2.putText(img_result, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 128), 2)
        
        return img_result, face_count
        
    except Exception as e:
        print(f"Caffe 모델 실행 오류: {e}")
        return None, 0

def detect_faces_opencv_dnn_tf(image_path, confidence_threshold=0.4):
    """OpenCV DNN with TensorFlow 모델 (더 정확한 검출)"""
    model_file = "opencv_face_detector_uint8.pb"
    config_file = "opencv_face_detector.pbtxt"
    
    # 모델 다운로드
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        print("TensorFlow 얼굴 검출 모델 다운로드 중...")
        try:
            # 더 안정적인 미러 사이트 사용
            model_urls = [
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb",
                "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/opencv_face_detector_uint8.pb"
            ]
            config_urls = [
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector.pbtxt",
                "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/opencv_face_detector.pbtxt"
            ]
            
            # 모델 파일 다운로드
            model_downloaded = False
            for url in model_urls:
                try:
                    urllib.request.urlretrieve(url, model_file)
                    print(f"모델 파일 다운로드 성공: {url}")
                    model_downloaded = True
                    break
                except:
                    continue
            
            # 설정 파일 다운로드
            config_downloaded = False
            for url in config_urls:
                try:
                    urllib.request.urlretrieve(url, config_file)
                    print(f"설정 파일 다운로드 성공: {url}")
                    config_downloaded = True
                    break
                except:
                    continue
            
            if not model_downloaded or not config_downloaded:
                print("모델 다운로드 실패 - 대안 방법 시도 중...")
                return detect_faces_opencv_dnn_caffe_fallback(image_path, confidence_threshold)
            
            print("TensorFlow 모델 다운로드 완료")
        except Exception as e:
            print(f"모델 다운로드 실패: {e}")
            return detect_faces_opencv_dnn_caffe_fallback(image_path, confidence_threshold)
    
    img = cv2.imread(image_path)
    if img is None:
        return None, 0
    
    h, w = img.shape[:2]
    img_result = img.copy()
    
    # DNN 네트워크 로드
    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    
    # 모델명 표시 (왼쪽 위)
    cv2.putText(img_result, "TF-DNN", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # 이미지 전처리
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            face_count += 1
            
            # 좌표 계산
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # 경계 확인
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 얼굴 사각형 그리기 (빨간색 - TensorFlow)
            cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 신뢰도 표시
            text = f"TF-DNN: {confidence:.2f}"
            cv2.putText(img_result, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img_result, face_count

def detect_faces_yunet_multiple_scales(image_path, score_threshold=0.5):
    """YuNet 다중 스케일 검출 (작은 얼굴까지 검출)"""
    model_file = download_yunet_model()
    if model_file is None:
        return None, 0
    
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 읽을 수 없습니다.")
        return None, 0
    
    h, w = img.shape[:2]
    img_result = img.copy()
    
    all_faces = []
    
    # 여러 스케일로 검출
    scales = [1.0, 1.2, 0.8]  # 원본, 확대, 축소
    
    for scale in scales:
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        if scaled_w < 100 or scaled_h < 100:
            continue
            
        # 이미지 스케일링
        if scale != 1.0:
            img_scaled = cv2.resize(img, (scaled_w, scaled_h))
        else:
            img_scaled = img
        
        # YuNet 모델 초기화
        detector = cv2.FaceDetectorYN.create(
            model_file,
            "",
            (scaled_w, scaled_h),
            score_threshold * (1.0 + (1.0 - scale) * 0.2),  # 스케일에 따라 threshold 조정
            0.3,
            5000
        )
        
        # 얼굴 검출
        _, faces = detector.detect(img_scaled)
        
        if faces is not None:
            for face in faces:
                # 원본 이미지 크기로 좌표 변환
                x, y, fw, fh = (face[:4] / scale).astype(int)
                confidence = face[14]
                
                # 중복 제거를 위한 면적 계산
                face_info = {
                    'x': x, 'y': y, 'w': fw, 'h': fh,
                    'confidence': confidence,
                    'landmarks': (face[4:14] / scale).reshape(5, 2).astype(int)
                }
                
                all_faces.append(face_info)
    
    # 모델명 표시 (왼쪽 위)
    cv2.putText(img_result, "YuNet-MS", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # NMS로 중복 제거
    if len(all_faces) > 0:
        boxes = np.array([[f['x'], f['y'], f['x']+f['w'], f['y']+f['h']] for f in all_faces])
        scores = np.array([f['confidence'] for f in all_faces])
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold, 0.4)
        
        face_count = 0
        if len(indices) > 0:
            for i in indices.flatten():
                face = all_faces[i]
                face_count += 1
                
                x, y, fw, fh = face['x'], face['y'], face['w'], face['h']
                confidence = face['confidence']
                
                # 얼굴 사각형 (노란색 - YuNet MultiScale)
                cv2.rectangle(img_result, (x, y), (x+fw, y+fh), (0, 255, 255), 2)
                
                # 신뢰도 표시
                text = f"YuNet-MS: {confidence:.2f}"
                cv2.putText(img_result, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # 랜드마크
                for landmark in face['landmarks']:
                    cv2.circle(img_result, tuple(landmark), 2, (255, 0, 0), -1)
    else:
        face_count = 0
    
    return img_result, face_count

def detect_faces_yunet_pro(image_path, score_threshold=0.5):
    """YuNet을 사용한 고급 얼굴 검출 (다양한 각도와 표정 검출)"""
    model_file = download_yunet_model()
    
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 읽을 수 없습니다.")
        return None, 0
    
    h, w = img.shape[:2]
    img_result = img.copy()
    
    # 모델명 표시 (왼쪽 위) - 더 크고 두껍게
    cv2.putText(img_result, "YuNet-Pro", (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    # YuNet 모델 초기화 (더 낮은 threshold로 더 많은 얼굴 검출)
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
    
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            confidence = face[14]
            
            face_count += 1
            
            # 얼굴 사각형 (파란색 - YuNet Pro)
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 신뢰도 표시
            text = f"YuNet-Pro: {confidence:.2f}"
            cv2.putText(img_result, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 랜드마크 표시 (눈, 코, 입)
            landmarks = face[4:14].reshape(5, 2).astype(int)
            for landmark in landmarks:
                cv2.circle(img_result, tuple(landmark), 2, (255, 0, 0), -1)
    
    return img_result, face_count

def detect_faces_cascade_lbp(image_path):
    """LBP Cascade를 사용한 빠른 얼굴 검출"""
    # LBP cascade 파일 다운로드
    lbp_file = "lbpcascade_frontalface_improved.xml"
    
    if not os.path.exists(lbp_file):
        print("LBP Cascade 모델 다운로드 중...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_frontalface_improved.xml"
        urllib.request.urlretrieve(url, lbp_file)
        print("LBP Cascade 모델 다운로드 완료")
    
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 읽을 수 없습니다.")
        return None, 0
    
    img_result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # LBP cascade 로드
    lbp_cascade = cv2.CascadeClassifier(lbp_file)
    
    # 모델명 표시 (왼쪽 위)
    cv2.putText(img_result, "LBP-Cascade", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # 얼굴 검출
    faces = lbp_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(20, 20)
    )
    
    face_count = len(faces)
    
    for (x, y, w, h) in faces:
        # 얼굴 사각형 (초록색 - LBP Cascade)
        cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_result, "LBP-Cascade: ", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_result, face_count

def compare_all_pro_methods(image_path):
    """모든 고급 얼굴 검출 방법 비교"""
    
    print(f"\n테스트 이미지: {image_path}")
    
    # 이미지 정보 출력
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 읽을 수 없습니다.")
        return
    
    h, w = img.shape[:2]
    print(f"이미지 크기: {w}x{h}")
    print("-" * 60)
    
    results = []
    
    # 1. OpenCV DNN with TensorFlow (더 정확한 검출)
    print("\n1. OpenCV DNN-TensorFlow 방법")
    start_time = time.time()
    try:
        img_tf, face_count_tf = detect_faces_opencv_dnn_tf(image_path, confidence_threshold=0.4)
        tf_time = time.time() - start_time
        if img_tf is not None:
            print(f"   검출된 얼굴: {face_count_tf}개")
            print(f"   처리 시간: {tf_time:.3f}초")
            results.append(("TF-DNN", img_tf, face_count_tf, tf_time))
    except Exception as e:
        print(f"   TF-DNN 오류: {e}")
    
    # 2. YuNet 다중 스케일 (작은 얼굴까지 검출)
    print("\n2. YuNet 다중 스케일 방법")
    start_time = time.time()
    try:
        img_yunet_ms, face_count_yunet_ms = detect_faces_yunet_multiple_scales(image_path, score_threshold=0.5)
        yunet_ms_time = time.time() - start_time
        if img_yunet_ms is not None:
            print(f"   검출된 얼굴: {face_count_yunet_ms}개")
            print(f"   처리 시간: {yunet_ms_time:.3f}초")
            results.append(("YuNet-MultiScale", img_yunet_ms, face_count_yunet_ms, yunet_ms_time))
    except Exception as e:
        print(f"   YuNet 다중 스케일 오류: {e}")
    
    # 3. YuNet Pro (개선된 버전)
    print("\n3. YuNet Pro 방법 (2023년 최신)")
    start_time = time.time()
    try:
        img_yunet, face_count_yunet = detect_faces_yunet_pro(image_path, score_threshold=0.5)
        yunet_time = time.time() - start_time
        if img_yunet is not None:
            print(f"   검출된 얼굴: {face_count_yunet}개")
            print(f"   처리 시간: {yunet_time:.3f}초")
            results.append(("YuNet Pro", img_yunet, face_count_yunet, yunet_time))
    except Exception as e:
        print(f"   YuNet 오류: {e}")
    
    # 4. LBP Cascade (빠른 검출)
    print("\n4. LBP Cascade 방법 (빠른 검출)")
    start_time = time.time()
    try:
        img_lbp, face_count_lbp = detect_faces_cascade_lbp(image_path)
        lbp_time = time.time() - start_time
        if img_lbp is not None:
            print(f"   검출된 얼굴: {face_count_lbp}개")
            print(f"   처리 시간: {lbp_time:.3f}초")
            results.append(("LBP Cascade", img_lbp, face_count_lbp, lbp_time))
    except Exception as e:
        print(f"   LBP Cascade 오류: {e}")
    
    # 결과 요약
    if len(results) > 0:
        print("\n" + "=" * 60)
        print("성능 요약:")
        print("-" * 60)
        
        # 가장 좋은 결과 찾기
        best_accuracy = max(results, key=lambda x: x[2])
        best_speed = min(results, key=lambda x: x[3])
        
        print(f"가장 많은 얼굴 검출: {best_accuracy[0]} ({best_accuracy[2]}개)")
        print(f"가장 빠른 속도: {best_speed[0]} ({best_speed[3]:.3f}초)")
        
        print("\n각 방법의 특징:")
        print("- TF-DNN: TensorFlow 기반 DNN, 높은 정확도 (빨간색 박스)")
        print("- YuNet-MultiScale: 다중 스케일, 작은 얼굴까지 검출 (노란색 박스)")  
        print("- YuNet Pro: 2023년 최신, 다양한 각도+표정 (파란색 박스)")
        print("- LBP Cascade: 가장 빠른 속도, CPU 친화적 (초록색 박스)")
        
        print("\n성능 비교:")
        for method, _, count, exec_time in results:
            print(f"   {method}: {count}개 얼굴 / {exec_time:.3f}초")
        
        # 이미지 표시
        images = [r[1] for r in results]
        
        # 결과 이미지 조합
        if len(images) >= 2:
            # 2x2 그리드로 배치
            if len(images) >= 4:
                top = np.hstack([images[0], images[1]])
                bottom = np.hstack([images[2], images[3]])
                combined = np.vstack([top, bottom])
            elif len(images) == 3:
                top = np.hstack([images[0], images[1]])
                # 세 번째 이미지를 같은 크기로 맞춤
                h, w = images[0].shape[:2]
                bottom = np.hstack([images[2], np.zeros((h, w, 3), dtype=np.uint8)])
                combined = np.vstack([top, bottom])
            else:
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
        
        cv2.imshow("Advanced Face Detection (Red:TF-DNN | Yellow:YuNet-MS | Blue:YuNet-Pro | Green:LBP)", combined)
        print("\n아무 키나 누르면 창이 닫힙니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 메인 실행
if __name__ == "__main__":
    # 필요한 패키지 설치 안내
    print("=" * 60)
    print("고급 얼굴 검출 프로그램")
    print("=" * 60)
    print("\n필요한 패키지:")
    print("pip install opencv-python (이미 설치됨)")
    print("-" * 60)
    
    # 테스트할 이미지 경로
    test_images = [
        "C:/develop/vision-test/file/soccer.png",
        # "C:/develop/vision-test/file/blur_image.webp",
        # "C:/develop/vision-test/file/bg_image.webp",
        # "C:/develop/vision-test/file/old_image.webp"
    ]
    
    # 사용 가능한 이미지 찾기
    for path in test_images:
        if os.path.exists(path):
            compare_all_pro_methods(path)
            break
    else:
        print("\n이미지 파일을 찾을 수 없습니다.")
        print("다음 경로에 이미지를 추가하세요:")
        for path in test_images:
            print(f"  - {path}")