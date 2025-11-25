import cv2
import numpy as np
import os

class SimpleSharpEnhancer:
    def __init__(self):
        pass
    
    def apply_sharpening_kernel(self, image):
        """
        강력한 샤프닝 커널 적용
        """
        # 고성능 샤프닝 커널
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def laplacian_sharpening(self, image):
        """
        라플라시안 기반 샤프닝
        """
        # 라플라시안 필터 적용
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # 원본과 합성
        sharpened = cv2.addWeighted(image, 1.0, laplacian, 0.3, 0)
        return sharpened
    
    def high_pass_filter(self, image):
        """
        하이패스 필터로 세부사항 강화
        """
        # 가우시안 블러 (로우패스)
        low_pass = cv2.GaussianBlur(image, (21, 21), 10.0)
        
        # 하이패스 = 원본 - 로우패스
        high_pass = cv2.subtract(image, low_pass)
        high_pass = cv2.add(high_pass, 128)  # 중간값 추가
        
        # 원본과 하이패스 결합
        result = cv2.addWeighted(image, 1.0, high_pass, 0.5, -64)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def adaptive_histogram_equalization(self, image):
        """
        적응적 히스토그램 평활화
        """
        # LAB 색공간으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # L 채널에 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # 채널 재결합
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detail_enhancement(self, image):
        """
        디테일 강화 필터
        """
        # 디테일 강화 커널
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        
        enhanced = cv2.filter2D(image, -1, kernel)
        return enhanced
    
    def noise_reduction_with_sharpening(self, image):
        """
        노이즈 감소하면서 샤프닝
        """
        # 약간의 노이즈 제거
        denoised = cv2.bilateralFilter(image, 5, 50, 50)
        
        # 강력한 샤프닝
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  6, -1],
            [ 0, -1,  0]
        ])
        
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return sharpened
    
    def multi_step_enhancement(self, image):
        """
        다단계 강화 처리
        """
        # 1단계: 기본 샤프닝
        step1 = self.apply_sharpening_kernel(image)
        
        # 2단계: 하이패스 필터
        step2 = self.high_pass_filter(step1)
        
        # 3단계: 디테일 강화
        step3 = self.detail_enhancement(step2)
        
        # 4단계: 적응적 히스토그램 평활화
        step4 = self.adaptive_histogram_equalization(step3)
        
        return step4
    
    def extreme_sharpening(self, image):
        """
        극강 샤프닝 (흐린 사진용)
        """
        # 1차 강력 샤프닝
        kernel1 = np.array([
            [-2, -2, -2],
            [-2, 17, -2],
            [-2, -2, -2]
        ])
        sharp1 = cv2.filter2D(image, -1, kernel1)
        
        # 2차 추가 샤프닝
        kernel2 = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        sharp2 = cv2.filter2D(sharp1, -1, kernel2)
        
        # 대비 향상
        enhanced = self.adaptive_histogram_equalization(sharp2)
        
        return enhanced
    
    def process_image(self, image_path, method='multi_step', output_path=None):
        """
        이미지 처리 실행
        """
        try:
            print(f"이미지 로드: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            print(f"처리 방법: {method}")
            
            if method == 'basic_sharp':
                result = self.apply_sharpening_kernel(image)
            elif method == 'laplacian':
                result = self.laplacian_sharpening(image)
            elif method == 'high_pass':
                result = self.high_pass_filter(image)
            elif method == 'detail':
                result = self.detail_enhancement(image)
            elif method == 'noise_sharp':
                result = self.noise_reduction_with_sharpening(image)
            elif method == 'multi_step':
                result = self.multi_step_enhancement(image)
            elif method == 'extreme':
                result = self.extreme_sharpening(image)
            else:
                print("알 수 없는 방법입니다. multi_step을 사용합니다.")
                result = self.multi_step_enhancement(image)
            
            # 출력 경로 설정
            if output_path is None:
                base_name = os.path.splitext(image_path)[0]
                ext = os.path.splitext(image_path)[1]
                output_path = f"{base_name}_{method}_enhanced{ext}"
            
            # 결과 저장
            cv2.imwrite(output_path, result)
            print(f"처리 완료: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"처리 중 오류: {str(e)}")
            return None

def main():
    image_path = "C:/develop/vision-test/file/jijin.png"
    enhancer = SimpleSharpEnhancer()
    
    print("=== 간단하고 효과적인 이미지 선명화 ===\n")
    
    # 다양한 방법 테스트
    methods = [
        ('basic_sharp', '기본 샤프닝'),
        ('high_pass', '하이패스 필터'),
        ('extreme', '극강 샤프닝'),
        ('multi_step', '다단계 강화')
    ]
    
    results = []
    
    for method, description in methods:
        print(f"--- {description} 적용 ---")
        result = enhancer.process_image(image_path, method=method)
        if result:
            results.append(result)
            print(f"저장됨: {result}\n")
        else:
            print(f"실패\n")
    
    print("=== 처리 완료 ===")
    print("생성된 파일들:")
    for result in results:
        print(f"- {os.path.basename(result)}")

if __name__ == "__main__":
    main()