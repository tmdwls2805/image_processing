import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os

class MotionBlurCorrector:
    def __init__(self):
        pass
    
    def detect_motion_direction(self, image):
        """
        흔들림 방향을 감지합니다.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 소벨 필터로 가장자리 검출
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 방향 계산
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        angle = np.arctan2(sobely, sobelx)
        
        # 평균 각도로 흔들림 방향 추정
        avg_angle = np.mean(angle[magnitude > np.mean(magnitude)])
        
        return avg_angle, magnitude
    
    def create_deblur_kernel(self, length=15, angle=0):
        """
        모션 블러 보정을 위한 커널을 생성합니다.
        """
        # 각도를 라디안으로 변환
        angle = np.deg2rad(angle)
        
        # 커널 크기 설정
        kernel_size = length
        kernel = np.zeros((kernel_size, kernel_size))
        
        # 중심점
        center = kernel_size // 2
        
        # 선형 모션 블러 커널 생성
        for i in range(kernel_size):
            x = int(center + (i - center) * np.cos(angle))
            y = int(center + (i - center) * np.sin(angle))
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # 정규화
        kernel = kernel / np.sum(kernel) if np.sum(kernel) != 0 else kernel
        
        return kernel
    
    def wiener_deconvolution(self, image, kernel, noise_ratio=0.01):
        """
        위너 디컨볼루션을 사용한 블러 제거
        """
        # 이미지를 float32로 변환
        image_float = image.astype(np.float32) / 255.0
        
        # FFT 변환
        img_fft = np.fft.fft2(image_float)
        kernel_fft = np.fft.fft2(kernel, s=image_float.shape)
        
        # 위너 필터 적용
        kernel_conj = np.conj(kernel_fft)
        kernel_mag_sq = np.abs(kernel_fft) ** 2
        
        # 위너 디컨볼루션 공식
        wiener_filter = kernel_conj / (kernel_mag_sq + noise_ratio)
        result_fft = img_fft * wiener_filter
        
        # 역 FFT 변환
        result = np.fft.ifft2(result_fft)
        result = np.abs(result)
        
        # 0-255 범위로 변환
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def advanced_deblur(self, image):
        """
        고급 디블러링 기법을 적용합니다.
        """
        # 언샤프 마스킹
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # 적응적 히스토그램 평활화
        if len(image.shape) == 3:
            lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(unsharp)
        
        return enhanced
    
    def denoise_image(self, image):
        """
        노이즈 제거
        """
        # Non-local means denoising
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def enhance_sharpness(self, image):
        """
        선명도 향상
        """
        # OpenCV를 PIL로 변환
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        
        # 선명도 향상
        enhancer = ImageEnhance.Sharpness(pil_image)
        sharpened = enhancer.enhance(1.5)
        
        # 대비 향상
        enhancer = ImageEnhance.Contrast(sharpened)
        contrasted = enhancer.enhance(1.2)
        
        # 다시 OpenCV로 변환
        result_array = np.array(contrasted)
        if len(image.shape) == 3:
            result_cv = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        else:
            result_cv = result_array
        
        return result_cv
    
    def correct_motion_blur(self, image_path, output_path=None):
        """
        모션 블러 보정을 수행합니다.
        """
        try:
            # 이미지 로드
            print("이미지 로드 중...")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            print("흔들림 감지 및 분석 중...")
            # 흔들림 방향 감지
            angle, magnitude = self.detect_motion_direction(image)
            print(f"감지된 흔들림 각도: {np.degrees(angle):.2f}도")
            
            # 각 채널별로 처리
            corrected_channels = []
            for i in range(3):  # BGR 채널
                channel = image[:, :, i]
                
                print(f"채널 {i+1} 처리 중...")
                
                # 디블러링 커널 생성
                kernel = self.create_deblur_kernel(length=15, angle=np.degrees(angle))
                
                # 위너 디컨볼루션 적용
                deblurred = self.wiener_deconvolution(channel, kernel, noise_ratio=0.01)
                
                corrected_channels.append(deblurred)
            
            # 채널 합치기
            corrected = cv2.merge(corrected_channels)
            
            print("고급 디블러링 적용 중...")
            # 고급 디블러링 기법 적용
            advanced_corrected = self.advanced_deblur(corrected)
            
            print("노이즈 제거 중...")
            # 노이즈 제거
            denoised = self.denoise_image(advanced_corrected)
            
            print("선명도 향상 중...")
            # 선명도 향상
            final_result = self.enhance_sharpness(denoised)
            
            # 출력 경로 설정
            if output_path is None:
                base_name = os.path.splitext(image_path)[0]
                ext = os.path.splitext(image_path)[1]
                output_path = f"{base_name}_motion_corrected{ext}"
            
            # 결과 저장
            cv2.imwrite(output_path, final_result)
            print(f"모션 블러 보정 완료: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return None

def main():
    # 이미지 경로
    image_path = "C:/develop/vision-test/file/jijin.png"
    
    # 모션 블러 보정기 생성
    corrector = MotionBlurCorrector()
    
    # 모션 블러 보정 실행
    result = corrector.correct_motion_blur(image_path)
    
    if result:
        print(f"성공적으로 보정된 이미지: {result}")
    else:
        print("이미지 보정에 실패했습니다.")

if __name__ == "__main__":
    main()