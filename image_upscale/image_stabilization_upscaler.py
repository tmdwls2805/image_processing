import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

class ImageStabilizationUpscaler:
    def __init__(self):
        self.stabilizer = cv2.createBackgroundSubtractorMOG2()
        
    def stabilize_image(self, image_path):
        """
        이미지 흔들림 보정을 수행합니다.
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 노이즈 제거를 통한 흔들림 보정
        stabilized = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 가우시안 블러로 추가 안정화
        stabilized = cv2.GaussianBlur(stabilized, (3, 3), 0)
        
        return stabilized
    
    def upscale_image(self, image, scale_factor=2):
        """
        이미지 스케일업을 수행합니다.
        """
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 고품질 인터폴레이션을 사용한 스케일업
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return upscaled
    
    def enhance_quality(self, image):
        """
        이미지 품질을 향상시킵니다.
        """
        # OpenCV 이미지를 PIL로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 선명도 향상
        enhancer = ImageEnhance.Sharpness(pil_image)
        enhanced = enhancer.enhance(1.2)
        
        # 대비 향상
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # PIL을 다시 OpenCV로 변환
        enhanced_array = np.array(enhanced)
        enhanced_cv = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)
        
        return enhanced_cv
    
    def process_image(self, input_path, output_path=None, scale_factor=2):
        """
        전체 이미지 처리 파이프라인을 실행합니다.
        """
        try:
            # 흔들림 보정
            print("흔들림 보정 중...")
            stabilized = self.stabilize_image(input_path)
            
            # 스케일업
            print(f"{scale_factor}x 스케일업 중...")
            upscaled = self.upscale_image(stabilized, scale_factor)
            
            # 품질 향상
            print("품질 향상 중...")
            enhanced = self.enhance_quality(upscaled)
            
            # 출력 경로 설정
            if output_path is None:
                base_name = os.path.splitext(input_path)[0]
                ext = os.path.splitext(input_path)[1]
                output_path = f"{base_name}_stabilized_upscaled{ext}"
            
            # 결과 저장
            cv2.imwrite(output_path, enhanced)
            print(f"처리 완료: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return None

def main():
    # 이미지 경로
    image_path = "C:/develop/vision-test/file/jijin.png"
    
    # 처리기 생성
    processor = ImageStabilizationUpscaler()
    
    # 이미지 처리 (2배 스케일업)
    result = processor.process_image(image_path, scale_factor=2)
    
    if result:
        print(f"성공적으로 처리된 이미지: {result}")
    else:
        print("이미지 처리에 실패했습니다.")

if __name__ == "__main__":
    main()