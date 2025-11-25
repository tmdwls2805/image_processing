import cv2
import numpy as np
from scipy import ndimage, signal
import os

class ProfessionalDeblur:
    def __init__(self):
        pass
    
    def estimate_psf_radon(self, image, psf_size=21):
        """
        Radon transform 기반 PSF 추정 (더 정확함)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 에지 검출로 선명한 특징 찾기
        edges = cv2.Canny(gray, 50, 150)
        
        # 방향별로 projection 계산
        angles = np.arange(0, 180, 2)
        projections = []
        
        for angle in angles:
            # 이미지 회전
            rotated = ndimage.rotate(edges, angle, reshape=False)
            # 수직 projection
            projection = np.sum(rotated, axis=0)
            projections.append(np.var(projection))  # 분산이 클수록 선명
        
        # 가장 선명한 방향 찾기 (모션과 수직)
        sharpest_angle = angles[np.argmax(projections)]
        motion_angle = np.radians(sharpest_angle + 90)  # 모션 방향
        
        # 블러 길이 추정
        blur_strength = 1.0 - (np.max(projections) / (np.mean(projections) + 1e-8))
        motion_length = int(blur_strength * 20) + 5
        motion_length = min(motion_length, psf_size - 2)
        
        # PSF 생성
        psf = np.zeros((psf_size, psf_size))
        center = psf_size // 2
        
        for i in range(motion_length):
            x = int(center + (i - motion_length//2) * np.cos(motion_angle))
            y = int(center + (i - motion_length//2) * np.sin(motion_angle))
            if 0 <= x < psf_size and 0 <= y < psf_size:
                psf[y, x] = 1.0
        
        # PSF 정규화 및 스무딩
        if np.sum(psf) > 0:
            psf /= np.sum(psf)
            psf = ndimage.gaussian_filter(psf, sigma=0.5)
            psf /= np.sum(psf)
        
        return psf, motion_angle, motion_length
    
    def iterative_deconvolution(self, image, psf, iterations=30):
        """
        개선된 반복 디컨볼루션
        """
        # 정규화
        image_norm = image.astype(np.float64) / 255.0
        estimate = image_norm.copy()
        psf_flipped = np.flipud(np.fliplr(psf))
        
        # 적응적 스텝 사이즈
        step_size = 1.0
        prev_error = float('inf')
        
        for i in range(iterations):
            # Forward convolution
            convolved = signal.convolve2d(estimate, psf, mode='same', boundary='wrap')
            convolved = np.maximum(convolved, 1e-12)  # 안정성
            
            # Error 계산
            error = np.mean((image_norm - convolved) ** 2)
            
            # 스텝 사이즈 조정
            if error > prev_error:
                step_size *= 0.8  # 발산하면 줄임
            else:
                step_size = min(step_size * 1.05, 1.2)  # 수렴하면 늘림
            
            # Correction 계산
            ratio = image_norm / convolved
            correction = signal.convolve2d(ratio, psf_flipped, mode='same', boundary='wrap')
            
            # 업데이트 (스텝 사이즈 적용)
            estimate = estimate * (1 + step_size * (correction - 1))
            estimate = np.maximum(estimate, 0)  # 음수 방지
            
            prev_error = error
            
            if i % 5 == 0:
                print(f"  반복 {i+1}/{iterations}, 오차: {error:.6f}")
        
        return estimate * 255.0
    
    def shock_filter(self, image, iterations=5):
        """
        Shock filter로 엣지 강화
        """
        result = image.astype(np.float64)
        
        for i in range(iterations):
            # Gradient 계산
            grad_x = cv2.Sobel(result, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Laplacian (2차 미분)
            laplacian = cv2.Laplacian(result, cv2.CV_64F)
            
            # Shock filter 업데이트
            dt = 0.1
            shock_term = np.sign(laplacian) * grad_mag
            result = result + dt * shock_term
            
            print(f"  Shock filter {i+1}/{iterations}")
        
        return np.clip(result, 0, 255)
    
    def blind_deconvolution_em(self, blurred, max_iter=20, psf_size=21):
        """
        EM 알고리즘 기반 Blind Deconvolution
        """
        print("  EM 알고리즘으로 Blind Deconvolution 수행...")
        
        # 초기화
        if len(blurred.shape) == 3:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        else:
            gray = blurred.copy()
        
        # 초기 PSF 추정
        psf, _, _ = self.estimate_psf_radon(gray, psf_size)
        
        # 정규화
        blurred_norm = blurred.astype(np.float64) / 255.0
        
        for iteration in range(max_iter):
            print(f"    EM 반복 {iteration+1}/{max_iter}")
            
            # E-step: 이미지 추정
            if len(blurred.shape) == 3:
                estimated_img = np.zeros_like(blurred_norm)
                for c in range(3):
                    estimated_img[:, :, c] = self.iterative_deconvolution(
                        blurred[:, :, c], psf, iterations=10
                    ) / 255.0
            else:
                estimated_img = self.iterative_deconvolution(blurred, psf, iterations=10) / 255.0
            
            # M-step: PSF 업데이트 (간단한 형태)
            # 실제로는 복잡하지만 여기서는 초기 추정을 유지
            
        return estimated_img * 255.0
    
    def process_image(self, image_path, method='professional', output_path=None):
        """
        전문가급 이미지 처리
        """
        try:
            print(f"이미지 로드: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            print(f"처리 방법: {method}")
            
            if method == 'radon_psf':
                print("Radon Transform PSF 추정...")
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                psf, angle, length = self.estimate_psf_radon(gray, 25)
                print(f"  모션: {np.degrees(angle):.1f}°, 길이: {length}px")
                
                # 채널별 처리
                if len(image.shape) == 3:
                    result = np.zeros_like(image, dtype=np.float64)
                    for i in range(3):
                        print(f"  채널 {i+1} 처리...")
                        deblurred = self.iterative_deconvolution(image[:, :, i], psf)
                        result[:, :, i] = deblurred
                else:
                    result = self.iterative_deconvolution(image, psf)
            
            elif method == 'shock_filter':
                print("Shock Filter 적용...")
                if len(image.shape) == 3:
                    result = np.zeros_like(image, dtype=np.float64)
                    for i in range(3):
                        print(f"  채널 {i+1} 처리...")
                        result[:, :, i] = self.shock_filter(image[:, :, i])
                else:
                    result = self.shock_filter(image)
            
            elif method == 'blind_em':
                print("Blind Deconvolution (EM)...")
                result = self.blind_deconvolution_em(image)
            
            elif method == 'professional':
                print("전문가급 종합 처리...")
                
                # 1단계: Radon PSF 추정 및 디컨볼루션
                print("1단계: PSF 추정 및 디컨볼루션")
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                psf, angle, length = self.estimate_psf_radon(gray, 25)
                print(f"  모션 감지: {np.degrees(angle):.1f}°, {length}px")
                
                if len(image.shape) == 3:
                    step1 = np.zeros_like(image, dtype=np.float64)
                    for i in range(3):
                        step1[:, :, i] = self.iterative_deconvolution(image[:, :, i], psf, 25)
                else:
                    step1 = self.iterative_deconvolution(image, psf, 25)
                
                # 2단계: Shock Filter로 엣지 강화
                print("2단계: 엣지 강화")
                if len(image.shape) == 3:
                    step2 = np.zeros_like(step1)
                    for i in range(3):
                        step2[:, :, i] = self.shock_filter(step1[:, :, i], 3)
                else:
                    step2 = self.shock_filter(step1, 3)
                
                result = step2
            
            else:
                print("알 수 없는 방법, professional 사용")
                return self.process_image(image_path, 'professional', output_path)
            
            # 후처리
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            print("3단계: 최종 품질 향상")
            # CLAHE
            if len(result.shape) == 3:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                l = clahe.apply(l)
                result = cv2.merge([l, a, b])
                result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
            
            # 최종 샤프닝
            kernel = np.array([[-0.5, -1, -0.5], [-1, 7, -1], [-0.5, -1, -0.5]]) / 3
            result = cv2.filter2D(result, -1, kernel)
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # 출력
            if output_path is None:
                base_name = os.path.splitext(image_path)[0]
                ext = os.path.splitext(image_path)[1]
                output_path = f"{base_name}_pro_{method}{ext}"
            
            cv2.imwrite(output_path, result)
            print(f"전문 처리 완료: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    image_path = "C:/develop/vision-test/file/jijin.png"
    processor = ProfessionalDeblur()
    
    print("=== 전문가급 모션 블러 보정 ===\n")
    
    methods = [
        ('radon_psf', 'Radon Transform PSF 추정'),
        ('shock_filter', 'Shock Filter 엣지 강화'),
        ('blind_em', 'Blind Deconvolution (EM)'),
        ('professional', '전문가급 종합 처리')
    ]
    
    results = []
    
    for method, description in methods:
        print(f"--- {description} ---")
        result = processor.process_image(image_path, method=method)
        if result:
            results.append(result)
            print(f"✓ 완료: {os.path.basename(result)}\n")
        else:
            print("✗ 실패\n")
    
    print("=== 모든 처리 완료 ===")
    for result in results:
        print(f"- {os.path.basename(result)}")

if __name__ == "__main__":
    main()