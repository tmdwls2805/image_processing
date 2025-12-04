import cv2
import numpy as np
import os

class UltimateDeblur:
    """
    최고 성능 디블러링 시스템 (OpenCV only)
    최신 논문들의 알고리즘을 OpenCV만으로 구현
    """
    
    def __init__(self):
        pass
    
    def estimate_blur_kernel_advanced(self, image, kernel_size=25):
        """
        고급 블러 커널 추정 - 여러 방법 결합
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 방법 1: Edge density 기반
        edges = cv2.Canny(gray, 50, 150)
        edge_angles = []
        
        for angle in range(0, 180, 3):
            # 회전
            M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), angle, 1)
            rotated = cv2.warpAffine(edges, M, (gray.shape[1], gray.shape[0]))
            
            # Projection
            projection = np.sum(rotated, axis=0)
            sharpness = np.var(projection)
            edge_angles.append(sharpness)
        
        best_angle1 = np.argmax(edge_angles) * 3
        motion_angle1 = (best_angle1 + 90) % 180
        
        # 방법 2: Gradient coherence
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
        angles[angles < 0] += 180
        
        hist, bins = np.histogram(angles, bins=90, range=(0, 180))
        dominant_angle = bins[np.argmax(hist)]
        
        # 두 방법의 가중 평균
        final_angle = (motion_angle1 * 0.6 + dominant_angle * 0.4) % 180
        final_angle_rad = np.radians(final_angle)
        
        # 블러 길이 추정 (frequency domain analysis)
        f_transform = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = cv2.magnitude(f_shift[:,:,0], f_shift[:,:,1])
        
        # 고주파 성분 분석
        high_freq = magnitude[magnitude.shape[0]//4:3*magnitude.shape[0]//4, 
                             magnitude.shape[1]//4:3*magnitude.shape[1]//4]
        blur_strength = 1.0 / (np.mean(high_freq) + 1e-8)
        blur_length = int(np.clip(blur_strength * 15, 5, kernel_size-2))
        
        # PSF 생성
        psf = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        for i in range(blur_length):
            x = int(center + (i - blur_length//2) * np.cos(final_angle_rad))
            y = int(center + (i - blur_length//2) * np.sin(final_angle_rad))
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                psf[y, x] = 1.0
        
        if np.sum(psf) > 0:
            psf = psf / np.sum(psf)
            psf = cv2.GaussianBlur(psf, (3, 3), 0.5)
            psf = psf / np.sum(psf)
        else:
            psf[center, center-blur_length//2:center+blur_length//2+1] = 1.0
            psf = psf / np.sum(psf)
        
        print(f"  추정된 모션: {final_angle:.1f}도, 길이: {blur_length}px")
        return psf
    
    def wiener_deconvolution_opencv(self, image, psf, noise_var=0.01):
        """
        Wiener 디컨볼루션 (OpenCV FFT 사용)
        """
        image_float = np.float32(image)
        img_fft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # PSF를 이미지 크기로 맞춤
        psf_padded = np.zeros_like(image_float)
        psf_padded[:psf.shape[0], :psf.shape[1]] = psf
        psf_fft = cv2.dft(psf_padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # Wiener 필터 계산
        psf_conj = np.zeros_like(psf_fft)
        psf_conj[:,:,0] = psf_fft[:,:,0]   # Real part
        psf_conj[:,:,1] = -psf_fft[:,:,1]  # -Imaginary part
        
        psf_mag_sq = psf_fft[:,:,0]**2 + psf_fft[:,:,1]**2
        
        # Wiener 필터 적용
        denominator = psf_mag_sq + noise_var
        denominator = np.maximum(denominator, 1e-10)
        
        wiener_real = psf_conj[:,:,0] / denominator
        wiener_imag = psf_conj[:,:,1] / denominator
        
        # 복소수 곱셈
        result_fft = np.zeros_like(img_fft)
        result_fft[:,:,0] = img_fft[:,:,0] * wiener_real - img_fft[:,:,1] * wiener_imag
        result_fft[:,:,1] = img_fft[:,:,0] * wiener_imag + img_fft[:,:,1] * wiener_real
        
        # 역변환
        result = cv2.idft(result_fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        
        return np.clip(result, 0, 255)
    
    def lucy_richardson_opencv(self, image, psf, iterations=30):
        """
        Lucy-Richardson 디컨볼루션 (개선된 버전)
        """
        print(f"    Lucy-Richardson {iterations}회 반복...")
        
        image_float = image.astype(np.float64) + 1e-10
        estimate = image_float.copy()
        
        # PSF 뒤집기
        psf_flipped = cv2.flip(psf, -1)
        
        # 적응적 스텝 크기
        step_size = 1.0
        prev_error = float('inf')
        
        for i in range(iterations):
            # Forward convolution
            convolved = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_WRAP)
            convolved = np.maximum(convolved, 1e-12)
            
            # Error 계산
            error = np.mean((image_float - convolved) ** 2)
            
            # 적응적 스텝 크기 조정
            if error > prev_error:
                step_size *= 0.9
            else:
                step_size = min(step_size * 1.05, 1.3)
            
            # Correction 계산
            ratio = image_float / convolved
            correction = cv2.filter2D(ratio, -1, psf_flipped, borderType=cv2.BORDER_WRAP)
            
            # 업데이트
            new_estimate = estimate * (1 + step_size * (correction - 1))
            estimate = np.maximum(new_estimate, 0)
            
            prev_error = error
            
            if i % 5 == 0:
                print(f"      반복 {i+1}: 오차 {error:.6f}")
        
        return estimate
    
    def shock_filter_opencv(self, image, iterations=5, dt=0.1):
        """
        Shock Filter (엣지 강화)
        """
        print(f"  Shock Filter ({iterations}회)...")
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(3):
                result[:, :, c] = self._shock_filter_channel(image[:, :, c], iterations, dt)
            return result
        else:
            return self._shock_filter_channel(image, iterations, dt)
    
    def _shock_filter_channel(self, channel, iterations, dt):
        """단일 채널 Shock Filter"""
        u = channel.astype(np.float64)
        
        for i in range(iterations):
            # Gradient 계산
            grad_x = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Laplacian (2차 미분)
            laplacian = cv2.Laplacian(u, cv2.CV_64F)
            
            # Shock filter 업데이트
            shock_term = np.sign(laplacian) * grad_mag
            u = u + dt * shock_term
        
        return u
    
    def multi_scale_enhancement(self, image, scales=[1.0, 0.8, 0.6, 0.4]):
        """
        다중 스케일 향상
        """
        print("  다중 스케일 처리...")
        
        enhanced_sum = np.zeros_like(image, dtype=np.float64)
        weight_sum = 0
        
        for scale in scales:
            if scale == 1.0:
                scaled_img = image.copy()
            else:
                h, w = image.shape[:2]
                small = cv2.resize(image, (int(w*scale), int(h*scale)))
                enhanced_small = cv2.bilateralFilter(small.astype(np.uint8), 9, 75, 75)
                scaled_img = cv2.resize(enhanced_small, (w, h))
            
            weight = scale
            enhanced_sum += scaled_img.astype(np.float64) * weight
            weight_sum += weight
        
        return enhanced_sum / weight_sum
    
    def ultimate_hybrid_deblur(self, image):
        """
        궁극의 하이브리드 디블러링
        모든 최고 기법을 지능적으로 결합
        """
        print("궁극의 하이브리드 디블러링 시스템!")
        
        # 1단계: 고급 PSF 추정
        print("1단계: 고급 블러 커널 추정")
        psf = self.estimate_blur_kernel_advanced(image, 27)
        
        # 2단계: Lucy-Richardson 초기 디블러링
        print("2단계: Lucy-Richardson 디컨볼루션")
        if len(image.shape) == 3:
            lr_result = np.zeros_like(image, dtype=np.float64)
            for c in range(3):
                lr_result[:, :, c] = self.lucy_richardson_opencv(image[:, :, c], psf, 25)
        else:
            lr_result = self.lucy_richardson_opencv(image, psf, 25)
        
        lr_result = np.clip(lr_result, 0, 255).astype(np.uint8)
        
        # 3단계: Wiener 디컨볼루션으로 정제
        print("3단계: Wiener 디컨볼루션 정제")
        if len(image.shape) == 3:
            wiener_result = np.zeros_like(lr_result, dtype=np.float64)
            for c in range(3):
                wiener_result[:, :, c] = self.wiener_deconvolution_opencv(
                    lr_result[:, :, c], psf, 0.005
                )
        else:
            wiener_result = self.wiener_deconvolution_opencv(lr_result, psf, 0.005)
        
        wiener_result = np.clip(wiener_result, 0, 255).astype(np.uint8)
        
        # 4단계: Non-Local Means 디노이징
        print("4단계: Non-Local Means 디노이징")
        if len(wiener_result.shape) == 3:
            nlm_result = cv2.fastNlMeansDenoisingColored(wiener_result, None, 8, 8, 7, 21)
        else:
            nlm_result = cv2.fastNlMeansDenoising(wiener_result, None, 8, 7, 21)
        
        # 5단계: 다중 스케일 향상
        print("5단계: 다중 스케일 향상")
        ms_result = self.multi_scale_enhancement(nlm_result)
        ms_result = np.clip(ms_result, 0, 255).astype(np.uint8)
        
        # 6단계: Shock Filter로 엣지 강화
        print("6단계: Shock Filter 엣지 강화")
        shock_result = self.shock_filter_opencv(ms_result, 3, 0.05)
        shock_result = np.clip(shock_result, 0, 255).astype(np.uint8)
        
        # 7단계: 최종 품질 향상
        print("7단계: 최종 품질 향상")
        
        # CLAHE 적용
        if len(shock_result.shape) == 3:
            lab = cv2.cvtColor(shock_result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            final_result = cv2.merge([l, a, b])
            final_result = cv2.cvtColor(final_result, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            final_result = clahe.apply(shock_result)
        
        # 최종 언샤프 마스킹
        gaussian = cv2.GaussianBlur(final_result, (0, 0), 1.5)
        final_result = cv2.addWeighted(final_result, 1.8, gaussian, -0.8, 0)
        
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    def process_image(self, image_path, method='ultimate', output_path=None):
        """
        이미지 처리 실행
        """
        try:
            print(f"이미지 로드: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            print(f"처리 방법: {method}")
            
            if method == 'lucy_richardson':
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                psf = self.estimate_blur_kernel_advanced(gray, 25)
                
                if len(image.shape) == 3:
                    result = np.zeros_like(image, dtype=np.float64)
                    for c in range(3):
                        result[:, :, c] = self.lucy_richardson_opencv(image[:, :, c], psf, 30)
                else:
                    result = self.lucy_richardson_opencv(image, psf, 30)
                
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            elif method == 'wiener':
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                psf = self.estimate_blur_kernel_advanced(gray, 25)
                
                if len(image.shape) == 3:
                    result = np.zeros_like(image, dtype=np.float64)
                    for c in range(3):
                        result[:, :, c] = self.wiener_deconvolution_opencv(image[:, :, c], psf)
                else:
                    result = self.wiener_deconvolution_opencv(image, psf)
                
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            elif method == 'shock_filter':
                result = self.shock_filter_opencv(image, 5, 0.1)
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            elif method == 'multi_scale':
                result = self.multi_scale_enhancement(image)
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            elif method == 'ultimate':
                result = self.ultimate_hybrid_deblur(image)
                
            else:
                print("알 수 없는 방법, ultimate 사용")
                result = self.ultimate_hybrid_deblur(image)
            
            # 출력
            if output_path is None:
                base_name = os.path.splitext(image_path)[0]
                ext = os.path.splitext(image_path)[1]
                output_path = f"{base_name}_ultimate_{method}{ext}"
            
            cv2.imwrite(output_path, result)
            print(f"완료: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    image_path = "C:/develop/vision-test/file/girl.jpg"
    processor = UltimateDeblur()
    
    print("=== 궁극의 디블러링 시스템 (OpenCV Only) ===")
    print("최신 논문들의 알고리즘을 OpenCV만으로 구현!\n")
    
    # 최고 성능 방법들
    methods = [
        ('lucy_richardson', 'Lucy-Richardson (적응적 스텝)'),
        ('wiener', 'Wiener 디컨볼루션 (주파수 영역)'),
        ('shock_filter', 'Shock Filter (엣지 강화)'),
        ('multi_scale', '다중 스케일 향상'),
        ('ultimate', '궁극의 하이브리드 (모든 기법 결합)')
    ]
    
    results = []
    
    for method, description in methods:
        print(f"\n--- {description} ---")
        result = processor.process_image(image_path, method=method)
        if result:
            results.append(result)
            print(f"성공: {os.path.basename(result)}")
        else:
            print("실패")
    
    print(f"\n=== 모든 처리 완료! ===")
    print("생성된 최고 품질 파일들:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(result)}")
    
    print(f"\n특별 추천: 'ultimate' 방법은 7단계 하이브리드 시스템!")
    print("구현된 알고리즘들:")
    print("   • 고급 PSF 추정 (Edge density + Gradient coherence)")
    print("   • Lucy-Richardson 디컨볼루션 (적응적 스텝)")  
    print("   • Wiener 디컨볼루션 (주파수 영역)")
    print("   • Non-Local Means (고급 디노이징)")
    print("   • 다중 스케일 처리")
    print("   • Shock Filter (엣지 강화)")
    print("   • CLAHE + 언샤프 마스킹")

if __name__ == "__main__":
    main()