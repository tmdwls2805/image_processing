import cv2
import numpy as np
import os
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import minimize

class UltimateDeblur:
    def __init__(self):
        self.patch_size = 7
        self.search_window = 21
    
    def estimate_noise_level(self, image):
        """
        노이즈 레벨 자동 추정
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Robust noise estimation using MAD (Median Absolute Deviation)
        gray_float = gray.astype(np.float64)
        
        # High-pass filter for noise estimation
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8.0
        filtered = cv2.filter2D(gray_float, -1, kernel)
        
        # MAD estimation
        noise_sigma = 1.4826 * np.median(np.abs(filtered - np.median(filtered)))
        
        return max(noise_sigma, 1.0)  # Ensure minimum noise level
    
    def non_local_means_deblur(self, image, psf, h=10, search_window=21, patch_size=7):
        """
        Non-Local Means 기반 디블러링 - 패치 유사성 활용
        """
        print("  Non-Local Means 디블러링 적용...")
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(3):
                result[:, :, c] = self._nlm_channel(image[:, :, c], psf, h, search_window, patch_size)
            return result
        else:
            return self._nlm_channel(image, psf, h, search_window, patch_size)
    
    def _nlm_channel(self, channel, psf, h, search_window, patch_size):
        """
        단일 채널에 대한 NLM 디블러링
        """
        channel = channel.astype(np.float64) / 255.0
        h_norm = h / 255.0
        
        # 초기 추정 (Lucy-Richardson)
        estimate = self._richardson_lucy_fast(channel, psf, 15)
        
        # Non-local means refinement
        refined = self._nlm_restoration(estimate, channel, psf, h_norm, search_window, patch_size)
        
        return refined * 255.0
    
    def _nlm_restoration(self, estimate, observed, psf, h, search_window, patch_size):
        """
        NLM 기반 복원
        """
        rows, cols = estimate.shape
        result = np.zeros_like(estimate)
        
        # Padding
        pad = search_window // 2
        estimate_padded = np.pad(estimate, pad, mode='reflect')
        
        for i in range(rows):
            for j in range(cols):
                # Current pixel
                pi, pj = i + pad, j + pad
                current_patch = estimate_padded[pi-patch_size//2:pi+patch_size//2+1, 
                                               pj-patch_size//2:pj+patch_size//2+1]
                
                # Search window
                si_start = max(pi - search_window//2, patch_size//2)
                si_end = min(pi + search_window//2, rows + pad - patch_size//2)
                sj_start = max(pj - search_window//2, patch_size//2)
                sj_end = min(pj + search_window//2, cols + pad - patch_size//2)
                
                weights = []
                values = []
                
                for si in range(si_start, si_end):
                    for sj in range(sj_start, sj_end):
                        search_patch = estimate_padded[si-patch_size//2:si+patch_size//2+1,
                                                       sj-patch_size//2:sj+patch_size//2+1]
                        
                        # Patch distance
                        if current_patch.shape == search_patch.shape:
                            distance = np.mean((current_patch - search_patch) ** 2)
                            weight = np.exp(-distance / (h * h))
                            weights.append(weight)
                            values.append(estimate_padded[si, sj])
                
                # Weighted average
                if weights:
                    weights = np.array(weights)
                    values = np.array(values)
                    result[i, j] = np.sum(weights * values) / np.sum(weights)
                else:
                    result[i, j] = estimate[i, j]
        
        return result
    
    def tv_l1_deconvolution(self, image, psf, lambda_tv=0.02, lambda_l1=0.01, iterations=100):
        """
        Total Variation + L1 정규화 디컨볼루션
        """
        print("  TV-L1 정규화 디컨볼루션 적용...")
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(3):
                result[:, :, c] = self._tv_l1_channel(image[:, :, c], psf, lambda_tv, lambda_l1, iterations)
            return result
        else:
            return self._tv_l1_channel(image, psf, lambda_tv, lambda_l1, iterations)
    
    def _tv_l1_channel(self, channel, psf, lambda_tv, lambda_l1, iterations):
        """
        단일 채널 TV-L1 처리
        """
        channel_norm = channel.astype(np.float64) / 255.0
        
        # 초기 추정
        x = channel_norm.copy()
        
        # Gradient operators
        def grad_x(u):
            return np.diff(u, axis=1, append=u[:, -1:])
        
        def grad_y(u):
            return np.diff(u, axis=0, append=u[-1:, :])
        
        def div_op(px, py):
            # Divergence operator
            dx = np.diff(px, axis=1, prepend=px[:, :1])
            dy = np.diff(py, axis=0, prepend=py[:1, :])
            return dx + dy
        
        # Dual variables
        px = np.zeros_like(x)
        py = np.zeros_like(x)
        
        # Parameters
        tau = 0.02
        sigma = 1.0 / (8.0 * tau)
        theta = 1.0
        
        x_bar = x.copy()
        
        for i in range(iterations):
            # Update dual variables
            grad_x_bar = grad_x(x_bar)
            grad_y_bar = grad_y(x_bar)
            
            px_new = px + sigma * grad_x_bar
            py_new = py + sigma * grad_y_bar
            
            # Projection onto TV constraint
            norm_p = np.sqrt(px_new**2 + py_new**2)
            norm_p = np.maximum(norm_p, lambda_tv)
            px = px_new * lambda_tv / norm_p
            py = py_new * lambda_tv / norm_p
            
            # Update primal variable
            x_old = x.copy()
            
            # Data fidelity + L1 regularization
            div_p = div_op(px, py)
            x_temp = x + tau * div_p
            
            # Proximal operator for data fidelity
            conv_x = ndimage.convolve(x_temp, psf, mode='wrap')
            residual = conv_x - channel_norm
            conv_residual = ndimage.convolve(residual, np.flipud(np.fliplr(psf)), mode='wrap')
            
            x = x_temp - tau * conv_residual
            
            # L1 proximal operator (soft thresholding)
            x = np.sign(x) * np.maximum(np.abs(x) - tau * lambda_l1, 0)
            
            # Ensure non-negative
            x = np.maximum(x, 0)
            
            # Update extrapolated variable
            x_bar = x + theta * (x - x_old)
            
            if i % 20 == 0:
                print(f"    TV-L1 반복 {i+1}/{iterations}")
        
        return x * 255.0
    
    def bm3d_style_deblur(self, image, psf, sigma_noise=None):
        """
        BM3D 스타일 3D 변환 디블러링
        """
        print("  BM3D 스타일 3D 변환 디블러링 적용...")
        
        if sigma_noise is None:
            sigma_noise = self.estimate_noise_level(image)
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(3):
                result[:, :, c] = self._bm3d_channel(image[:, :, c], psf, sigma_noise)
            return result
        else:
            return self._bm3d_channel(image, psf, sigma_noise)
    
    def _bm3d_channel(self, channel, psf, sigma_noise):
        """
        단일 채널 BM3D 스타일 처리
        """
        channel_norm = channel.astype(np.float64) / 255.0
        
        # 1단계: 초기 디블러링
        initial = self._richardson_lucy_fast(channel_norm, psf, 10)
        
        # 2단계: 블록 매칭 및 3D 변환
        denoised = self._block_matching_3d(initial, sigma_noise / 255.0)
        
        # 3단계: 다시 디컨볼루션으로 정제
        refined = self._richardson_lucy_fast(denoised, psf, 10)
        
        return refined * 255.0
    
    def _block_matching_3d(self, image, sigma, block_size=8, search_window=39, max_matches=16):
        """
        간단한 블록 매칭 3D 변환
        """
        rows, cols = image.shape
        result = np.zeros_like(image)
        weight_map = np.zeros_like(image)
        
        step = block_size // 2
        
        for i in range(0, rows - block_size + 1, step):
            for j in range(0, cols - block_size + 1, step):
                # Reference block
                ref_block = image[i:i+block_size, j:j+block_size]
                
                # Find similar blocks
                similar_blocks = []
                positions = []
                
                # Search window
                si_start = max(0, i - search_window//2)
                si_end = min(rows - block_size, i + search_window//2)
                sj_start = max(0, j - search_window//2)
                sj_end = min(cols - block_size, j + search_window//2)
                
                distances = []
                candidates = []
                
                for si in range(si_start, si_end, step):
                    for sj in range(sj_start, sj_end, step):
                        candidate = image[si:si+block_size, sj:sj+block_size]
                        distance = np.mean((ref_block - candidate) ** 2)
                        distances.append(distance)
                        candidates.append((candidate, si, sj))
                
                # Select most similar blocks
                indices = np.argsort(distances)[:max_matches]
                
                for idx in indices:
                    block, si, sj = candidates[idx]
                    similar_blocks.append(block)
                    positions.append((si, sj))
                
                if similar_blocks:
                    # Stack into 3D array
                    block_3d = np.stack(similar_blocks, axis=2)
                    
                    # 3D DCT denoising
                    denoised_3d = self._dct_denoise_3d(block_3d, sigma)
                    
                    # Aggregate back
                    for k, (si, sj) in enumerate(positions):
                        denoised_block = denoised_3d[:, :, k]
                        result[si:si+block_size, sj:sj+block_size] += denoised_block
                        weight_map[si:si+block_size, sj:sj+block_size] += 1
        
        # Normalize by weights
        weight_map = np.maximum(weight_map, 1)
        result = result / weight_map
        
        return result
    
    def _dct_denoise_3d(self, block_3d, sigma):
        """
        3D DCT 기반 디노이징
        """
        # 3D DCT
        dct_coeffs = np.zeros_like(block_3d)
        for k in range(block_3d.shape[2]):
            dct_coeffs[:, :, k] = cv2.dct(block_3d[:, :, k].astype(np.float32))
        
        # Hard thresholding
        threshold = 3 * sigma
        dct_coeffs = np.sign(dct_coeffs) * np.maximum(np.abs(dct_coeffs) - threshold, 0)
        
        # Inverse DCT
        denoised = np.zeros_like(block_3d)
        for k in range(block_3d.shape[2]):
            denoised[:, :, k] = cv2.idct(dct_coeffs[:, :, k])
        
        return denoised
    
    def dark_channel_deblur(self, image, psf, omega=0.95, t0=0.1):
        """
        Dark Channel Prior 기반 디블러링
        """
        print("  Dark Channel Prior 디블러링 적용...")
        
        if len(image.shape) != 3:
            # Convert grayscale to RGB for processing
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        image_norm = image.astype(np.float64) / 255.0
        
        # Dark channel 계산
        dark_channel = self._get_dark_channel(image_norm)
        
        # Atmospheric light 추정
        A = self._estimate_atmospheric_light(image_norm, dark_channel)
        
        # Transmission 추정
        transmission = self._estimate_transmission(image_norm, A, omega)
        transmission = np.maximum(transmission, t0)
        
        # Scene radiance 복원
        J = np.zeros_like(image_norm)
        for c in range(3):
            J[:, :, c] = (image_norm[:, :, c] - A[c]) / transmission + A[c]
        
        J = np.clip(J, 0, 1)
        
        # 디블러링 적용
        deblurred = np.zeros_like(J)
        for c in range(3):
            deblurred[:, :, c] = self._richardson_lucy_fast(J[:, :, c], psf, 20)
        
        return deblurred * 255.0
    
    def _get_dark_channel(self, image, window_size=15):
        """
        Dark channel 계산
        """
        min_channels = np.min(image, axis=2)
        kernel = np.ones((window_size, window_size))
        dark_channel = cv2.erode(min_channels, kernel)
        return dark_channel
    
    def _estimate_atmospheric_light(self, image, dark_channel):
        """
        Atmospheric light 추정
        """
        h, w = dark_channel.shape
        num_pixels = h * w
        num_brightest = max(num_pixels // 1000, 1)
        
        # 가장 밝은 픽셀들 찾기
        flat_dark = dark_channel.flatten()
        indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
        
        A = np.zeros(3)
        for c in range(3):
            flat_channel = image[:, :, c].flatten()
            A[c] = np.mean(flat_channel[indices])
        
        return A
    
    def _estimate_transmission(self, image, A, omega):
        """
        Transmission 추정
        """
        normalized = np.zeros_like(image)
        for c in range(3):
            normalized[:, :, c] = image[:, :, c] / A[c]
        
        transmission = 1 - omega * self._get_dark_channel(normalized)
        return transmission
    
    def _richardson_lucy_fast(self, image, psf, iterations=20):
        """
        빠른 Richardson-Lucy 구현
        """
        estimate = image.copy()
        psf_flipped = np.flipud(np.fliplr(psf))
        
        for _ in range(iterations):
            conv_estimate = ndimage.convolve(estimate, psf, mode='wrap')
            conv_estimate = np.maximum(conv_estimate, 1e-12)
            ratio = image / conv_estimate
            correction = ndimage.convolve(ratio, psf_flipped, mode='wrap')
            estimate = estimate * correction
            estimate = np.maximum(estimate, 0)
        
        return estimate
    
    def intelligent_hybrid_deblur(self, image, psf_size=25):
        """
        지능적 하이브리드 시스템 - 모든 최고 기법 결합
        """
        print("지능적 하이브리드 디블러링 시스템...")
        
        # PSF 추정
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        psf = self._estimate_smart_psf(gray, psf_size)
        noise_level = self.estimate_noise_level(image)
        
        print(f"  추정된 노이즈 레벨: {noise_level:.2f}")
        
        # 1단계: BM3D 스타일 초기 디블러링
        print("1단계: BM3D 스타일 처리")
        result1 = self.bm3d_style_deblur(image, psf, noise_level)
        
        # 2단계: Non-Local Means 정제
        print("2단계: Non-Local Means 정제")
        h_param = max(noise_level * 0.8, 10)
        result2 = self.non_local_means_deblur(result1.astype(np.uint8), psf, h_param)
        
        # 3단계: TV-L1 정규화로 최종 정제
        print("3단계: TV-L1 정규화")
        lambda_tv = 0.01 if noise_level > 20 else 0.005
        lambda_l1 = 0.005 if noise_level > 20 else 0.001
        result3 = self.tv_l1_deconvolution(result2.astype(np.uint8), psf, lambda_tv, lambda_l1, 50)
        
        return np.clip(result3, 0, 255).astype(np.uint8)
    
    def _estimate_smart_psf(self, image, psf_size):
        """
        스마트 PSF 추정
        """
        # 여러 방법으로 PSF 추정 후 최적 선택
        methods = []
        
        # Method 1: Edge-based
        edges = cv2.Canny(image, 50, 150)
        angles1 = []
        for angle in range(0, 180, 5):
            rotated = ndimage.rotate(edges, angle, reshape=False)
            projection = np.sum(rotated, axis=0)
            angles1.append(np.var(projection))
        
        best_angle1 = np.argmax(angles1) * 5 + 90
        
        # Method 2: Gradient-based
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(grad_y, grad_x)
        hist, bins = np.histogram(angles, bins=180)
        best_angle2 = np.degrees(bins[np.argmax(hist)])
        
        # 두 방법의 평균
        final_angle = np.radians((best_angle1 + best_angle2) / 2)
        
        # 길이 추정
        blur_strength = np.std(cv2.Laplacian(image, cv2.CV_64F)) / 255.0
        length = int(max(5, min(psf_size-2, blur_strength * 25)))
        
        # PSF 생성
        psf = np.zeros((psf_size, psf_size))
        center = psf_size // 2
        
        for i in range(length):
            x = int(center + (i - length//2) * np.cos(final_angle))
            y = int(center + (i - length//2) * np.sin(final_angle))
            if 0 <= x < psf_size and 0 <= y < psf_size:
                psf[y, x] = 1.0
        
        if np.sum(psf) > 0:
            psf /= np.sum(psf)
            # 스무딩
            psf = ndimage.gaussian_filter(psf, sigma=0.5)
            psf /= np.sum(psf)
        
        return psf

def main():
    image_path = "C:/develop/vision-test/file/jijin.png"
    
    <function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Implement Non-Local Means deblurring", "status": "in_progress", "activeForm": "Implementing Non-Local Means deblurring"}, {"content": "Create TV-L1 regularized deconvolution", "status": "pending", "activeForm": "Creating TV-L1 regularized deconvolution"}, {"content": "Build BM3D-style 3D transform deblurring", "status": "pending", "activeForm": "Building BM3D-style 3D transform deblurring"}, {"content": "Implement Dark Channel Prior deblurring", "status": "pending", "activeForm": "Implementing Dark Channel Prior deblurring"}, {"content": "Create intelligent hybrid system", "status": "pending", "activeForm": "Creating intelligent hybrid system"}]