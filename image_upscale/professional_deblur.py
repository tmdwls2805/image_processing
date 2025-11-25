import cv2
import numpy as np
from scipy import ndimage, signal, fft
from scipy.optimize import minimize
import os

class ProfessionalDeblur:
    def __init__(self):
        pass
    
    def estimate_psf_blind(self, blurred_img, psf_size=21):
        """
        Blind PSF estimation using gradient-based approach
        """
        # Convert to grayscale if needed
        if len(blurred_img.shape) == 3:
            gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = blurred_img.copy()
        
        # Normalize image
        gray = gray.astype(np.float64) / 255.0
        
        # Calculate gradients
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find dominant gradient direction
        angles = np.arctan2(grad_y, grad_x)
        
        # Histogram of gradient angles
        angle_hist, bins = np.histogram(angles.flatten(), bins=180, range=(-np.pi, np.pi))
        
        # Find peak angle (motion direction)
        peak_angle_idx = np.argmax(angle_hist)
        motion_angle = bins[peak_angle_idx]
        
        # Create linear motion PSF
        psf = np.zeros((psf_size, psf_size))
        center = psf_size // 2
        
        # Length estimation based on blur severity
        blur_strength = np.std(grad_mag)
        motion_length = min(max(int(blur_strength * 30), 5), psf_size - 2)
        
        # Draw line in PSF
        for i in range(motion_length):
            x = int(center + (i - motion_length//2) * np.cos(motion_angle))
            y = int(center + (i - motion_length//2) * np.sin(motion_angle))
            
            if 0 <= x < psf_size and 0 <= y < psf_size:
                psf[y, x] = 1.0
        
        # Normalize PSF
        if np.sum(psf) > 0:
            psf /= np.sum(psf)
        else:
            # Fallback to simple horizontal motion
            psf[center, center-motion_length//2:center+motion_length//2+1] = 1.0
            psf /= np.sum(psf)
        
        return psf, motion_angle, motion_length
    
    def lucy_richardson_advanced(self, image, psf, iterations=50, damping=0.9):
        """
        Advanced Lucy-Richardson deconvolution with damping
        """
        # Ensure float64 for precision
        image = image.astype(np.float64)
        psf = psf.astype(np.float64)
        
        # Initialize estimate
        estimate = image.copy()
        psf_flipped = np.flipud(np.fliplr(psf))
        
        # Store previous estimate for damping
        prev_estimate = estimate.copy()
        
        for i in range(iterations):
            # Forward convolution
            conv_estimate = signal.convolve2d(estimate, psf, mode='same', boundary='wrap')
            
            # Avoid division by zero
            conv_estimate = np.maximum(conv_estimate, 1e-10)
            
            # Calculate ratio
            ratio = image / conv_estimate
            
            # Backward convolution
            correction = signal.convolve2d(ratio, psf_flipped, mode='same', boundary='wrap')
            
            # Update estimate with damping
            new_estimate = estimate * correction
            estimate = damping * new_estimate + (1 - damping) * prev_estimate
            
            # Ensure non-negative values
            estimate = np.maximum(estimate, 0)
            
            # Damping factor decay
            if i > 10:
                damping = max(damping * 0.99, 0.7)
            
            prev_estimate = estimate.copy()
            
            # Progress tracking
            if i % 10 == 0:
                print(f"  반복 {i+1}/{iterations}")
        
        return estimate
    
    def frequency_domain_deblur(self, image, psf, noise_factor=0.01):
        """
        Frequency domain deblurring (Wiener filter)
        """
        # Convert to frequency domain
        img_fft = fft.fft2(image)
        psf_fft = fft.fft2(psf, s=image.shape)
        
        # Wiener filter
        psf_conj = np.conj(psf_fft)
        psf_abs2 = np.abs(psf_fft) ** 2
        
        wiener_filter = psf_conj / (psf_abs2 + noise_factor)
        
        # Apply filter
        restored_fft = img_fft * wiener_filter
        
        # Convert back to spatial domain
        restored = fft.ifft2(restored_fft)
        restored = np.real(restored)
        
        return restored
    
    def total_variation_deblur(self, blurred, psf, lambda_reg=0.02, iterations=100):
        """
        Total Variation regularized deblurring
        """
        def tv_norm(x):
            """Total variation norm"""
            diff_x = np.diff(x, axis=1)
            diff_y = np.diff(x, axis=0)
            return np.sum(np.sqrt(diff_x[:-1, :]**2 + diff_y[:, :-1]**2))
        
        def objective(x_flat):
            x = x_flat.reshape(blurred.shape)
            
            # Data fidelity term
            conv_x = signal.convolve2d(x, psf, mode='same', boundary='wrap')
            data_term = np.sum((conv_x - blurred)**2)
            
            # Total variation regularization
            tv_term = tv_norm(x)
            
            return data_term + lambda_reg * tv_term
        
        # Initialize with blurred image
        x0 = blurred.flatten()
        
        # Optimize
        print("  Total Variation 최적화 중...")
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': iterations, 'disp': False})
        
        restored = result.x.reshape(blurred.shape)
        return np.maximum(restored, 0)  # Ensure non-negative
    
    def edge_preserving_deblur(self, image, psf):
        """
        Edge-preserving deblurring using bilateral filtering
        """
        # Initial deblurring
        initial = self.lucy_richardson_advanced(image, psf, iterations=20)
        
        # Edge-preserving smoothing
        if len(image.shape) == 3:
            # Color image
            preserved = np.zeros_like(initial)
            for i in range(3):
                channel = initial[:, :, i].astype(np.float32)
                # Normalize to 0-1 for bilateral filter
                channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
                filtered = cv2.bilateralFilter(channel_norm, 9, 0.1, 10)
                # Restore original range
                preserved[:, :, i] = filtered * (channel.max() - channel.min()) + channel.min()
        else:
            # Grayscale
            image_norm = (initial - initial.min()) / (initial.max() - initial.min() + 1e-8)
            preserved = cv2.bilateralFilter(image_norm.astype(np.float32), 9, 0.1, 10)
            preserved = preserved * (initial.max() - initial.min()) + initial.min()
        
        return preserved
    
    def multi_scale_deblur(self, image, psf_size=21):
        """
        Multi-scale deblurring approach
        """
        # Create image pyramid
        pyramid = [image.astype(np.float64)]
        temp = image.copy()
        
        # Downsample
        for level in range(3):
            temp = cv2.pyrDown(temp.astype(np.float32)).astype(np.float64)
            pyramid.append(temp)
        
        # Process from coarse to fine
        results = []
        
        for level, img in enumerate(reversed(pyramid)):
            print(f"  레벨 {level+1} 처리 중...")
            
            # Estimate PSF for this scale
            scale_factor = 2 ** (len(pyramid) - level - 1)
            scaled_psf_size = max(psf_size // scale_factor, 5)
            
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = img.astype(np.uint8)
            
            psf, angle, length = self.estimate_psf_blind(gray, scaled_psf_size)
            
            # Deblur
            if len(img.shape) == 3:
                # Color image - process each channel
                deblurred = np.zeros_like(img)
                for i in range(3):
                    channel = img[:, :, i]
                    # Normalize for processing
                    channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
                    deblurred_channel = self.lucy_richardson_advanced(channel_norm, psf, iterations=20)
                    # Restore range
                    deblurred[:, :, i] = deblurred_channel * (channel.max() - channel.min()) + channel.min()
            else:
                # Grayscale
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                deblurred = self.lucy_richardson_advanced(img_norm, psf, iterations=20)
                deblurred = deblurred * (img.max() - img.min()) + img.min()
            
            results.append(deblurred)
            
            # If not the finest level, use this result to initialize next level
            if level < len(pyramid) - 1:
                next_size = pyramid[-(level+2)].shape
                if len(img.shape) == 3:
                    upsampled = np.zeros((next_size[0], next_size[1], 3))
                    for i in range(3):
                        upsampled[:, :, i] = cv2.resize(deblurred[:, :, i], 
                                                      (next_size[1], next_size[0]), 
                                                      interpolation=cv2.INTER_CUBIC)
                else:
                    upsampled = cv2.resize(deblurred, (next_size[1], next_size[0]), 
                                         interpolation=cv2.INTER_CUBIC)
                
                # Use as initialization for next level
                pyramid[-(level+2)] = 0.7 * pyramid[-(level+2)] + 0.3 * upsampled
        
        return results[-1]  # Return finest level result
    
    def process_image(self, image_path, method='multi_scale', output_path=None):
        """
        Professional deblurring pipeline
        """
        try:
            print(f"이미지 로드: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            original = image.copy()
            print(f"전문 디블러링 방법: {method}")
            
            if method == 'blind_psf':
                print("Blind PSF 추정 및 디블러링...")
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                psf, angle, length = self.estimate_psf_blind(gray)
                print(f"  추정된 모션: 각도={np.degrees(angle):.1f}°, 길이={length}px")
                
                # Apply to each channel
                if len(image.shape) == 3:
                    result = np.zeros_like(image, dtype=np.float64)
                    for i in range(3):
                        channel = image[:, :, i].astype(np.float64) / 255.0
                        deblurred = self.lucy_richardson_advanced(channel, psf)
                        result[:, :, i] = deblurred * 255.0
                else:
                    img_norm = image.astype(np.float64) / 255.0
                    result = self.lucy_richardson_advanced(img_norm, psf) * 255.0
            
            elif method == 'frequency':
                print("주파수 영역 디블러링...")
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                psf, _, _ = self.estimate_psf_blind(gray)
                
                if len(image.shape) == 3:
                    result = np.zeros_like(image, dtype=np.float64)
                    for i in range(3):
                        channel = image[:, :, i].astype(np.float64)
                        deblurred = self.frequency_domain_deblur(channel, psf)
                        result[:, :, i] = deblurred
                else:
                    result = self.frequency_domain_deblur(image.astype(np.float64), psf)
            
            elif method == 'tv_regularized':
                print("Total Variation 정규화 디블러링...")
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                psf, _, _ = self.estimate_psf_blind(gray)
                
                if len(image.shape) == 3:
                    result = np.zeros_like(image, dtype=np.float64)
                    for i in range(3):
                        channel = image[:, :, i].astype(np.float64)
                        deblurred = self.total_variation_deblur(channel, psf)
                        result[:, :, i] = deblurred
                else:
                    result = self.total_variation_deblur(image.astype(np.float64), psf)
            
            elif method == 'edge_preserving':
                print("엣지 보존 디블러링...")
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                psf, _, _ = self.estimate_psf_blind(gray)
                result = self.edge_preserving_deblur(image, psf)
            
            elif method == 'multi_scale':
                print("다중 스케일 디블러링...")
                result = self.multi_scale_deblur(image)
            
            else:
                print("알 수 없는 방법, multi_scale 사용")
                result = self.multi_scale_deblur(image)
            
            # Clamp and convert to uint8
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Post-processing enhancement
            print("후처리 개선 중...")
            
            # Contrast enhancement
            if len(result.shape) == 3:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                result = cv2.merge([l, a, b])
                result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                result = clahe.apply(result)
            
            # Gentle sharpening
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            result = cv2.filter2D(result, -1, kernel)
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Output path
            if output_path is None:
                base_name = os.path.splitext(image_path)[0]
                ext = os.path.splitext(image_path)[1]
                output_path = f"{base_name}_professional_{method}{ext}"
            
            # Save result
            cv2.imwrite(output_path, result)
            print(f"전문 디블러링 완료: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"처리 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    image_path = "C:/develop/vision-test/file/jijin.png"
    
    <function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Research advanced motion blur correction techniques", "status": "completed", "activeForm": "Researching advanced motion blur correction techniques"}, {"content": "Implement kernel estimation methods", "status": "completed", "activeForm": "Implementing kernel estimation methods"}, {"content": "Apply frequency domain deblurring", "status": "completed", "activeForm": "Applying frequency domain deblurring"}, {"content": "Test edge-preserving deconvolution", "status": "completed", "activeForm": "Testing edge-preserving deconvolution"}, {"content": "Create unified processing pipeline", "status": "in_progress", "activeForm": "Creating unified processing pipeline"}]