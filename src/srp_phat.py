"""
وحدة SRP-PHAT (Steered Response Power - Phase Transform)

خوارزمية متقدمة لتحديد موقع الصوت
أدق من GCC-PHAT في البيئات الصاخبة

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
from scipy.fft import fft, ifft
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class SRPResult:
    """نتيجة SRP-PHAT"""
    position: np.ndarray  # [x, y, z]
    power: float  # قوة الإشارة
    confidence: float  # ثقة النتيجة [0, 1]


class SRPPHAT:
    """
    خوارزمية SRP-PHAT لتحديد موقع الصوت
    
    Args:
        mic_positions: مواقع الميكروفونات [4, 3]
        sample_rate: معدل العيّنات (Hz)
        speed_of_sound: سرعة الصوت (m/s)
    """
    
    def __init__(
        self,
        mic_positions: np.ndarray,
        sample_rate: int = 768000,
        speed_of_sound: float = 343.0
    ):
        self.mic_positions = mic_positions
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound
        self.num_mics = len(mic_positions)
        
    def calculate_steered_response(
        self,
        audio: np.ndarray,
        grid_point: np.ndarray
    ) -> float:
        """
        حساب الاستجابة الموجهة لنقطة معينة
        
        Args:
            audio: مصفوفة صوتية [samples, channels]
            grid_point: النقطة المستهدفة [x, y, z]
            
        Returns:
            قوة الاستجابة
        """
        # حساب المسافات من النقطة إلى كل ميكروفون
        distances = np.linalg.norm(self.mic_positions - grid_point, axis=1)
        
        # حساب أوقات الوصول
        arrival_times = distances / self.speed_of_sound
        
        # FFT للإشارات
        audio_fft = fft(audio, axis=0)
        num_samples = audio.shape[0]
        freqs = np.fft.fftfreq(num_samples, 1/self.sample_rate)
        
        # حساب SRP-PHAT
        srp_power = 0.0
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                # فرق الوقت بين ميك i و j
                time_diff = arrival_times[i] - arrival_times[j]
                
                # Phase shift
                phase_shift = np.exp(-2j * np.pi * freqs * time_diff)
                
                # Cross-spectrum مع PHAT
                R = audio_fft[:, i] * np.conj(audio_fft[:, j])
                R_phat = R / (np.abs(R) + 1e-10)
                
                # تطبيق phase shift
                R_shifted = R_phat * phase_shift
                
                # جمع القوة
                srp_power += np.sum(np.real(R_shifted))
        
        return srp_power
    
    def localize(
        self,
        audio: np.ndarray,
        grid_points: np.ndarray
    ) -> SRPResult:
        """
        تحديد موقع الصوت
        
        Args:
            audio: مصفوفة صوتية [samples, channels]
            grid_points: نقاط الشبكة [N, 3]
            
        Returns:
            SRPResult يحتوي على الموقع
        """
        # حساب الاستجابة لكل نقطة
        powers = np.zeros(len(grid_points))
        for i, point in enumerate(grid_points):
            powers[i] = self.calculate_steered_response(audio, point)
        
        # إيجاد النقطة الأقوى
        best_idx = np.argmax(powers)
        best_position = grid_points[best_idx]
        best_power = powers[best_idx]
        
        # حساب الثقة
        confidence = (best_power - np.mean(powers)) / (np.std(powers) + 1e-10)
        confidence = np.clip(confidence / 10, 0, 1)  # تطبيع
        
        return SRPResult(
            position=best_position,
            power=best_power,
            confidence=confidence
        )
    
    def create_grid(
        self,
        center: np.ndarray,
        size: float = 0.1,
        resolution: float = 0.001
    ) -> np.ndarray:
        """
        إنشاء شبكة نقاط للبحث
        
        Args:
            center: مركز الشبكة [x, y, z]
            size: حجم الشبكة (متر)
            resolution: دقة الشبكة (متر)
            
        Returns:
            مصفوفة نقاط [N, 3]
        """
        # إنشاء شبكة 3D
        x = np.arange(center[0] - size/2, center[0] + size/2, resolution)
        y = np.arange(center[1] - size/2, center[1] + size/2, resolution)
        z = np.arange(center[2] - size/2, center[2] + size/2, resolution)
        
        # إنشاء جميع النقاط
        grid_points = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T
        
        return grid_points


# مثال استخدام
if __name__ == "__main__":
    # مواقع الميكروفونات (متر)
    mic_positions = np.array([
        [0.01, 0.0, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, -0.01, 0.0]
    ])
    
    # إنشاء SRP-PHAT
    srp = SRPPHAT(mic_positions, sample_rate=768000)
    
    # إنشاء شبكة بحث
    center = np.array([0.0, 0.0, 0.05])  # 5 سم أمام الميكروفونات
    grid = srp.create_grid(center, size=0.05, resolution=0.001)
    
    # محاكاة صوت
    samples = 7680  # 10 ms
    audio = np.random.randn(samples, 4)
    
    # تحديد الموقع
    result = srp.localize(audio, grid)
    
    print(f"الموقع: x={result.position[0]*1000:.2f}, y={result.position[1]*1000:.2f}, z={result.position[2]*1000:.2f} ملم")
    print(f"الثقة: {result.confidence:.2f}")