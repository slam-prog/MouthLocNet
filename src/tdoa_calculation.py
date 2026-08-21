"""
وحدة حساب TDOA (Time Difference of Arrival)

تدعم:
- Cross-Correlation
- GCC-PHAT
- SRP-PHAT

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
from scipy.signal import correlate
from scipy.fft import fft, ifft
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class TDOAResult:
    """نتيجة حساب TDOA"""
    tdoa_01: float  # بين ميك 0 و 1
    tdoa_02: float  # بين ميك 0 و 2
    tdoa_03: float  # بين ميك 0 و 3
    tdoa_12: float  # بين ميك 1 و 2
    tdoa_13: float  # بين ميك 1 و 3
    tdoa_23: float  # بين ميك 2 و 3


class TDOACalculator:
    """
    حاسبة فروق وقت الوصول
    
    Args:
        sample_rate: معدل العيّنات (Hz)
        method: طريقة الحساب ('cross_correlation', 'gcc_phat', 'srp_phat')
    """
    
    def __init__(self, sample_rate: int = 768000, method: str = 'gcc_phat'):
        self.sample_rate = sample_rate
        self.method = method
        
    def cross_correlation(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        حساب TDOA باستخدام Cross-Correlation
        
        Args:
            sig1: الإشارة الأولى
            sig2: الإشارة الثانية
            
        Returns:
            TDOA بالثواني
        """
        corr = correlate(sig1, sig2, mode='full')
        lag = np.argmax(corr) - len(sig1) + 1
        tdoa = lag / self.sample_rate
        return tdoa
    
    def gcc_phat(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        حساب TDOA باستخدام GCC-PHAT
        
        Args:
            sig1: الإشارة الأولى
            sig2: الإشارة الثانية
            
        Returns:
            TDOA بالثواني
        """
        # FFT
        SIG1 = fft(sig1)
        SIG2 = fft(sig2)
        
        # Cross-spectrum
        R = SIG1 * np.conj(SIG2)
        
        # PHAT weighting
        R = R / (np.abs(R) + 1e-10)
        
        # IFFT
        corr = ifft(R).real
        
        # Peak detection
        lag = np.argmax(corr) - len(sig1) // 2
        tdoa = lag / self.sample_rate
        
        return tdoa
    
    def calculate_all(self, audio: np.ndarray) -> TDOAResult:
        """
        حساب جميع فروق TDOA بين الميكروفونات
        
        Args:
            audio: مصفوفة صوتية (samples, 4 channels)
            
        Returns:
            TDOAResult يحتوي على جميع الفروق
        """
        # استخراج القنوات
        mic0 = audio[:, 0]
        mic1 = audio[:, 1]
        mic2 = audio[:, 2]
        mic3 = audio[:, 3]
        
        # حساب TDOA لكل زوج
        if self.method == 'cross_correlation':
            calc_func = self.cross_correlation
        elif self.method == 'gcc_phat':
            calc_func = self.gcc_phat
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        tdoa_01 = calc_func(mic0, mic1)
        tdoa_02 = calc_func(mic0, mic2)
        tdoa_03 = calc_func(mic0, mic3)
        tdoa_12 = calc_func(mic1, mic2)
        tdoa_13 = calc_func(mic1, mic3)
        tdoa_23 = calc_func(mic2, mic3)
        
        return TDOAResult(
            tdoa_01=tdoa_01,
            tdoa_02=tdoa_02,
            tdoa_03=tdoa_03,
            tdoa_12=tdoa_12,
            tdoa_13=tdoa_13,
            tdoa_23=tdoa_23
        )


# مثال استخدام
if __name__ == "__main__":
    # محاكاة صوت
    sample_rate = 768000
    duration = 0.01
    samples = int(sample_rate * duration)
    
    audio = np.random.randn(samples, 4)
    
    calc = TDOACalculator(sample_rate=sample_rate, method='gcc_phat')
    result = calc.calculate_all(audio)
    
    print(f"TDOA 0-1: {result.tdoa_01*1e6:.2f} μs")
    print(f"TDOA 0-2: {result.tdoa_02*1e6:.2f} μs")
    print(f"TDOA 0-3: {result.tdoa_03*1e6:.2f} μs")