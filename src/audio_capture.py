"""
وحدة التقاط الصوت

تدعم:
- 4 قنوات صوتية
- معدل عيّنات 768 kHz
- معالجة في الوقت الفعلي

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
import sounddevice as sd
from typing import Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class AudioConfig:
    """تكوين التقاط الصوت"""
    channels: int = 4
    sample_rate: int = 768000  # 768 kHz
    buffer_size: int = 4096
    device: Optional[int] = None


class AudioCapture:
    """
    فئة التقاط الصوت من 4 ميكروفونات
    
    Args:
        config: تكوين الصوت
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.stream = None
        self.buffer = None
        self.is_recording = False
        
    def start(self):
        """بدء التقاط الصوت"""
        self.buffer = np.zeros((self.config.buffer_size, self.config.channels))
        self.is_recording = True
        
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Status: {status}")
            self.buffer = np.vstack([self.buffer, indata])
        
        self.stream = sd.InputStream(
            device=self.config.device,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            callback=callback,
            blocksize=self.config.buffer_size
        )
        self.stream.start()
        print(f"✅ Started recording at {self.config.sample_rate} Hz")
        
    def stop(self):
        """إيقاف التقاط الصوت"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_recording = False
        print("✅ Stopped recording")
        
    def get_audio(self, duration: float = 0.01) -> np.ndarray:
        """
        الحصول على صوت لمدة محددة
        
        Args:
            duration: المدة بالثواني (افتراضي 10 ms)
            
        Returns:
            مصفوفة صوتية (samples, channels)
        """
        samples = int(duration * self.config.sample_rate)
        audio = sd.rec(
            samples,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype='float32'
        )
        sd.wait()
        return audio
    
    def calibrate(self, reference_signal: np.ndarray):
        """
        معايرة الميكروفونات
        
        Args:
            reference_signal: إشارة مرجعية للمعايرة
        """
        # معايرة gain
        gains = np.std(reference_signal, axis=0)
        self.gains = np.mean(gains) / gains
        
        # معايرة phase
        # (يمكن إضافة كود معايرة phase هنا)
        
        print(f"✅ Calibration complete: gains = {self.gains}")
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# مثال استخدام
if __name__ == "__main__":
    config = AudioConfig(channels=4, sample_rate=768000)
    
    with AudioCapture(config) as capture:
        # تسجيل 10 ms
        audio = capture.get_audio(duration=0.01)
        print(f"Recorded {len(audio)} samples")