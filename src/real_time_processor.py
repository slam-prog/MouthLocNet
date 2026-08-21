"""
وحدة المعالجة في الوقت الفعلي

معالجة صوتية بزمن < 0.1 ms
مناسبة للتطبيقات الحية

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
import torch
from typing import Optional, Callable
from dataclasses import dataclass
from collections import deque
import time
import threading


@dataclass
class RTConfig:
    """تكوين المعالجة في الوقت الفعلي"""
    buffer_size: int = 768  # 1 ms عند 768 kHz
    num_buffers: int = 10
    sample_rate: int = 768000
    num_channels: int = 4
    latency_target: float = 0.0001  # 0.1 ms


@dataclass
class RTResult:
    """نتيجة المعالجة في الوقت الفعلي"""
    position: np.ndarray  # [x, y, z]
    phoneme: Optional[str]
    confidence: float
    processing_time: float  # ms
    timestamp: float


class RealTimeProcessor:
    """
    معالج في الوقت الفعلي
    
    Args:
        model: نموذج MouthLocNet
        phoneme_classifier: مصنف الأحرف (اختياري)
        config: التكوين
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        phoneme_classifier: Optional[torch.nn.Module] = None,
        config: RTConfig = None
    ):
        self.config = config or RTConfig()
        self.model = model
        self.phoneme_classifier = phoneme_classifier
        
        # Buffer
        self.buffer = deque(maxlen=self.config.num_buffers)
        self.current_buffer = np.zeros(
            (self.config.buffer_size, self.config.num_channels)
        )
        
        # State
        self.is_running = False
        self.thread = None
        self.callback: Optional[Callable[[RTResult], None]] = None
        
        # Statistics
        self.latency_history = deque(maxlen=100)
        
    def start(self, callback: Callable[[RTResult], None]):
        """
        بدء المعالجة في الوقت الفعلي
        
        Args:
            callback: دالة تُستدعى عند كل نتيجة
        """
        self.callback = callback
        self.is_running = True
        
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.daemon = True
        self.thread.start()
        
        print(f"✅ Real-time processing started (target latency: {self.config.latency_target*1000:.2f} ms)")
        
    def stop(self):
        """إيقاف المعالجة"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("✅ Real-time processing stopped")
        
    def _processing_loop(self):
        """حلقة المعالجة الرئيسية"""
        while self.is_running:
            start_time = time.perf_counter()
            
            # الحصول على صوت
            if len(self.buffer) > 0:
                audio = np.vstack(self.buffer)
                
                # معالجة
                result = self._process_audio(audio)
                
                # استدعاء callback
                if self.callback:
                    self.callback(result)
                
                # تسجيل latency
                processing_time = (time.perf_counter() - start_time) * 1000
                self.latency_history.append(processing_time)
            
            # انتظار
            time.sleep(self.config.latency_target)
    
    def _process_audio(self, audio: np.ndarray) -> RTResult:
        """
        معالجة صوت واحد
        
        Args:
            audio: صوت [samples, channels]
            
        Returns:
            RTResult
        """
        start_time = time.perf_counter()
        
        # تحويل إلى tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).transpose(1, 2)
        
        # توقع الموقع
        self.model.eval()
        with torch.no_grad():
            position = self.model(audio_tensor).squeeze(0).numpy()
        
        # توقع الحرف
        phoneme = None
        if self.phoneme_classifier:
            phoneme_result = self.phoneme_classifier.predict(position)
            phoneme = phoneme_result.phoneme
            confidence = phoneme_result.confidence
        else:
            confidence = 1.0
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return RTResult(
            position=position,
            phoneme=phoneme,
            confidence=confidence,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    def get_statistics(self) -> dict:
        """
        الحصول على إحصائيات الأداء
        
        Returns:
            dict يحتوي على latency المتوسط وغيرها
        """
        if len(self.latency_history) == 0:
            return {}
        
        latencies = list(self.latency_history)
        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
        }
    
    def add_audio(self, audio: np.ndarray):
        """
        إضافة صوت إلى buffer
        
        Args:
            audio: صوت [samples, channels]
        """
        self.buffer.append(audio)


# مثال استخدام
if __name__ == "__main__":
    # تحميل نموذج
    from mouthlocnet import MouthLocNet
    model = MouthLocNet.from_pretrained('mouthloc_net_v2.pt')
    
    # إنشاء معالج
    processor = RealTimeProcessor(model)
    
    # callback
    def on_result(result: RTResult):
        print(
            f"Position: {result.position*1000} mm, "
            f"Phoneme: {result.phoneme}, "
            f"Latency: {result.processing_time:.2f} ms"
        )
    
    # بدء
    processor.start(on_result)
    
    # محاكاة صوت
    for i in range(100):
        audio = np.random.randn(768, 4)
        processor.add_audio(audio)
        time.sleep(0.001)
    
    # إيقاف
    processor.stop()
    
    # إحصائيات
    stats = processor.get_statistics()
    print(f"\nStatistics: {stats}")