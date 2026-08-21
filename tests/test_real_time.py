"""
اختبارات المعالجة في الوقت الفعلي

تم التطوير بمساعدة Perplexity AI
"""

import pytest
import numpy as np
import time
from src.real_time_processor import RealTimeProcessor, RTConfig, RTResult
from src.deep_learning_model import MouthLocNet, ModelConfig


class TestRTConfig:
    """اختبارات RTConfig"""
    
    def test_default_config(self):
        """اختبار التكوين الافتراضي"""
        config = RTConfig()
        
        assert config.buffer_size == 768
        assert config.num_buffers == 10
        assert config.sample_rate == 768000
        assert config.num_channels == 4
        assert config.latency_target == 0.0001


class TestRTResult:
    """اختبارات RTResult"""
    
    def test_create(self):
        """اختبار إنشاء RTResult"""
        result = RTResult(
            position=np.array([0.01, 0.02, 0.05]),
            phoneme='ب',
            confidence=0.95,
            processing_time=0.05,
            timestamp=time.time()
        )
        
        assert result.position.shape == (3,)
        assert result.phoneme == 'ب'
        assert 0 <= result.confidence <= 1
        assert result.processing_time >= 0


class TestRealTimeProcessor:
    """اختبارات RealTimeProcessor"""
    
    def setup_method(self):
        """إعداد قبل كل اختبار"""
        config = ModelConfig()
        self.model = MouthLocNet(config)
        self.processor = RealTimeProcessor(self.model)
    
    def test_start_stop(self):
        """اختبار بدء وإيقاف"""
        def dummy_callback(result):
            pass
        
        self.processor.start(dummy_callback)
        assert self.processor.is_running
        
        self.processor.stop()
        assert not self.processor.is_running
    
    def test_add_audio(self):
        """اختبار إضافة صوت"""
        audio = np.random.randn(768, 4).astype(np.float32)
        
        self.processor.add_audio(audio)
        assert len(self.processor.buffer) == 1
    
    def test_process_audio(self):
        """اختبار معالجة صوت"""
        audio = np.random.randn(768, 4).astype(np.float32)
        
        result = self.processor._process_audio(audio)
        
        assert isinstance(result, RTResult)
        assert result.position.shape == (3,)
        assert result.processing_time >= 0
    
    def test_statistics(self):
        """اختبار الإحصائيات"""
        # إضافة بعض البيانات
        for i in range(10):
            audio = np.random.randn(768, 4).astype(np.float32)
            self.processor.add_audio(audio)
            result = self.processor._process_audio(audio)
            self.processor.latency_history.append(result.processing_time)
        
        stats = self.processor.get_statistics()
        
        assert 'mean_latency_ms' in stats
        assert 'std_latency_ms' in stats
        assert 'min_latency_ms' in stats
        assert 'max_latency_ms' in stats
        assert 'p95_latency_ms' in stats
        assert 'p99_latency_ms' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])