"""
اختبارات دمج المستشعرات

تم التطوير بمساعدة Perplexity AI
"""

import pytest
import numpy as np
from src.sensor_fusion import SensorFusion, SensorData, FusedPosition


class TestSensorData:
    """اختبارات SensorData"""
    
    def test_default_values(self):
        """اختبار القيم الافتراضية"""
        data = SensorData()
        
        assert data.audio is None
        assert data.imu_accel is None
        assert data.imu_gyro is None
        assert data.camera_landmarks is None
        assert data.emg_signals is None
    
    def test_create_with_data(self):
        """اختبار إنشاء ببيانات"""
        data = SensorData(
            audio=np.random.randn(7680, 4),
            imu_accel=np.array([0.1, 0.0, 9.8]),
            camera_landmarks=np.random.randn(10, 3)
        )
        
        assert data.audio.shape == (7680, 4)
        assert data.imu_accel.shape == (3,)
        assert data.camera_landmarks.shape == (10, 3)


class TestFusedPosition:
    """اختبارات FusedPosition"""
    
    def test_create(self):
        """اختبار إنشاء FusedPosition"""
        pos = FusedPosition(
            position=np.array([0.01, 0.02, 0.05]),
            velocity=np.array([0.0, 0.0, 0.0]),
            confidence=0.95,
            timestamp=1234567890.0
        )
        
        assert pos.position.shape == (3,)
        assert pos.velocity.shape == (3,)
        assert 0 <= pos.confidence <= 1


class TestSensorFusion:
    """اختبارات SensorFusion"""
    
    def setup_method(self):
        """إعداد قبل كل اختبار"""
        self.fusion = SensorFusion(
            audio_weight=0.6,
            imu_weight=0.2,
            camera_weight=0.15,
            emg_weight=0.05
        )
    
    def test_fuse_audio_only(self):
        """اختبار دمج صوت فقط"""
        sensor_data = SensorData(
            audio=np.random.randn(7680, 4)
        )
        
        result = self.fusion.fuse(sensor_data, dt=0.01)
        
        assert isinstance(result, FusedPosition)
        assert result.position.shape == (3,)
    
    def test_fuse_all_sensors(self):
        """اختبار دمج جميع المستشعرات"""
        sensor_data = SensorData(
            audio=np.random.randn(7680, 4),
            imu_accel=np.array([0.1, 0.0, 9.8]),
            imu_gyro=np.array([0.0, 0.0, 0.0]),
            camera_landmarks=np.random.randn(10, 3) * 0.01,
            emg_signals=np.random.randn(8)
        )
        
        result = self.fusion.fuse(sensor_data, dt=0.01)
        
        assert isinstance(result, FusedPosition)
        assert result.position.shape == (3,)
        assert result.velocity.shape == (3,)
        assert 0 <= result.confidence <= 1
    
    def test_weights_sum_to_one(self):
        """اختبار أن الأوزان مجموعها 1"""
        total_weight = (
            self.fusion.weights['audio'] +
            self.fusion.weights['imu'] +
            self.fusion.weights['camera'] +
            self.fusion.weights['emg']
        )
        
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_fuse_empty_data(self):
        """اختبار دمج بدون بيانات"""
        sensor_data = SensorData()
        
        result = self.fusion.fuse(sensor_data, dt=0.01)
        
        assert isinstance(result, FusedPosition)
        # يجب أن يعطي قيمة افتراضية
    
    def test_multiple_fusions(self):
        """اختبار دمجات متعددة"""
        positions = []
        
        for i in range(10):
            sensor_data = SensorData(
                audio=np.random.randn(7680, 4) * (i + 1)
            )
            result = self.fusion.fuse(sensor_data, dt=0.01)
            positions.append(result.position.copy())
        
        # المواقع يجب أن تكون مختلفة
        positions = np.array(positions)
        assert np.std(positions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])