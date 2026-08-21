"""
وحدة دمج المستشعرات

دمج:
- Audio (4 ميكروفونات)
- IMU (تتبع حركة الرأس)
- Camera (تتبع حركة الشفاه)
- EMG (إشارات عضلية)

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.linalg import cholesky
import kalman_filter as kf


@dataclass
class SensorData:
    """بيانات المستشعرات"""
    audio: Optional[np.ndarray] = None  # [samples, 4]
    imu_accel: Optional[np.ndarray] = None  # [3] x, y, z
    imu_gyro: Optional[np.ndarray] = None  # [3] roll, pitch, yaw
    camera_landmarks: Optional[np.ndarray] = None  # [N, 3] نقاط الشفاه
    emg_signals: Optional[np.ndarray] = None  # [N_channels]


@dataclass
class FusedPosition:
    """الموقع المدمج"""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    confidence: float  # [0, 1]
    timestamp: float


class SensorFusion:
    """
    دمج المستشعرات باستخدام Kalman Filter
    
    Args:
        audio_weight: وزن الصوت (افتراضي 0.6)
        imu_weight: وزن IMU (افتراضي 0.2)
        camera_weight: وزن الكاميرا (افتراضي 0.15)
        emg_weight: وزن EMG (افتراضي 0.05)
    """
    
    def __init__(
        self,
        audio_weight: float = 0.6,
        imu_weight: float = 0.2,
        camera_weight: float = 0.15,
        emg_weight: float = 0.05
    ):
        self.weights = {
            'audio': audio_weight,
            'imu': imu_weight,
            'camera': camera_weight,
            'emg': emg_weight
        }
        
        # Kalman Filter state
        self.kf = self._init_kalman_filter()
        self.last_position = None
        
    def _init_kalman_filter(self) -> kf.KalmanFilter:
        """تهيئة Kalman Filter"""
        # State: [x, y, z, vx, vy, vz]
        F = np.eye(6)  # State transition
        
        # Measurement: [x, y, z]
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        
        # Process noise
        Q = np.eye(6) * 0.01
        
        # Measurement noise
        R = np.eye(3) * 0.1
        
        # Initial state
        x0 = np.zeros(6)
        P0 = np.eye(6) * 10
        
        kalman = kf.KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
        return kalman
    
    def fuse(self, sensor_data: SensorData, dt: float = 0.01) -> FusedPosition:
        """
        دمج بيانات المستشعرات
        
        Args:
            sensor_data: بيانات المستشعرات
            dt: فرق الوقت (ثواني)
            
        Returns:
            FusedPosition
        """
        # تحديث Kalman Filter
        self.kf.predict()
        
        # دمج القياسات
        measurements = []
        weights = []
        
        # Audio
        if sensor_data.audio is not None:
            audio_pos = self._audio_to_position(sensor_data.audio)
            measurements.append(audio_pos)
            weights.append(self.weights['audio'])
        
        # IMU
        if sensor_data.imu_accel is not None:
            imu_pos = self._imu_to_position(sensor_data.imu_accel, dt)
            measurements.append(imu_pos)
            weights.append(self.weights['imu'])
        
        # Camera
        if sensor_data.camera_landmarks is not None:
            camera_pos = self._camera_to_position(sensor_data.camera_landmarks)
            measurements.append(camera_pos)
            weights.append(self.weights['camera'])
        
        # EMG
        if sensor_data.emg_signals is not None:
            emg_pos = self._emg_to_position(sensor_data.emg_signals)
            measurements.append(emg_pos)
            weights.append(self.weights['emg'])
        
        # Weighted average
        if len(measurements) > 0:
            measurements = np.array(measurements)
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            fused_measurement = np.average(measurements, axis=0, weights=weights)
            
            # Update Kalman Filter
            self.kf.update(fused_measurement)
        
        # Extract state
        position = self.kf.x[:3]
        velocity = self.kf.x[3:]
        
        # Calculate confidence
        confidence = np.min(weights) if len(weights) > 0 else 0.0
        
        return FusedPosition(
            position=position,
            velocity=velocity,
            confidence=confidence,
            timestamp=dt
        )
    
    def _audio_to_position(self, audio: np.ndarray) -> np.ndarray:
        """تحويل صوت إلى موقع"""
        # (يمكن دمج مع MouthLocNet هنا)
        return np.zeros(3)
    
    def _imu_to_position(self, accel: np.ndarray, dt: float) -> np.ndarray:
        """تحويل IMU إلى موقع"""
        # Double integration
        if self.last_position is not None:
            velocity = accel * dt
            position = self.last_position + velocity * dt
            self.last_position = position
            return position
        else:
            self.last_position = np.zeros(3)
            return np.zeros(3)
    
    def _camera_to_position(self, landmarks: np.ndarray) -> np.ndarray:
        """تحويل نقاط الكاميرا إلى موقع"""
        # centroid of landmarks
        return np.mean(landmarks, axis=0)
    
    def _emg_to_position(self, emg: np.ndarray) -> np.ndarray:
        """تحويل EMG إلى موقع"""
        # (يتطلب نموذج مدرب)
        return np.zeros(3)


# مثال استخدام
if __name__ == "__main__":
    # إنشاء SensorFusion
    fusion = SensorFusion()
    
    # بيانات محاكاة
    sensor_data = SensorData(
        audio=np.random.randn(7680, 4),
        imu_accel=np.array([0.1, 0.0, 9.8]),
        imu_gyro=np.array([0.0, 0.0, 0.0]),
        camera_landmarks=np.random.randn(10, 3) * 0.01,
        emg_signals=np.random.randn(8)
    )
    
    # دمج
    result = fusion.fuse(sensor_data, dt=0.01)
    
    print(f"الموقع: {result.position} متر")
    print(f"السرعة: {result.velocity} م/ث")
    print(f"الثقة: {result.confidence:.2f}")