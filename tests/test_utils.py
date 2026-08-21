"""
اختبارات الأدوات المساعدة

تم التطوير بمساعدة Perplexity AI
"""

import pytest
import numpy as np
from src.utils import (
    load_phoneme_patterns,
    save_phoneme_patterns,
    create_mic_array,
    calculate_accuracy,
)


class TestCreateMicArray:
    """اختبارات create_mic_array"""
    
    def test_square_array(self):
        """اختبار مصفوفة مربعة"""
        positions = create_mic_array(type='square', size=0.02)
        
        assert positions.shape == (4, 3)
        assert np.all(np.abs(positions) <= 0.01)
    
    def test_circle_array(self):
        """اختبار مصفوفة دائرية"""
        positions = create_mic_array(type='circle', size=0.02)
        
        assert positions.shape == (4, 3)
        distances = np.linalg.norm(positions[:, :2], axis=1)
        assert np.allclose(distances, 0.01, atol=1e-6)
    
    def test_linear_array(self):
        """اختبار مصفوفة خطية"""
        positions = create_mic_array(type='linear', size=0.02)
        
        assert positions.shape == (4, 3)
        assert np.all(positions[:, 1] == 0)
        assert np.all(positions[:, 2] == 0)
    
    def test_invalid_type(self):
        """اختبار نوع غير صالح"""
        with pytest.raises(ValueError):
            create_mic_array(type='invalid')


class TestCalculateAccuracy:
    """اختبارات calculate_accuracy"""
    
    def test_perfect_accuracy(self):
        """اختبار دقة مثالية"""
        predicted = np.random.randn(100, 3)
        true = predicted.copy()
        
        metrics = calculate_accuracy(predicted, true)
        
        assert metrics['mean_error_mm'] == 0.0
        assert metrics['std_error_mm'] == 0.0
        assert metrics['median_error_mm'] == 0.0
    
    def test_constant_error(self):
        """اختبار خطأ ثابت"""
        true = np.zeros((100, 3))
        predicted = true + 0.001  # 1 mm error
    
        metrics = calculate_accuracy(predicted, true)
        
        assert np.isclose(metrics['mean_error_mm'], 1.0, atol=0.1)
        assert np.isclose(metrics['median_error_mm'], 1.0, atol=0.1)
    
    def test_random_error(self):
        """اختبار خطأ عشوائي"""
        true = np.zeros((1000, 3))
        predicted = true + np.random.randn(1000, 3) * 0.001
        
        metrics = calculate_accuracy(predicted, true)
        
        assert metrics['mean_error_mm'] > 0
        assert metrics['std_error_mm'] > 0
        assert metrics['rmse_mm'] > 0


class TestPhonemePatterns:
    """اختبارات أنماط الأحرف"""
    
    def test_load_save(self, tmp_path):
        """اختبار تحميل وحفظ"""
        patterns = {
            'ب': {'position': [0.0, 0.0, 0.050]},
            'م': {'position': [0.0, 0.0, 0.048]},
        }
        
        path = tmp_path / "test_patterns.json"
        save_phoneme_patterns(patterns, str(path))
        loaded = load_phoneme_patterns(str(path))
        
        assert loaded == patterns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])