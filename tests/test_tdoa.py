"""
اختبارات TDOA

تم التطوير بمساعدة Perplexity AI
"""

import pytest
import numpy as np
from src.tdoa_calculation import TDOACalculator, TDOAResult


class TestTDOACalculator:
    """اختبارات TDOACalculator"""
    
    def setup_method(self):
        """إعداد قبل كل اختبار"""
        self.sample_rate = 768000
        self.calc = TDOACalculator(sample_rate=self.sample_rate, method='gcc_phat')
    
    def test_cross_correlation_zero_delay(self):
        """اختبار cross-correlation مع تأخير صفر"""
        # إشارة متطابقة
        signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, int(0.01 * self.sample_rate)))
        
        tdoa = self.calc.cross_correlation(signal, signal)
        
        assert abs(tdoa) < 1e-6, "TDOA يجب أن يكون صفر لإشارات متطابقة"
    
    def test_cross_correlation_known_delay(self):
        """اختبار cross-correlation مع تأخير معروف"""
        # إنشاء إشارة
        t = np.linspace(0, 0.01, int(0.01 * self.sample_rate))
        signal1 = np.sin(2 * np.pi * 1000 * t)
        
        # تأخير 100 عينة
        delay_samples = 100
        signal2 = np.roll(signal1, delay_samples)
        
        tdoa = self.calc.cross_correlation(signal1, signal2)
        expected_tdoa = delay_samples / self.sample_rate
        
        assert abs(tdoa - expected_tdoa) < 2e-6, f"TDOA يجب أن يكون {expected_tdoa}"
    
    def test_gcc_phat(self):
        """اختبار GCC-PHAT"""
        t = np.linspace(0, 0.01, int(0.01 * self.sample_rate))
        signal1 = np.sin(2 * np.pi * 1000 * t)
        
        delay_samples = 50
        signal2 = np.roll(signal1, delay_samples)
        
        tdoa = self.calc.gcc_phat(signal1, signal2)
        expected_tdoa = delay_samples / self.sample_rate
        
        assert abs(tdoa - expected_tdoa) < 2e-6
    
    def test_calculate_all(self):
        """اختبار حساب جميع TDOAs"""
        # محاكاة صوت 4 قنوات
        samples = int(0.01 * self.sample_rate)
        audio = np.random.randn(samples, 4)
        
        result = self.calc.calculate_all(audio)
        
        assert isinstance(result, TDOAResult)
        assert hasattr(result, 'tdoa_01')
        assert hasattr(result, 'tdoa_02')
        assert hasattr(result, 'tdoa_03')
        assert hasattr(result, 'tdoa_12')
        assert hasattr(result, 'tdoa_13')
        assert hasattr(result, 'tdoa_23')
    
    def test_tdoa_with_noise(self):
        """اختبار TDOA مع ضوضاء"""
        t = np.linspace(0, 0.01, int(0.01 * self.sample_rate))
        signal1 = np.sin(2 * np.pi * 1000 * t)
        
        delay_samples = 75
        signal2 = np.roll(signal1, delay_samples)
        
        # إضافة ضوضاء
        noise1 = np.random.randn(len(signal1)) * 0.1
        noise2 = np.random.randn(len(signal2)) * 0.1
        
        signal1_noisy = signal1 + noise1
        signal2_noisy = signal2 + noise2
        
        tdoa = self.calc.gcc_phat(signal1_noisy, signal2_noisy)
        expected_tdoa = delay_samples / self.sample_rate
        
        # مع ضوضاء، الخطأ يجب أن يكون < 10 عينات
        assert abs(tdoa - expected_tdoa) < 10 / self.sample_rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])