#!/usr/bin/env python3
"""
مثال: تطبيق طبي (تشخيص اضطرابات النطق)

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
from mouthlocnet import MouthLocNet, PhonemeClassifier
from datetime import datetime


class SpeechDisorderAnalyzer:
    """
    محلل اضطرابات النطق
    
    يكشف عن:
    - التأتأة
    - اضطرابات النطق
    - مشاكل المخارج
    """
    
    def __init__(self, model_path: str):
        self.model = MouthLocNet.from_pretrained(model_path)
        self.phoneme_classifier = PhonemeClassifier.from_pretrained(
            'models/phoneme_classifier.pt'
        )
        
        # قواعد الأحرف
        self.phoneme_rules = {
            'ب': {'expected_z': 0.050, 'tolerance': 0.005},
            'م': {'expected_z': 0.048, 'tolerance': 0.005},
            'س': {'expected_z': 0.040, 'tolerance': 0.005},
            'ش': {'expected_z': 0.038, 'tolerance': 0.005},
            'ت': {'expected_z': 0.035, 'tolerance': 0.005},
            'ك': {'expected_z': 0.030, 'tolerance': 0.005},
        }
    
    def analyze(self, audio: np.ndarray, expected_phoneme: str) -> dict:
        """
        تحليل صوت لاكتشاف اضطرابات
        
        Args:
            audio: صوت
            expected_phoneme: الحرف المتوقع
            
        Returns:
            dict يحتوي على نتائج التحليل
        """
        # 1. تحديد الموقع
        position = self.model.localize(audio)
        
        # 2. تصنيف الحرف الفعلي
        phoneme_result = self.phoneme_classifier.predict(position)
        
        # 3. تحليل
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'expected_phoneme': expected_phoneme,
            'detected_phoneme': phoneme_result.phoneme,
            'position_mm': (position * 1000).tolist(),
            'phoneme_confidence': phoneme_result.confidence,
            'is_correct': phoneme_result.phoneme == expected_phoneme,
            'position_error_mm': None,
            'disorder_detected': False,
            'recommendations': []
        }
        
        # 4. التحقق من القواعد
        if expected_phoneme in self.phoneme_rules:
            rule = self.phoneme_rules[expected_phoneme]
            expected_z = rule['expected_z'] * 1000  # mm
            actual_z = position[2] * 1000  # mm
            
            position_error = abs(actual_z - expected_z)
            analysis['position_error_mm'] = position_error
            
            if position_error > rule['tolerance'] * 1000:
                analysis['disorder_detected'] = True
                analysis['recommendations'].append(
                    f"موقع {expected_phoneme} غير دقيق (خطأ: {position_error:.2f} ملم)"
                )
        
        if phoneme_result.phoneme != expected_phoneme:
            analysis['disorder_detected'] = True
            analysis['recommendations'].append(
                f"الحرف المكتشف ({phoneme_result.phoneme}) لا يطابق المتوقع ({expected_phoneme})"
            )
        
        if phoneme_result.confidence < 0.7:
            analysis['recommendations'].append(
                "ثقة منخفضة - قد يحتاج إلى مزيد من التدريب"
            )
        
        return analysis


def main():
    """مثال تطبيق طبي"""
    
    print("=" * 70)
    print("MouthLocNet - تطبيق طبي: تحليل اضطرابات النطق")
    print("=" * 70)
    
    # 1. تحميل نماذج
    print("\n📥 تحميل نماذج...")
    analyzer = SpeechDisorderAnalyzer('models/mouthloc_net_v2.pt')
    print("✅ تم تحميل النماذج")
    
    # 2. محاكاة أصوات
    print("\n🎤 محاكاة أصوات...")
    
    test_cases = [
        ('ب', np.random.randn(7680, 4).astype(np.float32)),
        ('م', np.random.randn(7680, 4).astype(np.float32)),
        ('س', np.random.randn(7680, 4).astype(np.float32)),
    ]
    
    # 3. تحليل
    print("\n🔍 تحليل...")
    
    for expected_phoneme, audio in test_cases:
        result = analyzer.analyze(audio, expected_phoneme)
        
        print(f"\n{'='*50}")
        print(f"الحرف المتوقع: {expected_phoneme}")
        print(f"الحرف المكتشف: {result['detected_phoneme']}")
        print(f"الموقع: {result['position_mm']} ملم")
        print(f"الثقة: {result['phoneme_confidence']:.2f}")
        print(f"صحيح: {'✅' if result['is_correct'] else '❌'}")
        
        if result['disorder_detected']:
            print(f"⚠️ اضطراب محتمل:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
        else:
            print("✅ لا اضطرابات")
    
    print("\n" + "=" * 70)
    print("✅ اكتمل المثال الطبي!")
    print("=" * 70)


if __name__ == "__main__":
    main()