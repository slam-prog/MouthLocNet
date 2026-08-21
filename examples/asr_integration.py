#!/usr/bin/env python3
"""
مثال: دمج مع ASR (Automatic Speech Recognition)

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
from mouthlocnet import MouthLocNet, PhonemeClassifier


class ASRWithMouthLoc:
    """
    نظام ASR مع MouthLocNet
    
    يحسن دقة ASR باستخدام معلومات موقع الصوت
    """
    
    def __init__(self, asr_model, mouthloc_model):
        self.asr_model = asr_model
        self.mouthloc_model = mouthloc_model
        self.phoneme_classifier = PhonemeClassifier.from_pretrained(
            'models/phoneme_classifier.pt'
        )
    
    def transcribe(self, audio: np.ndarray) -> dict:
        """
        نسخ صوت مع تحسين MouthLoc
        
        Args:
            audio: صوت [samples, channels]
            
        Returns:
            dict يحتوي على النص والموقع والأحرف
        """
        # 1. تحديد الموقع
        position = self.mouthloc_model.localize(audio)
        
        # 2. تصنيف الحرف
        phoneme_result = self.phoneme_classifier.predict(position)
        
        # 3. نسخ ASR (محاكاة)
        # (في الواقع استخدم نموذج ASR حقيقي)
        text = self._simulate_asr(audio, phoneme_result.phoneme)
        
        return {
            'text': text,
            'position_mm': (position * 1000).tolist(),
            'phoneme': phoneme_result.phoneme,
            'phoneme_confidence': phoneme_result.confidence,
            'all_phoneme_probs': phoneme_result.all_probs
        }
    
    def _simulate_asr(self, audio: np.ndarray, phoneme: str) -> str:
        """محاكاة ASR (استبدل بنموذج حقيقي)"""
        # (هذا مجرد مثال - استخدم نموذج ASR حقيقي)
        return f"[{phoneme}] ..."


def main():
    """مثال دمج ASR"""
    
    print("=" * 60)
    print("MouthLocNet - دمج مع ASR")
    print("=" * 60)
    
    # 1. تحميل نماذج
    print("\n📥 تحميل نماذج...")
    mouthloc_model = MouthLocNet.from_pretrained('models/mouthloc_net_v2.pt')
    
    # (في الواقع حمل نموذج ASR حقيقي)
    class DummyASR:
        pass
    
    asr_model = DummyASR()
    print("✅ تم تحميل النماذج")
    
    # 2. إنشاء نظام
    print("\n⚙️ إنشاء نظام ASR...")
    system = ASRWithMouthLoc(asr_model, mouthloc_model)
    print("✅ تم إنشاء النظام")
    
    # 3. محاكاة صوت
    print("\n🎤 محاكاة صوت...")
    audio = np.random.randn(7680, 4).astype(np.float32)
    
    # 4. نسخ
    print("\n📝 نسخ...")
    result = system.transcribe(audio)
    
    print(f"\n✅ النص: {result['text']}")
    print(f"✅ الموقع: {result['position_mm']} ملم")
    print(f"✅ الحرف: {result['phoneme']} (ثقة: {result['phoneme_confidence']:.2f})")
    
    print("\n" + "=" * 60)
    print("✅ اكتمل مثال ASR!")
    print("=" * 60)


if __name__ == "__main__":
    main()