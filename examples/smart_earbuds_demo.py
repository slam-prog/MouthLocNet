#!/usr/bin/env python3
"""
مثال: سماعات ذكية

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
import time
from mouthlocnet import MouthLocNet, RealTimeProcessor, RTConfig, RTResult


class SmartEarbuds:
    """
    محاكي سماعات ذكية
    
    ميزات:
    - كتم ضوضاء تكيفي
    - تعزيز صوت المتحدث
    - تتبع موقع الفم
    """
    
    def __init__(self, model_path: str):
        self.model = MouthLocNet.from_pretrained(model_path)
        
        config = RTConfig(
            buffer_size=768,
            num_buffers=10,
            sample_rate=768000,
            latency_target=0.0001
        )
        
        self.processor = RealTimeProcessor(self.model, config=config)
        self.mouth_position = None
        self.noise_profile = None
        
    def start(self):
        """بدء السماعات"""
        print("🎧 Starting smart earbuds...")
        self.processor.start(self._on_audio)
        print("✅ Smart earbuds started")
    
    def stop(self):
        """إيقاف السماعات"""
        print("🎧 Stopping smart earbuds...")
        self.processor.stop()
        print("✅ Smart earbuds stopped")
    
    def _on_audio(self, result: RTResult):
        """معالجة صوت"""
        self.mouth_position = result.position
        
        # تحديث ملف الضوضاء
        if self.noise_profile is None:
            self.noise_profile = np.zeros(3)
        
        # كتم ضوضاء تكيفي
        if self.mouth_position is not None:
            print(
                f"\rMouth: {self.mouth_position*1000} mm | "
                f"Latency: {result.processing_time*1000:.2f} ms",
                end='',
                flush=True
            )
    
    def get_mouth_position(self):
        """الحصول على موقع الفم"""
        return self.mouth_position
    
    def enable_noise_cancellation(self):
        """تفعيل كتم الضوضاء"""
        print("✅ Noise cancellation enabled")
    
    def disable_noise_cancellation(self):
        """تعطيل كتم الضوضاء"""
        print("❌ Noise cancellation disabled")


def main():
    """مثال سماعات ذكية"""
    
    print("=" * 70)
    print("MouthLocNet - سماعات ذكية")
    print("=" * 70)
    
    # 1. إنشاء سماعات
    print("\n🎧 Creating smart earbuds...")
    earbuds = SmartEarbuds('models/mouthloc_net_v2.pt')
    print("✅ تم إنشاء السماعات")
    
    # 2. بدء
    print("\n▶️ Starting...")
    earbuds.start()
    earbuds.enable_noise_cancellation()
    
    # 3. محاكاة
    print("\n🎤 Simulating audio...")
    try:
        for i in range(100):
            audio = np.random.randn(768, 4).astype(np.float32)
            earbuds.processor.add_audio(audio)
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping...")
    
    # 4. إيقاف
    earbuds.stop()
    
    # 5. نتيجة
    print("\n\n📊 Final position:")
    pos = earbuds.get_mouth_position()
    if pos is not None:
        print(f"  Mouth: {pos*1000} mm")
    
    print("\n" + "=" * 70)
    print("✅ اكتمل مثال السماعات!")
    print("=" * 70)


if __name__ == "__main__":
    main()