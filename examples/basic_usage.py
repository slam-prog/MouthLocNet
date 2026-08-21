#!/usr/bin/env python3
"""
مثال أساسي لاستخدام MouthLocNet

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
from mouthlocnet import MouthLocNet, AudioCapture, AudioConfig


def main():
    """مثال أساسي"""
    
    print("=" * 60)
    print("MouthLocNet - مثال أساسي")
    print("=" * 60)
    
    # 1. تحميل نموذج مدرب
    print("\n📥 تحميل نموذج مدرب...")
    model = MouthLocNet.from_pretrained('models/mouthloc_net_v2.pt')
    print("✅ تم تحميل النموذج")
    
    # 2. التقاط صوت (محاكاة)
    print("\n🎤 التقاط صوت...")
    config = AudioConfig(channels=4, sample_rate=768000)
    
    # محاكاة صوت (في الواقع استخدم AudioCapture)
    duration = 0.01  # 10 ms
    samples = int(duration * config.sample_rate)
    audio = np.random.randn(samples, config.channels).astype(np.float32)
    print(f"✅ تم التقاط {samples} عينة ({duration*1000:.1f} ms)")
    
    # 3. تحديد الموقع
    print("\n🎯 تحديد الموقع...")
    position = model.localize(audio)
    print(f"✅ الموقع: x={position[0]*1000:.2f}, y={position[1]*1000:.2f}, z={position[2]*1000:.2f} ملم")
    
    # 4. حفظ النتيجة
    print("\n💾 حفظ النتيجة...")
    np.save('position.npy', position)
    print("✅ تم الحفظ في position.npy")
    
    print("\n" + "=" * 60)
    print("✅ اكتمل المثال الأساسي!")
    print("=" * 60)


if __name__ == "__main__":
    main()