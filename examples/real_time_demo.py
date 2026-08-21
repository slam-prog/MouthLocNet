#!/usr/bin/env python3
"""
مثال: معالجة في الوقت الفعلي

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
import time
from mouthlocnet import MouthLocNet, RealTimeProcessor, RTConfig, RTResult


def on_result(result: RTResult):
    """Callback عند كل نتيجة"""
    print(
        f"\rPosition: x={result.position[0]*1000:6.2f}, "
        f"y={result.position[1]*1000:6.2f}, "
        f"z={result.position[2]*1000:6.2f} mm | "
        f"Phoneme: {result.phoneme or 'N/A':3s} | "
        f"Latency: {result.processing_time:.2f} ms",
        end='',
        flush=True
    )


def main():
    """مثال المعالجة في الوقت الفعلي"""
    
    print("=" * 80)
    print("MouthLocNet - معالجة في الوقت الفعلي")
    print("=" * 80)
    
    # 1. تحميل نموذج
    print("\n📥 تحميل نموذج...")
    model = MouthLocNet.from_pretrained('models/mouthloc_net_v2.pt')
    print("✅ تم تحميل النموذج")
    
    # 2. إنشاء معالج
    print("\n⚙️ إعداد معالج الوقت الفعلي...")
    config = RTConfig(
        buffer_size=768,  # 1 ms
        num_buffers=10,
        sample_rate=768000,
        latency_target=0.0001  # 0.1 ms
    )
    
    processor = RealTimeProcessor(model, config=config)
    print("✅ تم إعداد المعالج")
    
    # 3. بدء المعالجة
    print("\n🚀 بدء المعالجة...")
    processor.start(on_result)
    print("✅ بدأت المعالجة (اضغط Ctrl+C للإيقاف)")
    
    # 4. محاكاة صوت
    print("\n🎤 محاكاة صوت...")
    try:
        for i in range(1000):
            # توليد صوت عشوائي
            audio = np.random.randn(768, 4).astype(np.float32) * 0.1
            processor.add_audio(audio)
            time.sleep(0.001)  # 1 ms
    except KeyboardInterrupt:
        print("\n\n⏹️ إيقاف...")
    
    # 5. إيقاف
    print("\n⏹️ إيقاف المعالج...")
    processor.stop()
    print("✅ تم الإيقاف")
    
    # 6. إحصائيات
    print("\n📊 إحصائيات الأداء:")
    stats = processor.get_statistics()
    if stats:
        print(f"  Mean Latency: {stats['mean_latency_ms']:.2f} ms")
        print(f"  Std Latency: {stats['std_latency_ms']:.2f} ms")
        print(f"  P95 Latency: {stats['p95_latency_ms']:.2f} ms")
        print(f"  P99 Latency: {stats['p99_latency_ms']:.2f} ms")
    
    print("\n" + "=" * 80)
    print("✅ اكتمل مثال الوقت الفعلي!")
    print("=" * 80)


if __name__ == "__main__":
    main()