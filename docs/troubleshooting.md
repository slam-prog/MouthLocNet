# 🔧 استكشاف الأخطاء

## مشاكل شائعة

### 1. `ModuleNotFoundError: No module named 'mouthlocnet'`

**الحل:**
```bash
pip install -e .
```

### 2. `CUDA out of memory`

**الحل:**
```bash
# استخدام CPU
export CUDA_VISIBLE_DEVICES=""

# أو تقليل batch size
```

### 3. `libsndfile not found`

**الحل:**
```bash
# Linux
sudo apt-get install libsndfile1

# Mac
brew install libsndfile

# Windows
# تحميل من https://www.mega-nerd.com/libsndfile/
```

### 4. `RuntimeError: Expected all tensors to be on the same device`

**الحل:**
```python
# تأكد من أن جميع tensors على نفس الجهاز
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
audio = audio.to(device)
```

### 5. دقة منخفضة

**الحل:**
```python
# تحقق من:
# 1. معدل العيّنات (يجب أن يكون 768000)
# 2. عدد القنوات (يجب أن يكون 4)
# 3. النموذج المدرب (يجب أن يكون v2.0)

print(f"Sample rate: {config.sample_rate}")
print(f"Channels: {config.num_channels}")
```

### 6. Latency عالي

**الحل:**
```python
# استخدم RTConfig مع latency_target منخفض
config = RTConfig(
    buffer_size=768,  # 1 ms
    latency_target=0.0001  # 0.1 ms
)
```

---

## الحصول على مساعدة

- [Issues](https://github.com/slam-prog/MouthLocNet/issues)
- [Discussions](https://github.com/slam-prog/MouthLocNet/discussions)

---

## الإبلاغ عن مشكلة

عند الإبلاغ عن مشكلة، أضف:

1. وصف المشكلة
2. خطوات لإعادة الإنتاج
3. رسالة الخطأ الكاملة
4. معلومات البيئة (OS, Python, GPU)
5. لقطات شاشة إن أمكن