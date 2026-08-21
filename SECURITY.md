# 🔒 الأمان

## الإبلاغ عن ثغرات

للإبلاغ عن ثغرة أمنية:

1. **لا تفتح Issue عام**
2. راسل المشرفين مباشرة
3. انتظر الرد

---

## الممارسات الآمنة

### 1. المفاتيح والبيانات الحساسة

```bash
# ❌ لا تضع مفاتيح في الكود
API_KEY = "sk-xxxx"

# ✅ استخدم متغيرات البيئة
import os
API_KEY = os.getenv("API_KEY")
```

### 2. Dependencies

```bash
# تحديث منتظم
pip install --upgrade -r requirements.txt

# فحص ثغرات
pip-audit
```

### 3. Input Validation

```python
# ✅ تحقق من المدخلات
def process_audio(audio: np.ndarray):
    assert audio.shape == 4, "Expected 4 channels"
    assert audio.dtype == np.float32, "Expected float32"
```

---

## التحديثات الأمنية

| الإصدار | التاريخ | الثغرات |
|---------|---------|---------|
| 2.0.0 | 2026-08-21 | - |

---

## الاتصال

للأسئلة الأمنية:
- security@example.com

---

## الشكر

شكرًا للمبلغين عن الثغرات!