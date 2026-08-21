# ❓ الأسئلة الشائعة

## عام

### ما هو MouthLocNet؟

نظام تحديد موقع الصوت من الفم باستخدام التعلم العميق بدقة 0.70 ملم.

### ما هو الترخيص؟

HEUL v2.0 (Humanitarian & Ethical Use License) - للاستخدام الإنساني والأخلاقي فقط.

### هل هو مجاني؟

نعم، مجاني للاستخدامات المسموحة (بحث، تعليم، طب غير ربحي، إلخ).

---

## تقني

### ما هي متطلبات الهاردوير؟

- 4 ميكروفونات MEMS
- ADC 768 kHz
- GPU (اختياري، للتدريب)

### ما هي دقة النظام؟

0.70 ± 0.35 ملم (متوسط خطأ)

### ما هي السرعة؟

0.1 ms في الوقت الفعلي (على FPGA)

### هل يعمل على Raspberry Pi؟

نعم، لكن على CPU فقط (أبطأ).

### هل يدعم اللغة العربية؟

نعم، يدعم جميع اللغات.

---

## استخدام

### كيف أبدأ؟

```bash
git clone https://github.com/slam-prog/MouthLocNet.git
cd MouthLocNet
pip install -r requirements.txt
python examples/basic_usage.py
```

### كيف أدرب نموذجي الخاص؟

انظر `notebooks/deep_learning_training.ipynb`

### كيف أدمج مع ASR؟

انظر `examples/asr_integration.py`

---

## تطوير

### كيف أساهم؟

انظر `CONTRIBUTING.md`

### هل هناك API؟

نعم، انظر `docs/api_reference.md`

### كيف أنشر في إنتاج؟

انظر `docs/deployment_guide.md`

---

## ترخيص

### هل يمكنني استخدام تجاريًا؟

نعم، بشرط الالتزام بالشروط الأخلاقية (انظر LICENSE).

### هل يمكنني تعديل الكود؟

نعم، التعديل مسموح.

### هل يجب أن أشارك تعديلاتي؟

لا، لكن نرحب بالمساهمات!

---

## تواصل

### كيف أتواصل؟

- Issues: https://github.com/slam-prog/MouthLocNet/issues
- Discussions: https://github.com/slam-prog/MouthLocNet/discussions