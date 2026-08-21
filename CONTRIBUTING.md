# 🤝 المساهمة في MouthLocNet

شكرًا لاهتمامك بالمساهمة في MouthLocNet!

---

## 🌟 كيف تساهم؟

### 1. الإبلاغ عن مشاكل

- افتح [Issue](https://github.com/slam-prog/MouthLocNet/issues)
- وصف المشكلة بالتفصيل
- أضف خطوات لإعادة الإنتاج

### 2. اقتراح ميزات

- افتح [Issue](https://github.com/slam-prog/MouthLocNet/issues)
- وصف الميزة المقترحة
- اذكر الفائدة المتوقعة

### 3. تحسين الكود

- Fork المشروع
- أنشئ فرعًا جديدًا
- قدم Pull Request

### 4. تحسين التوثيق

- صحّح أخطاء إملائية
- أضف أمثلة
- حسّن الشروحات

---

## 📋 إرشادات المساهمة

### 1. Fork و Clone

```bash
git clone https://github.com/YOUR_USERNAME/MouthLocNet.git
cd MouthLocNet
```

### 2. إنشاء فرع

```bash
git checkout -b feature/your-feature-name
```

### 3. تطوير

```bash
# تثبيت متطلبات التطوير
pip install -e ".[dev]"

# تشغيل اختبارات
pytest tests/

# تنسيق الكود
black src/ tests/
flake8 src/ tests/
```

### 4. Commit

```bash
git add .
git commit -m "feat: إضافة ميزة جديدة"
```

### 5. Push و Pull Request

```bash
git push origin feature/your-feature-name
```

ثم افتح [Pull Request](https://github.com/slam-prog/MouthLocNet/pulls)

---

## 📝 معايير الكود

### 1. التنسيق

- استخدم `black` لتنسيق الكود
- طول السطر: 100 حرف
- استخدم 4 مسافات للـ indentation

### 2. التوثيق

```python
def my_function(arg1, arg2):
    """
    وصف الوظيفة
    
    Args:
        arg1: وصف
        arg2: وصف
    
    Returns:
        وصف العائد
    """
    pass
```

### 3. الاختبارات

```python
def test_my_function():
    """اختبار الوظيفة"""
    result = my_function(1, 2)
    assert result == 3
```

### 4. التسمية

- استخدم أسماء واضحة
- تجنب الاختصارات غير الواضحة
- استخدم English للكود

---

## 🎯 مجالات المساهمة

### 1. خوارزميات

- تحسين دقة التحديد
- تقليل latency
- تحسين الضوضاء

### 2. نماذج DL

- بنية جديدة
- تدريب أفضل
- تحسين الأداء

### 3. تطبيقات

- تكامل مع ASR
- تطبيقات طبية
- سماعات ذكية

### 4. توثيق

- شروحات
- أمثلة
- ترجمة

---

## 📞 التواصل

- [Issues](https://github.com/slam-prog/MouthLocNet/issues)
- [Discussions](https://github.com/slam-prog/MouthLocNet/discussions)

---

## 🙏 المساهمون

شكرًا لجميع المساهمين! 🎉

<a href="https://github.com/slam-prog/MouthLocNet/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=slam-prog/MouthLocNet" />
</a>

---

## 📄 الترخيص

بالمساهمة، فإنك توافق على ترخيص مساهمتك بموجب HEUL v2.0.