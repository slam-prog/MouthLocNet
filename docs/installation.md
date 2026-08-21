# 📦 التثبيت

## المتطلبات

- Python 3.8+
- pip
- Git

---

## التثبيت السريع

### 1. استنساخ المشروع

```bash
git clone https://github.com/slam-prog/MouthLocNet.git
cd MouthLocNet
```

### 2. إنشاء بيئة افتراضية

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. تثبيت المتطلبات

```bash
pip install -r requirements.txt
```

### 4. تثبيت الحزمة

```bash
pip install -e .
```

### 5. تحميل النماذج المدربة

```bash
python -m models.download_pretrained
```

---

## التحقق من التثبيت

```bash
python -c "import mouthlocnet; print(mouthlocnet.__version__)"
```

يجب أن يطبع: `2.0.0`

---

## التثبيت للتطوير

```bash
# تثبيت متطلبات التطوير
pip install -e ".[dev]"

# تثبيت متطلبات GPU (اختياري)
pip install -e ".[gpu]"
```

---

## اختبار التثبيت

```bash
# تشغيل اختبارات
pytest tests/

# تشغيل محاكاة سريعة
python -m src.simulation --demo
```

---

## استكشاف الأخطاء

### مشكلة: `ModuleNotFoundError: No module named 'mouthlocnet'`

**الحل:**
```bash
pip install -e .
```

### مشكلة: `CUDA out of memory`

**الحل:**
```bash
# استخدام CPU فقط
export CUDA_VISIBLE_DEVICES=""
```

### مشكلة: `libsndfile not found`

**الحل:**
```bash
# Linux
sudo apt-get install libsndfile1

# Mac
brew install libsndfile

# Windows
# تحميل من https://www.mega-nerd.com/libsndfile/
```

---

## التحديث

```bash
git pull origin main
pip install -e .
python -m models.download_pretrained
```

---

## التالي

- [دليل الاستخدام السريع](../README.md#-استخدام-سريع)
- [المرجع API](api_reference.md)
- [دليل التعلم العميق](deep_learning_guide.md)