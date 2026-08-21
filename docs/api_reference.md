# 📚 المرجع API

## MouthLocNet

### `mouthlocnet.MouthLocNet`

النموذج الرئيسي لتحديد موقع الصوت.

#### `__init__(config: ModelConfig = None)`

إنشاء نموذج جديد.

**Args:**
- `config`: تكوين النموذج

**Example:**
```python
from mouthlocnet import MouthLocNet, ModelConfig

config = ModelConfig()
model = MouthLocNet(config)
```

#### `forward(audio: torch.Tensor) -> torch.Tensor`

تمرير صوت خلال النموذج.

**Args:**
- `audio`: صوت [batch, channels, samples]

**Returns:**
- موقع [batch, 3] (متر)

**Example:**
```python
audio = torch.randn(2, 4, 7680)
position = model(audio)
```

#### `localize(audio: np.ndarray) -> np.ndarray`

تحديد موقع صوت واحد.

**Args:**
- `audio`: صوت [samples, channels]

**Returns:**
- موقع [3] (متر)

**Example:**
```python
audio = np.random.randn(7680, 4)
position = model.localize(audio)
```

#### `from_pretrained(path: str) -> MouthLocNet`

تحميل نموذج مدرب.

**Args:**
- `path`: مسار الملف

**Returns:**
- نموذج مدرب

**Example:**
```python
model = MouthLocNet.from_pretrained('mouthloc_net_v2.pt')
```

---

## AudioCapture

### `mouthlocnet.AudioCapture`

التقاط الصوت من الميكروفونات.

#### `__init__(config: AudioConfig = None)`

**Args:**
- `config`: تكوين الصوت

#### `start()`

بدء التقاط الصوت.

#### `stop()`

إيقاف التقاط الصوت.

#### `get_audio(duration: float = 0.01) -> np.ndarray`

الحصول على صوت.

**Args:**
- `duration`: المدة بالثواني

**Returns:**
- صوت [samples, channels]

---

## TDOACalculator

### `mouthlocnet.TDOACalculator`

حساب فروق وقت الوصول.

#### `cross_correlation(sig1, sig2) -> float`

TDOA باستخدام Cross-Correlation.

#### `gcc_phat(sig1, sig2) -> float`

TDOA باستخدام GCC-PHAT.

#### `calculate_all(audio: np.ndarray) -> TDOAResult`

حساب جميع TDOAs.

---

## SRPPHAT

### `mouthlocnet.SRPPHAT`

خوارزمية SRP-PHAT.

#### `localize(audio: np.ndarray, grid_points: np.ndarray) -> SRPResult`

تحديد موقع الصوت.

**Args:**
- `audio`: صوت
- `grid_points`: نقاط الشبكة

**Returns:**
- SRPResult

---

## RelativePatternMatcher

### `mouthlocnet.RelativePatternMatcher`

مطابقة النمط النسبي.

#### `match_pattern(measured_tdoas, candidate_positions) -> RelativePatternResult`

مطابقة نمط.

---

## PhonemeClassifier

### `mouthlocnet.PhonemeClassifier`

مصنف الأحرف الصوتية.

#### `predict(position: np.ndarray) -> PhonemeResult`

توقع حرف من موقع.

---

## SensorFusion

### `mouthlocnet.SensorFusion`

دمج المستشعرات.

#### `fuse(sensor_data: SensorData, dt: float) -> FusedPosition`

دمج بيانات المستشعرات.

---

## RealTimeProcessor

### `mouthlocnet.RealTimeProcessor`

معالجة في الوقت الفعلي.

#### `start(callback: Callable)`

بدء المعالجة.

#### `stop()`

إيقاف المعالجة.

#### `get_statistics() -> dict`

الحصول على إحصائيات.

---

## Utilities

### `mouthlocnet.load_audio(path: str)`

تحميل صوت.

### `mouthlocnet.save_audio(path, audio, sample_rate)`

حفظ صوت.

### `mouthlocnet.visualize_position(position, mic_positions, true_position)`

تصور موقع.

### `mouthlocnet.calculate_accuracy(predicted, true) -> dict`

حساب الدقة.

---

## التالي

- [دليل التثبيت](installation.md)
- [دليل التعلم العميق](deep_learning_guide.md)
- [دليل النشر](deployment_guide.md)