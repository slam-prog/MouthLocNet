# 🚀 دليل النشر

## نظرة عامة

نشر MouthLocNet في بيئات إنتاجية مختلفة.

---

## 1. النشر على خادم

### المتطلبات

- Python 3.8+
- 4 GB RAM
- GPU (اختياري)

### الإعداد

```bash
# تثبيت
pip install mouthlocnet

# تحميل نماذج
python -m models.download_pretrained
```

### استخدام

```python
from mouthlocnet import MouthLocNet

model = MouthLocNet.from_pretrained('v2.0')

def process_audio(audio_file):
    audio, sr = sf.read(audio_file)
    position = model.localize(audio)
    return position * 1000  # mm
```

---

## 2. النشر كـ API

### FastAPI

```python
from fastapi import FastAPI, File, UploadFile
from mouthlocnet import MouthLocNet
import soundfile as sf
import numpy as np

app = FastAPI()
model = MouthLocNet.from_pretrained('v2.0')

@app.post("/localize")
async def localize(file: UploadFile = File(...)):
    # قراءة صوت
    audio = sf.read(file.file)
    audio = np.frombuffer(audio, dtype=np.float32).reshape(-1, 4)
    
    # توقع
    position = model.localize(audio)
    
    return {
        "position_mm": (position * 1000).tolist(),
        "confidence": 0.95
    }
```

### تشغيل

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### اختبار

```bash
curl -X POST http://localhost:8000/localize \
  -F "file=@audio.wav"
```

---

## 3. النشر على جهاز طرفي (Edge)

### Raspberry Pi

```bash
# تثبيت
pip install mouthlocnet

# استخدام CPU فقط
export CUDA_VISIBLE_DEVICES=""

# تشغيل
python example.py
```

### Jetson Nano

```bash
# تثبيت مع GPU
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu102

# تشغيل
python example.py
```

---

## 4. النشر كـ مكتبة

### setup.py

```python
from setuptools import setup

setup(
    name='mouthlocnet',
    version='2.0.0',
    packages=['mouthlocnet'],
    install_requires=[
        'numpy>=1.24.0',
        'torch>=2.0.0',
        'soundfile>=0.12.0',
    ],
)
```

### نشر على PyPI

```bash
# بناء
python setup.py sdist bdist_wheel

# نشر
twine upload dist/*
```

### استخدام

```bash
pip install mouthlocnet
```

```python
import mouthlocnet
model = mouthlocnet.MouthLocNet.from_pretrained('v2.0')
```

---

## 5. النشر على FPGA

### VHDL Code

```vhdl
-- MouthLocNet FPGA Implementation
entity MouthLocNet is
    port (
        clk : in std_logic;
        audio_in : in std_logic_vector(15 downto 0);
        position_out : out std_logic_vector(31 downto 0)
    );
end entity;
```

### Synthesis

```bash
# Xilinx Vivado
vivado -mode batch -source synth.tcl
```

---

## 6. التحسين للأداء

### Quantization

```python
import torch

# Dynamic quantization
model_quantized = torch.quantization.quantize_dynamic(
    model,
    [(nn.Linear, nn.ReLU)],
    dtype=torch.qint8
)

# Save
torch.save(model_quantized.state_dict(), 'mouthloc_net_quantized.pt')
```

### ONNX Export

```python
import torch
import onnx

# Export
dummy_input = torch.randn(1, 4, 7680)
torch.onnx.export(
    model,
    dummy_input,
    'mouthloc_net.onnx',
    opset_version=11,
    input_names=['audio'],
    output_names=['position']
)

# Run with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession('mouthloc_net.onnx')
position = session.run(None, {'audio': dummy_input.numpy()})
```

---

## 7. المراقبة

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process(audio):
    logger.info(f'Processing audio: {audio.shape}')
    position = model.localize(audio)
    logger.info(f'Position: {position}')
    return position
```

### Metrics

```python
from prometheus_client import Counter, Histogram

REQUESTS = Counter('mouthlocnet_requests', 'Total requests')
LATENCY = Histogram('mouthlocnet_latency', 'Request latency')

@LATENCY.time()
def process(audio):
    REQUESTS.inc()
    return model.localize(audio)
```

---

## 8. الأمان

### Rate Limiting

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.post("/localize")
@limiter.limit("100 per minute")
def localize():
    # ...
```

### Authentication

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/localize")
async def localize(token: str = Depends(security)):
    # Verify token
    # ...
```

---

## التالي

- [دليل التثبيت](installation.md)
- [المرجع API](api_reference.md)
- [دليل التعلم العميق](deep_learning_guide.md)