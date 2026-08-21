# 🧠 دليل التعلم العميق

## نظرة عامة

MouthLocNet يستخدم شبكة عصبية متقدمة لتحديد موقع الصوت من الفم.

---

## بنية النموذج

### MouthLocNet

Audio Input [4 channels, 7680 samples]
↓
Conv1D (128 filters, kernel=7)
↓
Conv1D (256 filters, kernel=5)
↓
Conv1D (512 filters, kernel=3)
↓
Transformer Encoder (6 layers, 8 heads)
↓
Global Average Pooling
↓
FC (256 units)
↓
FC (3 units) → Position [x, y, z]



---

## التدريب

### 1. إعداد البيانات

```python
from torch.utils.data import Dataset, DataLoader

class MouthLocDataset(Dataset):
    def __init__(self, audio_paths, positions):
        self.audio_paths = audio_paths
        self.positions = positions
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio, sr = sf.read(self.audio_paths[idx])
        position = self.positions[idx]
        
        # Convert to tensor
        audio = torch.from_numpy(audio).float().transpose(0, 1)
        position = torch.from_numpy(position).float()
        
        return audio, position
```

### 2. التدريب

```python
import torch
import torch.nn as nn
from mouthlocnet import MouthLocNet, ModelConfig

# إعدادات
config = ModelConfig()
model = MouthLocNet(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss و optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# حلقة التدريب
for epoch in range(100):
    model.train()
    total_loss = 0
    
    for audio, positions in train_loader:
        audio = audio.to(device)
        positions = positions.to(device)
        
        # Forward
        predicted = model(audio)
        
        # Loss
        loss = criterion(predicted, positions)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}')
```

---

## التقييم

```python
model.eval()
errors = []

with torch.no_grad():
    for audio, true_pos in test_loader:
        audio = audio.to(device)
        pred_pos = model(audio)
        
        error = torch.norm(pred_pos - true_pos, dim=1) * 1000  # mm
        errors.extend(error.cpu().numpy())

print(f'Mean Error: {np.mean(errors):.2f} mm')
print(f'Std Error: {np.std(errors):.2f} mm')
print(f'90th Percentile: {np.percentile(errors, 90):.2f} mm')
```

---

## التحسين

### Data Augmentation

```python
def augment_audio(audio, sr):
    # Add noise
    noise = np.random.randn(*audio.shape) * 0.01
    audio = audio + noise
    
    # Time shift
    shift = np.random.randint(-100, 100)
    audio = np.roll(audio, shift, axis=0)
    
    # Speed perturbation
    speed = np.random.uniform(0.95, 1.05)
    # ... implement resampling
    
    return audio
```

### Transfer Learning

```python
# تحميل نموذج مدرب مسبقًا
model = MouthLocNet.from_pretrained('mouthloc_net_v2.pt')

# Fine-tuning
for param in model.encoder.parameters():
    param.requires_grad = False  # تجميد encoder

# تدريب فقط على head
optimizer = torch.optim.Adam(model.transformer.parameters(), lr=1e-4)
```

---

## نصائح

### 1. معدل التعلم
- ابدأ بـ `1e-4`
- استخدم learning rate scheduler
- ReduceLROnPlateau يعمل جيدًا

### 2. Batch Size
- GPU: 32-128
- CPU: 8-16

### 3. Regularization
- Dropout: 0.1-0.3
- Weight decay: 1e-5
- Data augmentation

### 4. Early Stopping
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
```

---

## التالي

- [دليل التثبيت](installation.md)
- [المرجع API](api_reference.md)
- [دليل دمج المستشعرات](sensor_fusion_guide.md)
