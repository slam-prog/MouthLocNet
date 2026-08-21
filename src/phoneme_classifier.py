"""
وحدة تمييز الأحرف الصوتية

تصنيف الأحرف الصوتية بناءً على موقع الصوت:
- "ب" vs "م"
- "س" vs "ش"
- "ت" vs "د"

تم التطوير بمساعدة Perplexity AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PhonemeResult:
    """نتيجة تصنيف الحرف"""
    phoneme: str  # الحرف predicted
    confidence: float  # ثقة [0, 1]
    all_probs: Dict[str, float]  # جميع الاحتمالات


class PhonemeClassifier(nn.Module):
    """
    مصنف الأحرف الصوتية
    
    Args:
        num_phonemes: عدد الأحرف (افتراضي 10)
        embedding_dim: بعد التضمين (افتراضي 256)
    """
    
    def __init__(self, num_phonemes: int = 10, embedding_dim: int = 256):
        super().__init__()
        
        self.num_phonemes = num_phonemes
        self.phoneme_labels = [
            'ب', 'م',  # شفوي
            'س', 'ش',  # أسناني
            'ت', 'د',  # لثوي
            'ك', 'ج',  # لهوي
            'ع', 'ح',  # حلقي
        ][:num_phonemes]
        
        # Network layers
        self.fc1 = nn.Linear(3, embedding_dim)  # position -> embedding
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, num_phonemes)
        
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        
    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position: موقع [batch, 3] (x, y, z)
            
        Returns:
            logits [batch, num_phonemes]
        """
        x = F.relu(self.bn1(self.fc1(position)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def predict(self, position: np.ndarray) -> PhonemeResult:
        """
        توقع حرف من موقع
        
        Args:
            position: موقع [3] (متر)
            
        Returns:
            PhonemeResult
        """
        self.eval()
        
        # تحويل إلى tensor
        pos_tensor = torch.from_numpy(position).float().unsqueeze(0)
        
        # توقع
        with torch.no_grad():
            logits = self.forward(pos_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()
        
        # إيجاد الأعلى احتمال
        best_idx = np.argmax(probs)
        best_phoneme = self.phoneme_labels[best_idx]
        best_confidence = probs[best_idx]
        
        # جميع الاحتمالات
        all_probs = {
            label: float(prob)
            for label, prob in zip(self.phoneme_labels, probs)
        }
        
        return PhonemeResult(
            phoneme=best_phoneme,
            confidence=best_confidence,
            all_probs=all_probs
        )
    
    @classmethod
    def from_pretrained(cls, path: str, num_phonemes: int = 10) -> 'PhonemeClassifier':
        """
        تحميل نموذج مدرب
        
        Args:
            path: مسار الملف
            num_phonemes: عدد الأحرف
            
        Returns:
            نموذج مدرب
        """
        model = cls(num_phonemes=num_phonemes)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded pretrained phoneme classifier from {path}")
        return model
    
    def save_pretrained(self, path: str):
        """حفظ النموذج"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_phonemes': self.num_phonemes,
            'phoneme_labels': self.phoneme_labels
        }, path)
        print(f"✅ Saved phoneme classifier to {path}")


class PhonemeSequenceClassifier(nn.Module):
    """
    مصنف تسلسل الأحرف (للكلمات)
    
    يستخدم LSTM لتسلسل المواقع
    """
    
    def __init__(self, num_phonemes: int = 10, hidden_dim: int = 512):
        super().__init__()
        
        self.num_phonemes = num_phonemes
        
        # Position embedding
        self.pos_embedding = nn.Linear(3, 128)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification head
        self.fc = nn.Linear(hidden_dim * 2, num_phonemes)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: تسلسل مواقع [batch, seq_len, 3]
            
        Returns:
            logits [batch, seq_len, num_phonemes]
        """
        # Embedding
        x = F.relu(self.pos_embedding(positions))
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Classification
        x = self.fc(x)
        
        return x


# مثال استخدام
if __name__ == "__main__":
    # إنشاء مصنف
    classifier = PhonemeClassifier(num_phonemes=10)
    
    # مواقع محاكاة (متر)
    positions = {
        'ب': np.array([0.0, 0.0, 0.050]),  # شفاه
        'م': np.array([0.0, 0.0, 0.048]),  # شفاه
        'س': np.array([0.0, 0.0, 0.040]),  # لسان أمام
        'ش': np.array([0.0, 0.0, 0.038]),  # لسان وسط
        'ت': np.array([0.0, 0.0, 0.035]),  # لسان لثوي
        'ك': np.array([0.0, 0.0, 0.030]),  # حنك
    }
    
    # اختبار
    for phoneme, pos in positions.items():
        result = classifier.predict(pos)
        print(f"الموقع: {pos*1000} ملم → المتوقع: {result.phoneme} (ثقة: {result.confidence:.2f})")
    
    # حفظ نموذج
    classifier.save_pretrained('phoneme_classifier.pt')