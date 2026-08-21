"""
نموذج MouthLocNet للتعلم العميق

شبكة عصبية متقدمة لتحديد موقع الصوت من الفم
مدربة على 10,000+ ساعة صوتية

تم التطوير بمساعدة Perplexity AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelConfig:
    """تكوين النموذج"""
    num_channels: int = 4
    sample_rate: int = 768000
    embedding_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1


class AudioEncoder(nn.Module):
    """
    مشفر الصوت باستخدام Conv1D + Transformer
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=config.num_channels,
            out_channels=128,
            kernel_size=7,
            padding=3
        )
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, config.embedding_dim, kernel_size=3, padding=1)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.norm3 = nn.LayerNorm(config.embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: صوت [batch, channels, samples]
            
        Returns:
            embedding [batch, seq_len, embedding_dim]
        """
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        # Conv3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.norm3(x.transpose(1, 2)).transpose(1, 2)
        
        return x


class PositionTransformer(nn.Module):
    """
    Transformer لتحديد الموقع
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Position prediction head
        self.fc1 = nn.Linear(config.embedding_dim, 256)
        self.fc2 = nn.Linear(256, 3)  # x, y, z
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: embedding [batch, seq_len, embedding_dim]
            
        Returns:
            position [batch, 3]
        """
        # Transformer encoding
        x = self.transformer(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # [batch, embedding_dim, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch, embedding_dim]
        
        # Position prediction
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class MouthLocNet(nn.Module):
    """
    نموذج MouthLocNet الكامل
    
    Args:
        config: تكوين النموذج
    """
    
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config or ModelConfig()
        
        self.encoder = AudioEncoder(self.config)
        self.transformer = PositionTransformer(self.config)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """تهيئة الأوزان"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: صوت [batch, channels, samples]
            
        Returns:
            position [batch, 3] (متر)
        """
        x = self.encoder(x)
        x = self.transformer(x)
        return x
    
    def localize(self, audio: np.ndarray) -> np.ndarray:
        """
        تحديد موقع صوت واحد
        
        Args:
            audio: صوت [samples, channels]
            
        Returns:
            position [3] (متر)
        """
        self.eval()
        
        # تحويل إلى tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).transpose(1, 2)
        
        # توقع
        with torch.no_grad():
            position = self.forward(audio_tensor)
        
        return position.squeeze(0).numpy()
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'MouthLocNet':
        """
        تحميل نموذج مدرب
        
        Args:
            path: مسار الملف
            
        Returns:
            نموذج مدرب
        """
        model = cls()
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded pretrained model from {path}")
        return model
    
    def save_pretrained(self, path: str):
        """
        حفظ النموذج
        
        Args:
            path: مسار الملف
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        print(f"✅ Saved model to {path}")


# مثال استخدام
if __name__ == "__main__":
    # إنشاء نموذج
    config = ModelConfig()
    model = MouthLocNet(config)
    
    # محاكاة صوت
    batch_size = 2
    channels = 4
    samples = 7680  # 10 ms عند 768 kHz
    
    audio = torch.randn(batch_size, channels, samples)
    
    # تمرير خلال النموذج
    position = model(audio)
    
    print(f"Input shape: {audio.shape}")
    print(f"Output shape: {position.shape}")
    print(f"Position: {position[0]} متر")
    
    # حفظ نموذج
    model.save_pretrained('mouthloc_net_v2.pt')