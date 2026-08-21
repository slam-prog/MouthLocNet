"""
اختبارات التعلم العميق

تم التطوير بمساعدة Perplexity AI
"""

import pytest
import torch
import numpy as np
from src.deep_learning_model import MouthLocNet, ModelConfig, AudioEncoder, PositionTransformer


class TestModelConfig:
    """اختبارات ModelConfig"""
    
    def test_default_config(self):
        """اختبار التكوين الافتراضي"""
        config = ModelConfig()
        
        assert config.num_channels == 4
        assert config.sample_rate == 768000
        assert config.embedding_dim == 512
        assert config.num_heads == 8
        assert config.num_layers == 6
        assert config.dropout == 0.1


class TestAudioEncoder:
    """اختبارات AudioEncoder"""
    
    def setup_method(self):
        """إعداد قبل كل اختبار"""
        self.config = ModelConfig()
        self.encoder = AudioEncoder(self.config)
    
    def test_encoder_forward(self):
        """اختبار تمرير خلال encoder"""
        # صوت: [batch=2, channels=4, samples=7680]
        x = torch.randn(2, 4, 7680)
        
        output = self.encoder(x)
        
        # Output: [batch, seq_len, embedding_dim]
        assert output.shape[0] == 2
        assert output.shape[2] == 512  # embedding_dim
    
    def test_encoder_preserves_batch(self):
        """اختبار الحفاظ على batch size"""
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 4, 7680)
            output = self.encoder(x)
            assert output.shape[0] == batch_size


class TestPositionTransformer:
    """اختبارات PositionTransformer"""
    
    def setup_method(self):
        """إعداد قبل كل اختبار"""
        self.config = ModelConfig()
        self.transformer = PositionTransformer(self.config)
    
    def test_transformer_forward(self):
        """اختبار تمرير خلال transformer"""
        # Embedding: [batch=2, seq_len=10, embedding_dim=512]
        x = torch.randn(2, 10, 512)
        
        output = self.transformer(x)
        
        # Output: [batch, 3]
        assert output.shape == (2, 3)
    
    def test_transformer_position_range(self):
        """اختبار نطاق الموقع"""
        x = torch.randn(1, 10, 512)
        
        output = self.transformer(x)
        
        # المواقع يجب أن تكون معقولة (ليست كبيرة جدًا)
        assert torch.all(torch.abs(output) < 1.0)


class TestMouthLocNet:
    """اختبارات MouthLocNet"""
    
    def setup_method(self):
        """إعداد قبل كل اختبار"""
        self.config = ModelConfig()
        self.model = MouthLocNet(self.config)
    
    def test_forward(self):
        """اختبار تمرير أمامي"""
        # صوت: [batch=2, channels=4, samples=7680]
        x = torch.randn(2, 4, 7680)
        
        output = self.model(x)
        
        # Output: [batch, 3]
        assert output.shape == (2, 3)
    
    def test_localize(self):
        """اختبار localize"""
        # صوت: [samples=7680, channels=4]
        audio = np.random.randn(7680, 4).astype(np.float32)
        
        position = self.model.localize(audio)
        
        assert position.shape == (3,)
        assert np.all(np.abs(position) < 1.0)  # معقول
    
    def test_save_load(self):
        """اختبار حفظ وتحميل نموذج"""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # حفظ
            self.model.save_pretrained(temp_path)
            
            # تحميل
            loaded_model = MouthLocNet.from_pretrained(temp_path)
            
            # اختبار
            audio = np.random.randn(7680, 4).astype(np.float32)
            pos1 = self.model.localize(audio)
            pos2 = loaded_model.localize(audio)
            
            assert np.allclose(pos1, pos2, rtol=1e-5)
        
        finally:
            os.unlink(temp_path)
    
    def test_gradient_flow(self):
        """اختبار تدفق التدرجات"""
        x = torch.randn(2, 4, 7680, requires_grad=True)
        
        output = self.model(x)
        loss = output.sum()
        
        loss.backward()
        
        # التحقق من وجود gradients
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
    
    def test_weight_initialization(self):
        """اختبار تهيئة الأوزان"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                assert torch.std(param) > 0, f"Weights not initialized for {name}"


class TestPhonemeClassifier:
    """اختبارات PhonemeClassifier"""
    
    def setup_method(self):
        """إعداد قبل كل اختبار"""
        from src.phoneme_classifier import PhonemeClassifier
        self.classifier = PhonemeClassifier(num_phonemes=10)
    
    def test_predict(self):
        """اختبار توقع حرف"""
        position = np.array([0.0, 0.0, 0.05])
        
        result = self.classifier.predict(position)
        
        assert hasattr(result, 'phoneme')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'all_probs')
        assert 0 <= result.confidence <= 1
    
    def test_batch_predict(self):
        """اختبار توقع batch"""
        positions = np.random.randn(10, 3) * 0.05
        
        for pos in positions:
            result = self.classifier.predict(pos)
            assert result.phoneme in self.classifier.phoneme_labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])