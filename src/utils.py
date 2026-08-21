"""
وحدة الأدوات المساعدة

وظائف مساعدة لـ:
- تحميل وحفظ الصوت
- تصور المواقع
- حساب الدقة
- معالجة البيانات

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import json
from pathlib import Path


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """
    تحميل ملف صوتي
    
    Args:
        path: مسار الملف
        
    Returns:
        (audio, sample_rate)
    """
    audio, sample_rate = sf.read(path)
    print(f"✅ Loaded audio: {path} ({len(audio)/sample_rate:.3f} s, {sample_rate} Hz)")
    return audio, sample_rate


def save_audio(path: str, audio: np.ndarray, sample_rate: int):
    """
    حفظ ملف صوتي
    
    Args:
        path: مسار الملف
        audio: مصفوفة صوتية
        sample_rate: معدل العيّنات
    """
    sf.write(path, audio, sample_rate)
    print(f"✅ Saved audio: {path}")


def visualize_position(
    position: np.ndarray,
    mic_positions: np.ndarray,
    true_position: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    تصور موقع الصوت والميكروفونات
    
    Args:
        position: الموقع المقدر [x, y, z]
        mic_positions: مواقع الميكروفونات [4, 3]
        true_position: الموقع الحقيقي (اختياري)
        save_path: مسار الحفظ (اختياري)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # ميكروفونات
    ax.scatter(
        mic_positions[:, 0] * 1000,
        mic_positions[:, 1] * 1000,
        mic_positions[:, 2] * 1000,
        c='red',
        s=100,
        label='ميكروفونات',
        marker='o'
    )
    
    # موقع مقدر
    ax.scatter(
        position[0] * 1000,
        position[1] * 1000,
        position[2] * 1000,
        c='blue',
        s=150,
        label='موقع مقدر',
        marker='^'
    )
    
    # موقع حقيقي
    if true_position is not None:
        ax.scatter(
            true_position[0] * 1000,
            true_position[1] * 1000,
            true_position[2] * 1000,
            c='green',
            s=150,
            label='موقع حقيقي',
            marker='*'
        )
        
        # خط الخطأ
        ax.plot(
            [position[0] * 1000, true_position[0] * 1000],
            [position[1] * 1000, true_position[1] * 1000],
            [position[2] * 1000, true_position[2] * 1000],
            'k--',
            linewidth=2,
            label=f'خطأ: {np.linalg.norm(position - true_position)*1000:.2f} ملم'
        )
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('تحديد موقع الصوت من الفم')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization: {save_path}")
    
    plt.show()


def calculate_accuracy(
    predicted_positions: np.ndarray,
    true_positions: np.ndarray
) -> Dict[str, float]:
    """
    حساب مقاييس الدقة
    
    Args:
        predicted_positions: مواقع مقدرة [N, 3]
        true_positions: مواقع حقيقية [N, 3]
        
    Returns:
        dict يحتوي على مقاييس الدقة
    """
    # أخطاء
    errors = np.linalg.norm(predicted_positions - true_positions, axis=1)
    
    # إحصائيات
    metrics = {
        'mean_error_mm': float(np.mean(errors) * 1000),
        'std_error_mm': float(np.std(errors) * 1000),
        'median_error_mm': float(np.median(errors) * 1000),
        'min_error_mm': float(np.min(errors) * 1000),
        'max_error_mm': float(np.max(errors) * 1000),
        'rmse_mm': float(np.sqrt(np.mean(errors**2)) * 1000),
        'p90_error_mm': float(np.percentile(errors, 90) * 1000),
        'p95_error_mm': float(np.percentile(errors, 95) * 1000),
        'p99_error_mm': float(np.percentile(errors, 99) * 1000),
    }
    
    return metrics


def load_phoneme_patterns(path: str = 'data/phoneme_patterns.json') -> Dict:
    """
    تحميل أنماط الأحرف الصوتية
    
    Args:
        path: مسار الملف
        
    Returns:
        dict يحتوي على الأنماط
    """
    with open(path, 'r', encoding='utf-8') as f:
        patterns = json.load(f)
    print(f"✅ Loaded phoneme patterns: {path}")
    return patterns


def save_phoneme_patterns(patterns: Dict, path: str = 'data/phoneme_patterns.json'):
    """
    حفظ أنماط الأحرف الصوتية
    
    Args:
        patterns: الأنماط
        path: مسار الملف
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved phoneme patterns: {path}")


def create_mic_array(
    type: str = 'square',
    size: float = 0.02
) -> np.ndarray:
    """
    إنشاء مصفوفة ميكروفونات
    
    Args:
        type: نوع المصفوفة ('square', 'circle', 'linear')
        size: حجم المصفوفة (متر)
        
    Returns:
        مواقع الميكروفونات [4, 3]
    """
    if type == 'square':
        # مربع
        positions = np.array([
            [ size/2,  size/2, 0],
            [-size/2,  size/2, 0],
            [-size/2, -size/2, 0],
            [ size/2, -size/2, 0],
        ])
    elif type == 'circle':
        # دائرة
        angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        positions = np.array([
            [size/2 * np.cos(a), size/2 * np.sin(a), 0]
            for a in angles
        ])
    elif type == 'linear':
        # خط
        positions = np.array([
            [-size/2, 0, 0],
            [-size/6, 0, 0],
            [ size/6, 0, 0],
            [ size/2, 0, 0],
        ])
    else:
        raise ValueError(f"Unknown type: {type}")
    
    return positions


def plot_error_distribution(errors: np.ndarray, save_path: Optional[str] = None):
    """
    رسم توزيع الأخطاء
    
    Args:
        errors: أخطاء [N] (متر)
        save_path: مسار الحفظ (اختياري)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    errors_mm = errors * 1000
    
    # 1. Histogram
    axes[0, 0].hist(errors_mm, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(errors_mm), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(errors_mm):.2f} mm')
    axes[0, 0].set_xlabel('الخطأ (ملم)')
    axes[0, 0].set_ylabel('التكرار')
    axes[0, 0].set_title('توزيع الأخطاء')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box Plot
    axes[0, 1].boxplot(errors_mm, vert=True, patch_artist=True)
    axes[0, 1].set_ylabel('الخطأ (ملم)')
    axes[0, 1].set_title('مخطط الصندوق')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. CDF
    sorted_errors = np.sort(errors_mm)
    cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    axes[1, 0].plot(sorted_errors, cdf, linewidth=2, color='blue')
    axes[1, 0].axhline(0.9, color='red', linestyle='--', linewidth=2, label='90th percentile')
    axes[1, 0].axhline(0.95, color='green', linestyle='--', linewidth=2, label='95th percentile')
    axes[1, 0].set_xlabel('الخطأ (ملم)')
    axes[1, 0].set_ylabel('CDF')
    axes[1, 0].set_title('التوزيع التراكمي')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error over samples
    axes[1, 1].plot(errors_mm, linewidth=1, alpha=0.5)
    axes[1, 1].axhline(np.mean(errors_mm), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].set_xlabel('عينة')
    axes[1, 1].set_ylabel('الخطأ (ملم)')
    axes[1, 1].set_title('الخطأ عبر العينات')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved error distribution: {save_path}")
    
    plt.show()


# مثال استخدام
if __name__ == "__main__":
    # تحميل صوت
    # audio, sr = load_audio('sample.wav')
    
    # إنشاء مصفوفة ميكروفونات
    mic_positions = create_mic_array(type='square', size=0.02)
    print(f"Mic positions: {mic_positions * 1000} ملم")
    
    # تحميل أنماط
    # patterns = load_phoneme_patterns()
    
    # حساب دقة
    predicted = np.random.randn(100, 3) * 0.002
    true = np.zeros((100, 3))
    metrics = calculate_accuracy(predicted, true)
    print(f"Accuracy metrics: {metrics}")
    
    # رسم توزيع أخطاء
    errors = np.linalg.norm(predicted - true, axis=1)
    # plot_error_distribution(errors, 'error_distribution.png')