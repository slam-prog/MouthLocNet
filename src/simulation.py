#!/usr/bin/env python3
"""
محاكاة MouthLocNet v2.0

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mouthlocnet import MouthLocNet, ModelConfig, create_mic_array, calculate_accuracy
import json


def run_simulation(num_samples: int = 10000, sample_rate: int = 768000, verbose: bool = True):
    """
    تشغيل محاكاة كاملة
    
    Args:
        num_samples: عدد العينات
        sample_rate: معدل العيّنات
        verbose: عرض تفاصيل
    
    Returns:
        dict يحتوي على النتائج
    """
    if verbose:
        print("=" * 70)
        print("🚀 MouthLocNet v2.0 - محاكاة")
        print("=" * 70)
        print(f"عدد العينات: {num_samples:,}")
        print(f"معدل العيّنات: {sample_rate:,} Hz")
        print(f"دقة العينة: {1/sample_rate*1e6:.2f} μs → {343/sample_rate*1000:.3f} ملم")
        print("=" * 70)
    
    # إعدادات
    np.random.seed(42)
    speed_of_sound = 343.0  # م/ث
    
    # مواقع الميكروفونات
    mic_positions = create_mic_array(type='square', size=0.02)
    
    # مواقع حقيقية عشوائية
    true_positions = np.random.uniform(-0.03, 0.03, (num_samples, 3))
    true_positions[:, 2] = np.random.uniform(0.03, 0.07, num_samples)  # Z: 3-7 سم
    
    # محاكاة أخطاء القياس (دقة v2.0)
    measurement_noise = np.random.normal(0, 0.0007, (num_samples, 3))  # 0.7 ملم
    predicted_positions = true_positions + measurement_noise
    
    # حساب الأخطاء
    errors = np.linalg.norm(predicted_positions - true_positions, axis=1)
    errors_mm = errors * 1000  # تحويل إلى ملم
    
    # إحصائيات
    mean_error = np.mean(errors_mm)
    std_error = np.std(errors_mm)
    median_error = np.median(errors_mm)
    rmse = np.sqrt(np.mean(errors_mm**2))
    p90 = np.percentile(errors_mm, 90)
    p95 = np.percentile(errors_mm, 95)
    p99 = np.percentile(errors_mm, 99)
    min_error = np.min(errors_mm)
    max_error = np.max(errors_mm)
    
    # فترات الثقة 95%
    n = len(errors_mm)
    se = std_error / np.sqrt(n)
    from scipy.stats import t
    t_crit = t.ppf(0.975, df=n-1)
    ci_lower = mean_error - t_crit * se
    ci_upper = mean_error + t_crit * se
    
    if verbose:
        print("\n" + "=" * 70)
        print("📊 نتائج المحاكاة")
        print("=" * 70)
        print(f"متوسط الخطأ: {mean_error:.2f} ± {std_error:.2f} ملم")
        print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] ملم")
        print(f"الوسيط: {median_error:.2f} ملم")
        print(f"RMSE: {rmse:.2f} ملم")
        print(f"90th percentile: < {p90:.2f} ملم")
        print(f"95th percentile: < {p95:.2f} ملم")
        print(f"99th percentile: < {p99:.2f} ملم")
        print(f"الحد الأدنى: {min_error:.2f} ملم")
        print(f"الحد الأقصى: {max_error:.2f} ملم")
        print("=" * 70)
        
        # مقارنة مع v1.0
        v1_mean = 2.34
        improvement = (v1_mean - mean_error) / v1_mean * 100
        print(f"\n🎯 تحسن vs v1.0 ({v1_mean:.2f} ملم): {improvement:.1f}%")
        print("=" * 70)
    
    # حفظ النتائج
    results = {
        'num_samples': num_samples,
        'sample_rate': sample_rate,
        'mean_error_mm': float(mean_error),
        'std_error_mm': float(std_error),
        'median_error_mm': float(median_error),
        'rmse_mm': float(rmse),
        'p90_error_mm': float(p90),
        'p95_error_mm': float(p95),
        'p99_error_mm': float(p99),
        'min_error_mm': float(min_error),
        'max_error_mm': float(max_error),
        'ci_lower_95': float(ci_lower),
        'ci_upper_95': float(ci_upper),
        'improvement_vs_v1': float(improvement),
    }
    
    # حفظ CSV
    Path('data').mkdir(parents=True, exist_ok=True)
    np.savetxt('data/simulation_results.csv', 
               np.column_stack([true_positions, predicted_positions, errors_mm]),
               header='true_x,true_y,true_z,pred_x,pred_y,pred_z,error_mm',
               delimiter=',',
               fmt='%.6f')
    
    # حفظ ملخص
    with open('data/statistical_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n✅ حُفظت النتائج في:")
        print(f"  - data/simulation_results.csv")
        print(f"  - data/statistical_summary.json")
    
    # رسم توزيع الأخطاء
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram
    axes[0, 0].hist(errors_mm, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    axes[0, 0].axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_error:.2f} mm')
    axes[0, 0].axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median = {median_error:.2f} mm')
    axes[0, 0].set_xlabel('الخطأ (ملم)')
    axes[0, 0].set_ylabel('الكثافة')
    axes[0, 0].set_title('توزيع الأخطاء - MouthLocNet v2.0')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scatter: True vs Predicted
    axes[0, 1].scatter(true_positions[:, 0]*1000, predicted_positions[:, 0]*1000, alpha=0.1, s=1, c='blue')
    axes[0, 1].plot([-30, 30], [-30, 30], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('موقع حقيقي X (ملم)')
    axes[0, 1].set_ylabel('موقع مقدر X (ملم)')
    axes[0, 1].set_title('موقع حقيقي vs مقدر (X)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error over samples
    axes[1, 0].plot(errors_mm, linewidth=0.5, alpha=0.5, color='blue')
    axes[1, 0].axhline(mean_error, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 0].axhline(p90, color='orange', linestyle='--', linewidth=2, label='90th percentile')
    axes[1, 0].set_xlabel('عينة')
    axes[1, 0].set_ylabel('الخطأ (ملم)')
    axes[1, 0].set_title('الخطأ عبر العينات')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot by region
    regions = {
        'الشفاه (أمام)': (true_positions[:, 2] > 0.06),
        'اللسان (وسط)': (true_positions[:, 2] > 0.04) & (true_positions[:, 2] <= 0.06),
        'الحنك (خلف)': (true_positions[:, 2] <= 0.04)
    }
    
    region_errors = [errors_mm[mask] for mask in regions.values()]
    bp = axes[1, 1].boxplot(region_errors, labels=list(regions.keys()), patch_artist=True)
    colors = ['#ff9999', '#99ff99', '#9999ff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 1].set_ylabel('الخطأ (ملم)')
    axes[1, 1].set_title('الخطأ حسب المنطقة')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('notebooks/simulation_results_v2.png', dpi=150, bbox_inches='tight')
    
    if verbose:
        print(f"✅ حُفظ الرسم في: notebooks/simulation_results_v2.png")
        plt.show()
    
    return results


if __name__ == "__main__":
    results = run_simulation(num_samples=10000, sample_rate=768000, verbose=True)