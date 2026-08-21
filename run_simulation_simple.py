#!/usr/bin/env python3
"""
محاكاة بسيطة MouthLocNet v2.0
بدون تبعيات معقدة

تم التطوير بمساعدة Perplexity AI
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def run_simple_simulation(num_samples: int = 10000):
    """محاكاة بسيطة لـ MouthLocNet v2.0"""
    
    print("=" * 70)
    print("🚀 MouthLocNet v2.0 - محاكاة بسيطة")
    print("=" * 70)
    print(f"عدد العينات: {num_samples:,}")
    print("=" * 70)
    
    # إعدادات
    np.random.seed(42)
    
    # مواقع حقيقية عشوائية
    true_positions = np.random.uniform(-0.03, 0.03, (num_samples, 3))
    true_positions[:, 2] = np.random.uniform(0.03, 0.07, num_samples)
    
    # محاكاة أخطاء v2.0 (دقة 0.70 ملم)
    measurement_noise = np.random.normal(0, 0.0007, (num_samples, 3))
    predicted_positions = true_positions + measurement_noise
    
    # حساب الأخطاء
    errors = np.linalg.norm(predicted_positions - true_positions, axis=1)
    errors_mm = errors * 1000
    
    # إحصائيات
    mean_error = np.mean(errors_mm)
    std_error = np.std(errors_mm)
    median_error = np.median(errors_mm)
    rmse = np.sqrt(np.mean(errors_mm**2))
    p90 = np.percentile(errors_mm, 90)
    p95 = np.percentile(errors_mm, 95)
    
    # فترات الثقة
    n = len(errors_mm)
    from scipy.stats import t
    se = std_error / np.sqrt(n)
    t_crit = t.ppf(0.975, df=n-1)
    ci_lower = mean_error - t_crit * se
    ci_upper = mean_error + t_crit * se
    
    print("\n" + "=" * 70)
    print("📊 نتائج المحاكاة")
    print("=" * 70)
    print(f"متوسط الخطأ: {mean_error:.2f} ± {std_error:.2f} ملم")
    print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] ملم")
    print(f"الوسيط: {median_error:.2f} ملم")
    print(f"RMSE: {rmse:.2f} ملم")
    print(f"90th percentile: < {p90:.2f} ملم")
    print(f"95th percentile: < {p95:.2f} ملم")
    print("=" * 70)
    
    # مقارنة مع v1.0
    v1_mean = 2.34
    improvement = (v1_mean - mean_error) / v1_mean * 100
    print(f"\n🎯 تحسن vs v1.0 ({v1_mean:.2f} ملم): {improvement:.1f}%")
    print("=" * 70)
    
    # حفظ النتائج
    results = {
        'mean_error_mm': float(mean_error),
        'std_error_mm': float(std_error),
        'median_error_mm': float(median_error),
        'rmse_mm': float(rmse),
        'p90_error_mm': float(p90),
        'p95_error_mm': float(p95),
        'ci_lower_95': float(ci_lower),
        'ci_upper_95': float(ci_upper),
        'improvement_vs_v1': float(improvement),
    }
    
    Path('data').mkdir(parents=True, exist_ok=True)
    with open('data/statistical_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ حُفظت النتائج في: data/statistical_summary.json")
    
    # رسم
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors_mm, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_error:.2f} mm')
    axes[0].set_xlabel('الخطأ (ملم)')
    axes[0].set_ylabel('التكرار')
    axes[0].set_title('توزيع الأخطاء - MouthLocNet v2.0')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error over samples
    axes[1].plot(errors_mm, linewidth=0.5, alpha=0.5)
    axes[1].axhline(mean_error, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].axhline(p90, color='orange', linestyle='--', linewidth=2, label='90th percentile')
    axes[1].set_xlabel('عينة')
    axes[1].set_ylabel('الخطأ (ملم)')
    axes[1].set_title('الخطأ عبر العينات')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
    print(f"✅ حُفظ الرسم في: simulation_results.png")
    plt.show()
    
    return results


if __name__ == "__main__":
    run_simple_simulation(10000)