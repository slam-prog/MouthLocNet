"""
تحميل النماذج المدربة

تحميل نماذج MouthLocNet و PhonemeClassifier من:
- Zenodo
- Hugging Face
- Google Drive

تم التطوير بمساعدة Perplexity AI
"""

import torch
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import os


def download_from_url(url: str, save_path: str):
    """
    تحميل ملف من URL
    
    Args:
        url: رابط التحميل
        save_path: مسار الحفظ
    """
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f, tqdm(
        desc=Path(save_path).name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    
    print(f"✅ Downloaded: {save_path}")


def download_pretrained_models(version: str = 'v2.0'):
    """
    تحميل جميع النماذج المدربة
    
    Args:
        version: إصدار النماذج
    """
    models_dir = Path(__file__).parent
    
    # URLs (أمثلة)
    urls = {
        'mouthloc_net_v2.pt': 'https://zenodo.org/record/12345678/files/mouthloc_net_v2.pt',
        'phoneme_classifier.pt': 'https://zenodo.org/record/12345678/files/phoneme_classifier.pt',
        'denoising_autoencoder.pt': 'https://zenodo.org/record/12345678/files/denoising_autoencoder.pt',
    }
    
    for filename, url in urls.items():
        save_path = models_dir / filename
        if not save_path.exists():
            download_from_url(url, str(save_path))
        else:
            print(f"✅ Already exists: {save_path}")
    
    print("\n🎉 All pretrained models downloaded!")


if __name__ == "__main__":
    download_pretrained_models(version='v2.0')