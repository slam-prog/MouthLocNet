#!/usr/bin/env python3
"""
واجهة سطر الأوامر لـ MouthLocNet

تم التطوير بمساعدة Perplexity AI
"""

import click
import numpy as np
import soundfile as sf
from mouthlocnet import MouthLocNet, AudioCapture, AudioConfig


@click.group()
@click.version_option(version='2.0.0')
def cli():
    """MouthLocNet - نظام تحديد موقع الصوت من الفم"""
    pass


@cli.command()
@click.option('--audio', '-a', required=True, help='ملف صوتي')
@click.option('--model', '-m', default='models/mouthloc_net_v2.pt', help='نموذج مدرب')
@click.option('--output', '-o', default=None, help='ملف المخرج')
def localize(audio, model, output):
    """تحديد موقع صوت من ملف"""
    click.echo(f"📥 Loading model: {model}")
    model = MouthLocNet.from_pretrained(model)
    
    click.echo(f"📥 Loading audio: {audio}")
    audio_data, sr = sf.read(audio)
    
    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(-1, 1)
    
    click.echo("🎯 Localizing...")
    position = model.localize(audio_data)
    
    click.echo(f"\n✅ Position: x={position[0]*1000:.2f}, y={position[1]*1000:.2f}, z={position[2]*1000:.2f} mm")
    
    if output:
        np.save(output, position)
        click.echo(f"💾 Saved to: {output}")


@cli.command()
@click.option('--model', '-m', default='models/mouthloc_net_v2.pt', help='نموذج مدرب')
@click.option('--duration', '-d', default=1.0, help='مدة التسجيل (ثواني)')
def record(model, duration):
    """تسجيل صوت وتحديد موقع"""
    click.echo(f"📥 Loading model: {model}")
    model = MouthLocNet.from_pretrained(model)
    
    click.echo(f"🎤 Recording {duration} seconds...")
    config = AudioConfig(channels=4, sample_rate=768000)
    
    with AudioCapture(config) as capture:
        audio = capture.get_audio(duration=duration)
    
    click.echo("🎯 Localizing...")
    position = model.localize(audio)
    
    click.echo(f"\n✅ Position: x={position[0]*1000:.2f}, y={position[1]*1000:.2f}, z={position[2]*1000:.2f} mm")


@cli.command()
def demo():
    """تشغيل عرض توضيحي"""
    click.echo("🎪 Running demo...")
    
    from mouthlocnet import MouthLocNet
    import numpy as np
    
    model = MouthLocNet.from_pretrained('models/mouthloc_net_v2.pt')
    
    audio = np.random.randn(7680, 4).astype(np.float32)
    position = model.localize(audio)
    
    click.echo(f"✅ Demo position: x={position[0]*1000:.2f}, y={position[1]*1000:.2f}, z={position[2]*1000:.2f} mm")


@cli.command()
def download_models():
    """تحميل النماذج المدربة"""
    click.echo("📥 Downloading pretrained models...")
    
    from models.download_pretrained import download_pretrained_models
    download_pretrained_models()
    
    click.echo("✅ Done!")


if __name__ == '__main__':
    cli()