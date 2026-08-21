from setuptools import setup, find_packages
from pathlib import Path

# قراءة README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# قراءة المتطلبات
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="mouthlocnet",
    version="2.0.0",
    author="NAJIB MOHAMMED AL-AMIR",
    author_email="najib@example.com",
    description="Mouth Sound Localization using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slam-prog/MouthLocNet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest>=7.3.0",
            "sphinx>=6.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu117",
            "torchaudio>=2.0.0+cu117",
        ],
    },
    entry_points={
        "console_scripts": [
            "mouthlocnet=mouthlocnet.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mouthlocnet": [
            "data/*.json",
            "models/*.pt",
        ],
    },
    license="HEUL-2.0",
    keywords=[
        "mouth localization",
        "sound source localization",
        "deep learning",
        "speech processing",
        "audio analysis",
        "AI-Human collaboration",
    ],
)