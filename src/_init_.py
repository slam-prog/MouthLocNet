"""
MouthLocNet - نظام تحديد موقع الصوت من الفم باستخدام التعلم العميق

الإصدار: 2.0.0
تم التطوير بمساعدة Perplexity AI كمساعد ذكي

المؤلف: NAJIB MOHAMMED AL-AMIR
المساهمون: DeepSeek AI, Perplexity AI
الترخيص: HEUL v2.0
"""

__version__ = "2.0.0"
__author__ = "NAJIB MOHAMMED AL-AMIR"
__contributors__ = ["DeepSeek AI", "Perplexity AI"]
__license__ = "HEUL-2.0"

from .audio_capture import AudioCapture
from .tdoa_calculation import TDOACalculator
from .srp_phat import SRPPHAT
from .relative_pattern_matching import RelativePatternMatcher
from .deep_learning_model import MouthLocNet
from .sensor_fusion import SensorFusion
from .phoneme_classifier import PhonemeClassifier
from .real_time_processor import RealTimeProcessor
from .utils import (
    load_audio,
    save_audio,
    visualize_position,
    calculate_accuracy,
)

__all__ = [
    "AudioCapture",
    "TDOACalculator",
    "SRPPHAT",
    "RelativePatternMatcher",
    "MouthLocNet",
    "SensorFusion",
    "PhonemeClassifier",
    "RealTimeProcessor",
    "load_audio",
    "save_audio",
    "visualize_position",
    "calculate_accuracy",
]