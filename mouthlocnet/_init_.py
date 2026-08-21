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
__email__ = "najib@example.com"
__url__ = "https://github.com/slam-prog/MouthLocNet"
__description__ = "Mouth Sound Localization using Deep Learning"

from .src.audio_capture import AudioCapture, AudioConfig
from .src.tdoa_calculation import TDOACalculator, TDOAResult
from .src.srp_phat import SRPPHAT, SRPResult
from .src.relative_pattern_matching import RelativePatternMatcher, RelativePatternResult
from .src.deep_learning_model import MouthLocNet, ModelConfig
from .src.sensor_fusion import SensorFusion, SensorData, FusedPosition
from .src.phoneme_classifier import PhonemeClassifier, PhonemeResult
from .src.real_time_processor import RealTimeProcessor, RTConfig, RTResult
from .src.utils import (
    load_audio,
    save_audio,
    visualize_position,
    calculate_accuracy,
    load_phoneme_patterns,
    save_phoneme_patterns,
    create_mic_array,
    plot_error_distribution,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__contributors__",
    "__license__",
    
    # Classes
    "AudioCapture",
    "AudioConfig",
    "TDOACalculator",
    "TDOAResult",
    "SRPPHAT",
    "SRPResult",
    "RelativePatternMatcher",
    "RelativePatternResult",
    "MouthLocNet",
    "ModelConfig",
    "SensorFusion",
    "SensorData",
    "FusedPosition",
    "PhonemeClassifier",
    "PhonemeResult",
    "RealTimeProcessor",
    "RTConfig",
    "RTResult",
    
    # Utilities
    "load_audio",
    "save_audio",
    "visualize_position",
    "calculate_accuracy",
    "load_phoneme_patterns",
    "save_phoneme_patterns",
    "create_mic_array",
    "plot_error_distribution",
]