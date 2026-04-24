"""
膜厚校准（TEM thickness calibration）核心模块集合。

本子包下包含：
- `calibration`：比例尺与几何距离换算
- `config`：与膜厚测量相关的默认配置
- `image_io`：膜厚测量常用的图像加载
- `image_preprocess`：预处理（降噪、对比度）
- `fft_core`：基于 FFT 的周期性分析
- `thickness`：面向 UI 的应用状态封装
"""

from .calibration import (
    calculate_pixel_distance,
    calculate_scale_factor,
    nanometers_to_pixels,
    pixels_to_nanometers,
)
from .config import (
    CONTRAST_ENHANCEMENT_ALPHA,
    CONTRAST_ENHANCEMENT_BETA,
    DEFAULT_MEASUREMENT_DIRECTION,
    DEFAULT_SCALE_LENGTH,
    MAGNIFICATION_MAP,
    MAGNIFICATION_OPTIONS,
    NOISE_REDUCTION_KERNEL,
    PEAK_DISTANCE,
    PEAK_PROMINENCE,
)
from .fft_core import FFTPeriodicityAnalyzer
from .image_io import load_image
from .image_preprocess import preprocess_image
from .thickness import TEMCalibrationApp

__all__ = [
    # calibration
    "calculate_scale_factor",
    "calculate_pixel_distance",
    "pixels_to_nanometers",
    "nanometers_to_pixels",
    # config
    "MAGNIFICATION_MAP",
    "MAGNIFICATION_OPTIONS",
    "DEFAULT_SCALE_LENGTH",
    "DEFAULT_MEASUREMENT_DIRECTION",
    "NOISE_REDUCTION_KERNEL",
    "CONTRAST_ENHANCEMENT_ALPHA",
    "CONTRAST_ENHANCEMENT_BETA",
    "PEAK_PROMINENCE",
    "PEAK_DISTANCE",
    # image I/O & preprocess
    "load_image",
    "preprocess_image",
    # FFT
    "FFTPeriodicityAnalyzer",
    # app
    "TEMCalibrationApp",
]


