"""
图像预处理相关封装。

实现来源于旧版 `tem_calibration.image_processor.preprocess_image`，
仅做降噪与对比度增强等数值处理，不涉及任何 UI。
"""

from typing import Optional

import cv2
import numpy as np

from .config import (
    CONTRAST_ENHANCEMENT_ALPHA,
    CONTRAST_ENHANCEMENT_BETA,
    NOISE_REDUCTION_KERNEL,
)


def preprocess_image(
    image: np.ndarray,
    noise_kernel: Optional[int] = None,
    contrast_alpha: Optional[float] = None,
    brightness_beta: Optional[int] = None,
    gaussian_blur: Optional[float] = None,
) -> np.ndarray:
    """图像预处理：高斯模糊 + 中值滤波降噪 + 线性对比度/亮度调整。

    Args:
        image: 输入灰度图像
        noise_kernel: 降噪核大小（必须是奇数，如 3,5,7...）
        contrast_alpha: 对比度增强系数（>1 增强，<1 减弱）
        brightness_beta: 亮度调整值（-100 到 100）
        gaussian_blur: 高斯模糊的标准差（0表示不模糊）
    """
    if noise_kernel is None:
        noise_kernel = NOISE_REDUCTION_KERNEL
    if contrast_alpha is None:
        contrast_alpha = CONTRAST_ENHANCEMENT_ALPHA
    if brightness_beta is None:
        brightness_beta = CONTRAST_ENHANCEMENT_BETA
    if gaussian_blur is None:
        gaussian_blur = 0.0

    # 确保是灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    result = image.copy()
    
    # 高斯模糊（如果启用）
    if gaussian_blur > 0:
        # 计算核大小（必须是奇数）
        ksize = int(6 * gaussian_blur + 1)
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.GaussianBlur(result, (ksize, ksize), gaussian_blur)
    
    # 对比度与亮度调整（已取消中值滤波）
    enhanced = cv2.convertScaleAbs(
        result, alpha=float(contrast_alpha), beta=int(brightness_beta)
    )

    return enhanced


