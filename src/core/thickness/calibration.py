"""
比例尺与几何计算相关封装（纯算法实现）。

本模块不依赖 NiceGUI，仅处理数值及几何计算，
实现来源于原旧版包 `tem_calibration.calibration`。
"""

from typing import Tuple

import numpy as np


def calculate_scale_factor(scale_length_nm: float, pixel_distance: float) -> float:
    """计算比例因子（nm/像素）。"""
    if pixel_distance <= 0:
        return 0.0
    return scale_length_nm / pixel_distance


def calculate_pixel_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """计算两点之间的像素欧氏距离。"""
    return float(np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))


def pixels_to_nanometers(pixel_distance: float, scale_factor: float) -> float:
    """将像素距离转换为实际尺寸（纳米）。"""
    return pixel_distance * scale_factor


def nanometers_to_pixels(nanometer_distance: float, scale_factor: float) -> float:
    """将实际尺寸（纳米）转换为像素距离。"""
    if scale_factor <= 0:
        return 0.0
    return nanometer_distance / scale_factor


