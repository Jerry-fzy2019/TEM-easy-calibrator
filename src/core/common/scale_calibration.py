"""
通用像素比例换算模块。

提供统一的像素比例换算功能，供各个分析模块使用。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from src.core.thickness.calibration import calculate_scale_factor


def calibrate_scale_from_points(
    point1: Tuple[int, int],
    point2: Tuple[int, int],
    physical_length_nm: float,
    force_horizontal: bool = True,
    pixel_distance_override: float | None = None,
) -> Dict[str, Any]:
    """通过两点画线进行像素比例换算。
    
    Args:
        point1: 第一个点的坐标 (x, y)
        point2: 第二个点的坐标 (x, y)
        physical_length_nm: 两点间的物理长度（纳米）
        force_horizontal: 是否强制水平（只使用 x 方向距离），默认 True
        pixel_distance_override: 若提供，则用此值作为原图像素距离（用于显示缩放后的精确换算）
        
    Returns:
        包含校准结果的字典：
        - 'status': 'success' 或 'error'
        - 'nm_per_pixel': 计算出的分辨率（nm/px）
        - 'pixel_distance': 像素距离
        - 'physical_length_nm': 物理长度（nm）
        - 'point1': 第一个点坐标
        - 'point2': 第二个点坐标（如果强制水平，y 坐标会被调整）
        - 'error_message': 错误信息（如果失败）
    """
    try:
        x1, y1 = point1
        x2, y2 = point2
        
        if force_horizontal:
            y2 = y1
        
        if pixel_distance_override is not None and pixel_distance_override > 0:
            pixel_distance = float(pixel_distance_override)
        elif force_horizontal:
            pixel_distance = float(abs(x2 - x1))
        else:
            pixel_distance = float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        
        if pixel_distance <= 0:
            return {
                'status': 'error',
                'error_message': '两点距离为0，请重新画线'
            }
        
        # 计算 nm_per_pixel
        nm_per_pixel = calculate_scale_factor(physical_length_nm, pixel_distance)
        
        return {
            'status': 'success',
            'nm_per_pixel': round(nm_per_pixel, 6),
            'pixel_distance': round(pixel_distance, 2),
            'physical_length_nm': round(physical_length_nm, 4),
            'point1': (x1, y1),
            'point2': (x2, y2),
        }
        
    except Exception as e:  # noqa: BLE001
        return {
            'status': 'error',
            'error_message': f"像素比例换算失败: {str(e)}",
        }


def convert_nm_per_pixel_to_saed_factor(nm_per_pixel: float) -> float:
    """将 nm/px 转换为 SAED 标定系数 (nm⁻¹/pixel)。
    
    SAED 标定系数 k = 1 / (nm_per_pixel * d_reference)
    其中 d_reference 是参考晶面间距（nm）
    
    但更常用的方式是直接使用倒易空间关系：
    k = 1 / (nm_per_pixel * image_size)
    
    这里提供一个简化的转换，假设图像尺寸为 1 nm 的参考。
    实际使用时，应该根据具体的标定方法计算。
    
    Args:
        nm_per_pixel: 像素分辨率（nm/px）
        
    Returns:
        SAED 标定系数 k (nm⁻¹/pixel)
    """
    if nm_per_pixel <= 0:
        return 0.0
    
    # 注意：这个转换需要根据实际的 SAED 标定方法调整
    # 这里提供一个占位实现
    # 实际使用时，应该使用已知的标定样品（如 Si）进行标定
    return 1.0 / nm_per_pixel  # 简化转换，实际应该考虑图像尺寸等因素

