"""
TEM 相关的核心算法与数据处理模块（UI 无关）。

按功能划分为五个子板块：
- thickness: 膜厚校准
- hr_spacing: 高分辨晶面间距校准
- saed: 选区电子衍射 (SAED) 校准
- drift: 图像漂移 / 畸变校准
- contamination: 污染率表征
"""

from . import contamination, drift, hr_spacing, saed, thickness

__all__ = [
    "thickness",
    "hr_spacing",
    "saed",
    "drift",
    "contamination",
]

