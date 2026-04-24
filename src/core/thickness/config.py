"""
TEM 相关算法配置（纯配置常量）。

内容来源于旧版 `tem_calibration.config`，在此作为 core 层统一配置，
供预处理、FFT 分析等算法模块使用。
"""

# 放大倍率映射表（根据文件名前缀）
MAGNIFICATION_MAP = {
    "1.0": 5,
    "1.4": 15,
    "1.8": 25,
    "1.12": 40,
    "1.16": 100,
    "1.20": 100,
    "1.24": 150,
    "1.28": 80,
    "1.32": 40,
    "1.36": 30,
    "1.40": 60,
}

# 放大倍率选项（用于下拉菜单）
MAGNIFICATION_OPTIONS = [5, 15, 25, 30, 40, 60, 80, 100, 150]

# 默认参数
DEFAULT_SCALE_LENGTH: float = 50.0  # 默认比例尺长度（nm）
DEFAULT_MEASUREMENT_DIRECTION: str = "vertical"  # 默认测量方向

# 图像处理参数
NOISE_REDUCTION_KERNEL: int = 3  # 降噪核大小
CONTRAST_ENHANCEMENT_ALPHA: float = 1.5  # 对比度增强系数
CONTRAST_ENHANCEMENT_BETA: int = 0  # 亮度调整

# 层识别参数
PEAK_PROMINENCE: float = 0.1  # 峰检测的最小突出度（相对于最大灰度值）
PEAK_DISTANCE: int = 5  # 峰之间的最小距离（像素）


