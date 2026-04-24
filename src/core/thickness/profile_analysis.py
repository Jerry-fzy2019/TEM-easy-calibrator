"""
灰度剖面分析模块 (增强版)：包含角度精修与抗噪投影算法。

FFT 只能测量周期性间距（黑+白），而本模块通过灰度剖面分析
可以单独测量黑色条纹的厚度。

核心改进：
1. 自动角度精修：在 FFT 粗算角度基础上微调，寻找投影对比度最大的角度
2. 鲁棒统计：使用 percentile 代替 min/max 进行归一化，抗噪能力提升
3. 智能过滤：自动剔除不合理的厚度值（如厚度 > 周期）
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import rotate
from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy.optimize import minimize_scalar


class ProfileThicknessAnalyzer:
    """基于灰度剖面分析的单层厚度测量器（带角度优化）。"""

    def __init__(self, nm_per_pixel: float, peak_height_min: float = 0.2):
        """初始化分析器。

        Args:
            nm_per_pixel: 每像素对应的纳米数（比例尺因子）
            peak_height_min: 峰检测最小高度（归一化 0~1），用于判断膜/黑条纹的波谷深度，越大越严格
        """
        self.nm_per_pixel = nm_per_pixel
        self.peak_height_min = peak_height_min

    def _calculate_profile_contrast(self, angle: float, image: np.ndarray) -> float:
        """计算指定旋转角度下的投影对比度（用于优化目标函数）。
        
        对比度越高（方差越大），说明条纹越垂直、越清晰。
        
        Args:
            angle: 旋转角度（度）
            image: 输入图像
            
        Returns:
            投影方差的负值（因为我们要最小化目标函数）
        """
        rotated = rotate(image, angle, reshape=False, order=1, mode='reflect')
        # 裁剪中心区域避免边缘黑边干扰
        h, w = rotated.shape
        crop = rotated[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        if crop.size == 0:
            return 0.0
        
        # 垂直投影
        profile = np.mean(crop, axis=0)
        # 返回方差的负值（因为我们要最小化目标函数）
        return -np.var(profile)

    def optimize_orientation(self, image: np.ndarray, coarse_angle: float) -> float:
        """在粗略角度基础上进行精修，寻找最佳投影角度。
        
        Args:
            image: 输入图像
            coarse_angle: 粗略角度（度）
            
        Returns:
            优化后的角度（度）
        """
        # 搜索范围：粗角度 ±5 度
        bounds = (coarse_angle - 5.0, coarse_angle + 5.0)
        
        # 使用有界优化寻找让 profile 方差最大的角度
        result = minimize_scalar(
            self._calculate_profile_contrast,
            args=(image,),
            bounds=bounds,
            method='bounded'
        )
        return result.x

    def get_coarse_angle(self, image: np.ndarray) -> float:
        """基于 FFT 计算粗略角度。
        
        Args:
            image: 输入灰度图像
            
        Returns:
            旋转角度（度），使得旋转后条纹垂直
        """
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # 屏蔽中心
        cv2.circle(magnitude, (cx, cy), 5, 0, -1)
        
        _, _, _, max_loc = cv2.minMaxLoc(magnitude)
        dx = max_loc[0] - cx
        dy = max_loc[1] - cy
        
        if dx == 0 and dy == 0:
            return 0.0
        
        angle_rad = np.arctan2(dy, dx)
        return np.degrees(angle_rad) + 90.0

    def analyze_profile_1d(
        self,
        profile: np.ndarray,
        estimated_period_nm: Optional[float] = None,
        invert_signal: bool = True,
    ) -> Tuple[Optional[float], Optional[float], Dict]:
        """分析一维灰度剖面信号，提取黑色条纹厚度。
        
        这个方法从图像处理中独立出来，可以处理任何来源的 1D profile 数据
        （包括自动投影、手动画线提取等）。
        
        Args:
            profile: 一维灰度剖面数组
            estimated_period_nm: 估计的周期（黑+白总宽度，nm），用于过滤异常值
            invert_signal: 如果 True，寻找波谷（黑色条纹）；如果 False，寻找波峰（白色条纹）
            
        Returns:
            (平均厚度_nm, 标准差_nm, 调试数据字典)
            如果检测失败则返回 (None, None, debug_data)
        """
        if profile is None or len(profile) == 0:
            return None, None, {}

        # 1. 鲁棒归一化 (抗异常值)
        # 使用 1% 和 99% 分位数作为 min/max，避免噪点拉伸曲线
        p_min = np.percentile(profile, 1)
        p_max = np.percentile(profile, 99)
        
        if p_max - p_min < 1e-5:
            return None, None, {"profile": profile.tolist()}
            
        profile_norm = (profile - p_min) / (p_max - p_min)
        profile_norm = np.clip(profile_norm, 0, 1)  # 截断

        # 2. 平滑处理
        window_len = max(5, int(len(profile) * 0.02) // 2 * 2 + 1)
        if window_len >= 3 and len(profile_norm) > window_len:
            try:
                profile_smooth = savgol_filter(profile_norm, window_len, 3)
            except Exception:
                profile_smooth = profile_norm
        else:
            profile_smooth = profile_norm

        # 3. 寻找黑色条纹（波谷 -> 反转找波峰）或白色条纹（直接找波峰）
        peak_height = getattr(self, 'peak_height_min', 0.2)
        peak_height = max(0.05, min(0.95, float(peak_height)))
        if invert_signal:
            signal_for_peaks = 1.0 - profile_smooth
            peaks, properties = find_peaks(signal_for_peaks, height=peak_height, distance=5)
            
            # 额外验证：确保这些峰值在原始信号中确实是波谷（低值）
            # 检查原始信号中对应位置的灰度值是否低于平均值
            if len(peaks) > 0:
                mean_val = np.mean(profile_smooth)
                valid_peaks = []
                for peak_idx in peaks:
                    # 在原始信号中，这个位置应该是低值（波谷）
                    if profile_smooth[peak_idx] < mean_val:
                        valid_peaks.append(peak_idx)
                peaks = np.array(valid_peaks)
        else:
            signal_for_peaks = profile_smooth
            peaks, properties = find_peaks(signal_for_peaks, height=peak_height, distance=5)

        if len(peaks) == 0:
            return None, None, {
                "profile": profile_norm.tolist(),
                "profile_smooth": profile_smooth.tolist(),
                "peaks": [],
                "widths_nm": [],
            }

        # 4. 测量半高宽 (FWHM, rel_height=0.5)
        widths_px, width_heights, left_ips, right_ips = peak_widths(
            signal_for_peaks, peaks, rel_height=0.5
        )
        
        widths_nm = widths_px * self.nm_per_pixel
        
        # 5. 智能筛选：剔除不合理的厚度
        valid_widths = []
        valid_indices = []
        
        for i, w in enumerate(widths_nm):
            # 规则1: 单层厚度必须 > 0
            # 规则2: 如果已知周期，单层厚度不应超过周期的 80% (预留给白层)
            if w <= 0:
                continue
            if estimated_period_nm and w > estimated_period_nm * 0.8:
                continue
            valid_widths.append(w)
            valid_indices.append(i)
            
        if not valid_widths:
            return None, None, {
                "profile": profile_norm.tolist(),
                "profile_smooth": profile_smooth.tolist(),
                "peaks": peaks.tolist(),
                "widths_nm": [],
            }

        mean_thick = np.mean(valid_widths)
        std_thick = np.std(valid_widths)

        # 6. 准备调试数据 (用于画图)
        # 还原高度用于可视化
        if invert_signal:
            # 反转回原始坐标系(0为黑, 1为白)
            visual_heights = 1.0 - width_heights[valid_indices]
        else:
            visual_heights = width_heights[valid_indices]
        
        debug_data = {
            "profile": profile_norm.tolist(),  # 归一化后的曲线
            "profile_smooth": profile_smooth.tolist(),  # 平滑后的曲线
            "peaks": peaks[valid_indices].tolist(),
            "widths_pixels": widths_px[valid_indices].tolist(),
            "widths_nm": valid_widths,
            "mean_width_nm": mean_thick,
            "std_width_nm": std_thick,
            "invert_signal": invert_signal,
            # 以下用于画红线
            "left_ips": left_ips[valid_indices].tolist(),
            "right_ips": right_ips[valid_indices].tolist(),
            "width_heights": visual_heights.tolist(),
        }

        return mean_thick, std_thick, debug_data

    def analyze_thickness(
        self,
        roi_image: np.ndarray,
        orientation_angle: Optional[float] = None,
        estimated_period_nm: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float], Dict]:
        """执行全自动厚度分析。
        
        Args:
            roi_image: ROI 区域图像（灰度）
            orientation_angle: 预设的旋转角度，如果为 None 则自动计算
            estimated_period_nm: 估计的周期（黑+白总宽度，nm），用于过滤异常值
            
        Returns:
            (平均厚度_nm, 标准差_nm, 调试数据字典)
            如果检测失败则返回 (None, None, debug_data)
        """
        if roi_image is None or roi_image.size == 0:
            return None, None, {}

        # 1. 角度计算：粗算 -> 精修
        if orientation_angle is None:
            coarse_angle = self.get_coarse_angle(roi_image)
            fine_angle = self.optimize_orientation(roi_image, coarse_angle)
        else:
            fine_angle = orientation_angle
        
        # 2. 旋转并生成投影
        rotated = rotate(roi_image, fine_angle, reshape=False, mode='reflect')
        h, w = rotated.shape
        # 取中间 60% 区域，更加稳健
        center_strip = rotated[int(h * 0.2):int(h * 0.8), :]
        profile = np.mean(center_strip, axis=0)

        # 3. 调用独立的 1D 分析函数
        mean_thick, std_thick, debug_data = self.analyze_profile_1d(
            profile, estimated_period_nm=estimated_period_nm, invert_signal=True
        )

        # 4. 在调试数据中添加角度信息
        if debug_data:
            debug_data["angle"] = fine_angle

        return mean_thick, std_thick, debug_data
