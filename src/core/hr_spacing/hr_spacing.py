"""
高分辨晶面间距校准（HRTEM lattice spacing calibration）核心模块。

实现基于 FFT 多峰检测的晶面间距分析。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import sys

import numpy as np

# 添加项目根目录到路径（用于导入其他模块）
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.thickness.fft_core import FFTPeriodicityAnalyzer
from src.core.thickness.image_io import load_image
from src.core.thickness.calibration import calculate_scale_factor
from src.core.common.scale_calibration import calibrate_scale_from_points
from typing import Tuple


class HRSpacingAnalyzer:
    """高分辨晶面间距分析器。"""
    
    def __init__(self):
        """初始化分析器。"""
        self.fft_analyzer: Optional[FFTPeriodicityAnalyzer] = None
    
    def _reconstruct_from_single_peak(
        self,
        fft_shifted: np.ndarray,
        center: Tuple[float, float],
        peak: Dict[str, Any],
        fft_shape: tuple,
    ) -> np.ndarray:
        """根据一个峰及其对称点进行IFFT重构图像。
        
        使用中心点、选择的峰点和其关于中心的对称点（共三个点）进行IFFT重构。
        
        Args:
            fft_shifted: FFT频谱（复数形式，已shift）
            center: 中心位置 (center_x, center_y)
            peak: 峰的位置，包含 'x', 'y'
            fft_shape: FFT频谱的形状 (h, w)
            
        Returns:
            重构的图像（灰度图，uint8）
        """
        if fft_shifted is None or peak is None:
            return None
        
        import cv2
        
        h, w = fft_shape
        center_x, center_y = center
        
        # 创建掩膜：只保留 DC、选择的峰及其对称峰（共三个点）
        mask = np.zeros((h, w), dtype=np.float64)
        
        # 保留中心直流分量
        dc_radius = max(3, int(min(h, w) * 0.03))
        cv2.circle(mask, (int(round(center_x)), int(round(center_y))), dc_radius, 1.0, -1)
        
        # 保留选择的峰及其对称峰
        peak_radius = 3
        x = peak.get('x', 0)
        y = peak.get('y', 0)
        x_int = int(round(x))
        y_int = int(round(y))
        
        # 确保坐标在范围内
        if 0 <= x_int < w and 0 <= y_int < h:
            # 在峰值位置创建圆形掩膜
            cv2.circle(mask, (x_int, y_int), peak_radius, 1.0, -1)
            
            # 计算对称位置（Friedel对称）
            sym_x = int(round(2 * center_x - x_int))
            sym_y = int(round(2 * center_y - y_int))
            if 0 <= sym_x < w and 0 <= sym_y < h:
                cv2.circle(mask, (sym_x, sym_y), peak_radius, 1.0, -1)
        
        # 应用掩膜到 FFT 频谱
        fft_filtered = fft_shifted * mask
        
        # 执行逆 FFT
        fft_ishifted = np.fft.ifftshift(fft_filtered)
        reconstructed = np.fft.ifft2(fft_ishifted)
        
        # 取模值并归一化到 0-255
        reconstructed_magnitude = np.abs(reconstructed)
        if reconstructed_magnitude.max() > 0:
            reconstructed_normalized = np.clip(
                (reconstructed_magnitude / reconstructed_magnitude.max() * 255),
                0, 255
            ).astype(np.uint8)
        else:
            reconstructed_normalized = np.zeros_like(reconstructed_magnitude, dtype=np.uint8)
        
        return reconstructed_normalized
    
    def _apply_friedel_filter(
        self,
        peaks_info: List[Dict[str, Any]],
        d_tolerance: float = 0.01,
        angle_tolerance_deg: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """应用 Friedel 定律过滤，去除对称的峰对。
        
        Args:
            peaks_info: 峰信息列表，每个元素包含 'position', 'distance_px', 'intensity' 等
            d_tolerance: d-spacing 的容差（nm），默认 0.01 nm
            angle_tolerance_deg: 角度的容差（度），默认 5.0 度
            
        Returns:
            过滤后的峰信息列表，只保留独立的晶面族
        """
        if len(peaks_info) <= 1:
            return peaks_info
        
        # 提取 d-spacing 和角度信息
        filtered_peaks = []
        used_indices = set()
        
        for i, peak in enumerate(peaks_info):
            if i in used_indices:
                continue
            
            d_i = peak.get('d_spacing_nm')
            angle_i = peak.get('angle_deg')
            pos_i = peak.get('position', (0, 0))
            x_i, y_i = pos_i
            
            # 检查是否有对称峰（角度相差约 180 度）
            is_symmetric = False
            for j, other_peak in enumerate(peaks_info[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                d_j = other_peak.get('d_spacing_nm')
                angle_j = other_peak.get('angle_deg')
                pos_j = other_peak.get('position', (0, 0))
                x_j, y_j = pos_j
                
                # 检查 d-spacing 是否接近
                if abs(d_i - d_j) > d_tolerance:
                    continue
                
                # 检查角度是否相差约 180 度
                angle_diff = abs(angle_i - angle_j)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if abs(angle_diff - 180.0) <= angle_tolerance_deg:
                    # 找到对称峰，只保留一个（优先保留 y>0 或 x>0 的）
                    is_symmetric = True
                    used_indices.add(j)
                    break
            
            # 如果没有对称峰，或者当前峰是保留的那个，添加到结果中
            if not is_symmetric:
                filtered_peaks.append(peak)
            else:
                # 选择保留哪个峰：优先保留 y>0 或 (y==0 且 x>0) 的
                if y_i > 0 or (y_i == 0 and x_i > 0):
                    filtered_peaks.append(peak)
                else:
                    # 保留对称峰
                    for j, other_peak in enumerate(peaks_info[i+1:], start=i+1):
                        if j in used_indices:
                            continue
                        d_j = other_peak.get('d_spacing_nm')
                        angle_j = other_peak.get('angle_deg')
                        if abs(d_i - d_j) <= d_tolerance:
                            angle_diff = abs(angle_i - angle_j)
                            if angle_diff > 180:
                                angle_diff = 360 - angle_diff
                            if abs(angle_diff - 180.0) <= angle_tolerance_deg:
                                filtered_peaks.append(other_peak)
                                break
        
        return filtered_peaks
    
    def analyze_single_image(
        self,
        image_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """分析单张 HRTEM 图像的晶面间距。
        
        Args:
            image_path: 图像文件路径
            config: 配置字典，可包含：
                - 'nm_per_pixel': 像素分辨率（nm/px），如果提供则直接使用
                - 'scale_length_nm': 标尺物理长度（nm），如果提供则尝试自动识别标尺
                - 'scale_point1', 'scale_point2': 手动标尺两点坐标（可选）
                - 'roi_coords': ROI 坐标 (x1, y1, x2, y2)（可选，如果不提供则使用整幅图像）
                - 'min_distance': 峰之间的最小像素距离（默认 10）
                - 'threshold_rel': 相对阈值（默认 0.1）
                - 'exclude_center_radius': 中心屏蔽半径（像素，默认自动计算）
                - 'max_peaks': 最大峰值数量（默认 20）
                - 'apply_friedel_filter': 是否应用 Friedel 过滤（默认 True）
                - 'd_tolerance': Friedel 过滤的 d-spacing 容差（nm，默认 0.01）
                - 'angle_tolerance_deg': Friedel 过滤的角度容差（度，默认 5.0）
            
        Returns:
            结果字典，包含：
            - 'status': 'success' 或 'error'
            - 'results': 晶面信息列表，每个元素包含：
                - 'd_spacing_nm': 晶面间距（nm）
                - 'angle_deg': 相对于 X 轴的角度（度，0-180）
                - 'intensity': 峰值强度
                - 'reciprocal_nm': 倒易矢量长度（1/nm）
                - 'position': 峰值位置 (x, y)
                - 'distance_px': 距离中心的像素距离
            - 'error_message': 错误信息（如果失败）
            - 'processing_time': 处理时间（秒）
        """
        start_time = time.time()
        
        if config is None:
            config = {}
        
        try:
            # 1. 初始化 FFT 分析器
            self.fft_analyzer = FFTPeriodicityAnalyzer()
            
            # 2. 加载图像
            image = load_image(image_path)
            if image is None:
                return {
                    'status': 'error',
                    'error_message': '图像加载失败，请检查文件格式',
                    'processing_time': time.time() - start_time,
                }
            
            # 为避免将图像下方的比例尺条纹/标注区域误算入晶面分析，
            # 默认裁掉图像底部 1/10 的高度，仅使用上方 90% 进行 FFT 与峰检测。
            try:
                h, w = image.shape[:2]
                if h > 10:
                    crop_h = int(h * 0.9)
                    if crop_h < h:
                        image = image[:crop_h, :]
            except Exception:
                # 出现任何异常时，回退为使用完整图像，避免崩溃
                pass
            
            self.fft_analyzer.original_image = image.copy()
            self.fft_analyzer.gray_image = image.copy()
            
            h, w = image.shape[:2]
            
            # 3. 设置 ROI（如果配置中提供了）
            roi_coords = config.get('roi_coords')
            if roi_coords is not None:
                x1, y1, x2, y2 = roi_coords
                self.fft_analyzer.set_roi(x1, y1, x2, y2)
            else:
                # 使用整幅图像
                self.fft_analyzer.set_roi(0, 0, w, h)
            
            # 4. 设置比例尺（统一使用通用像素比例换算逻辑）
            nm_per_pixel = config.get("nm_per_pixel")
            if nm_per_pixel is not None and nm_per_pixel > 0:
                # 直接使用提供的 nm_per_pixel
                self.fft_analyzer.nm_per_pixel = float(nm_per_pixel)
                self.fft_analyzer.scale_length_nm = config.get(
                    "scale_length_nm", self.fft_analyzer.scale_length_nm
                )
            else:
                scale_length_nm = config.get("scale_length_nm")
                if scale_length_nm is not None and scale_length_nm > 0:
                    scale_point1 = config.get("scale_point1")
                    scale_point2 = config.get("scale_point2")
                    if scale_point1 is not None and scale_point2 is not None:
                        calib_res = calibrate_scale_from_points(
                            scale_point1,
                            scale_point2,
                            scale_length_nm,
                            force_horizontal=False,
                        )
                        if calib_res.get("status") != "success":
                            return {
                                "status": "error",
                                "error_message": calib_res.get(
                                    "error_message", "像素比例换算失败"
                                ),
                                "processing_time": time.time() - start_time,
                            }
                        nm_per_pixel = calib_res.get("nm_per_pixel", 0.0)
                        self.fft_analyzer.nm_per_pixel = nm_per_pixel
                        self.fft_analyzer.scale_length_nm = scale_length_nm
                    else:
                        return {
                            "status": "error",
                            "error_message": "未提供比例尺信息（nm_per_pixel 或 scale_length_nm + 标尺点）",
                            "processing_time": time.time() - start_time,
                        }
                else:
                    return {
                        "status": "error",
                        "error_message": "未提供比例尺信息（nm_per_pixel 或 scale_length_nm）",
                        "processing_time": time.time() - start_time,
                    }
            
            if self.fft_analyzer.nm_per_pixel <= 0:
                return {
                    'status': 'error',
                    'error_message': '比例尺未校准（nm_per_pixel <= 0）',
                    'processing_time': time.time() - start_time,
                }
            
            # 5. 执行 FFT
            fft_shifted, magnitude_log = self.fft_analyzer.compute_fft()
            
            # 6. 峰检测（手动或自动）
            manual_peaks = config.get('manual_peaks')
            if manual_peaks and len(manual_peaks) > 0:
                # 手动选择的峰
                peaks_info = []
                h, w = magnitude_log.shape
                center_x, center_y = w // 2, h // 2
                
                for peak in manual_peaks:
                    x = peak.get('x', 0)
                    y = peak.get('y', 0)
                    
                    # 计算距离中心的距离
                    dx = x - center_x
                    dy = y - center_y
                    distance_px = np.sqrt(dx**2 + dy**2)
                    
                    # 获取强度值
                    x_int = int(round(x))
                    y_int = int(round(y))
                    if 0 <= x_int < w and 0 <= y_int < h:
                        intensity = float(magnitude_log[y_int, x_int])
                    else:
                        intensity = 0.0
                    
                    peaks_info.append({
                        'position': (float(x), float(y)),
                        'distance_px': float(distance_px),
                        'intensity': intensity,
                    })
            else:
                # 自动检测
                min_distance = config.get('min_distance', 10)
                threshold_rel = config.get('threshold_rel', 0.1)
                exclude_center_radius = config.get('exclude_center_radius')
                max_peaks = config.get('max_peaks', 20)
                
                peaks_info = self.fft_analyzer.detect_multiple_peaks(
                    magnitude_log,
                    min_distance=min_distance,
                    threshold_rel=threshold_rel,
                    exclude_center_radius=exclude_center_radius,
                    max_peaks=max_peaks,
                )
            
            if not peaks_info:
                return {
                    'status': 'error',
                        'error_message': '未检测到任何衍射峰',
                    'processing_time': time.time() - start_time,
                }
            
            # 7. 计算物理量（d-spacing 和角度）
            # 获取 FFT 计算时使用的 ROI 尺寸
            roi_size = self.fft_analyzer._fft_calc_size
            if roi_size is None:
                if self.fft_analyzer.roi is not None:
                    roi_size = min(self.fft_analyzer.roi.shape[:2])
                else:
                    roi_size = min(h, w)
            
            nm_per_pixel = self.fft_analyzer.nm_per_pixel
            h_fft, w_fft = magnitude_log.shape
            center_x, center_y = w_fft // 2, h_fft // 2
            
            results = []
            for peak in peaks_info:
                position = peak.get('position', (0, 0))
                distance_px = peak.get('distance_px', 0)
                intensity = peak.get('intensity', 0)
                
                x, y = position
                
                # 计算 d-spacing: d = (roi_size / distance_px) * nm_per_pixel
                if distance_px > 1e-5:
                    d_spacing_nm = (roi_size / distance_px) * nm_per_pixel
                else:
                    d_spacing_nm = 0.0
                
                # 计算倒易矢量长度（1/nm）
                if d_spacing_nm > 1e-5:
                    reciprocal_nm = 1.0 / d_spacing_nm
                else:
                    reciprocal_nm = 0.0
                
                # 计算角度（相对于 X 轴，0-180 度）
                # 注意：FFT 坐标系中，y 轴向下，x 轴向右
                # 使用 atan2 计算角度，然后转换为 0-180 度范围
                dx = x - center_x
                dy = y - center_y
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)
                
                # 标准化到 0-180 度范围（因为 Friedel 对称性）
                if angle_deg < 0:
                    angle_deg += 180.0
                if angle_deg >= 180:
                    angle_deg -= 180.0
                
                results.append({
                    'd_spacing_nm': round(d_spacing_nm, 4),
                    'angle_deg': round(angle_deg, 2),
                    'intensity': round(float(intensity), 2),
                    'reciprocal_nm': round(reciprocal_nm, 4),
                    'position': position,
                    'distance_px': round(distance_px, 2),
                })
            
            # 8. 如果手动选择了2个峰，进行IFFT重构
            reconstructed_image = None
            if manual_peaks and len(manual_peaks) == 2:
                # 使用两个峰进行IFFT重构
                reconstructed_image = self._reconstruct_from_two_peaks(
                    self.fft_analyzer.fft_shifted,
                    manual_peaks,
                    magnitude_log.shape
                )
            
            # 9. 应用 Friedel 过滤（如果启用）
            apply_friedel_filter = config.get('apply_friedel_filter', True)
            if apply_friedel_filter and len(results) > 1:
                d_tolerance = config.get('d_tolerance', 0.01)
                angle_tolerance_deg = config.get('angle_tolerance_deg', 5.0)
                results = self._apply_friedel_filter(
                    results,
                    d_tolerance=d_tolerance,
                    angle_tolerance_deg=angle_tolerance_deg,
                )
            
            # 10. 按 d-spacing 从大到小排序
            results.sort(key=lambda r: r['d_spacing_nm'], reverse=True)
            
            result_dict = {
                'status': 'success',
                'results': results,
                'processing_time': round(time.time() - start_time, 2),
                'is_calibrated': self.fft_analyzer.nm_per_pixel > 0,
            }
            
            # 如果进行了IFFT重构，添加重构图像
            if reconstructed_image is not None:
                result_dict['reconstructed_image'] = reconstructed_image
            
            return result_dict
        
        except Exception as e:
            import traceback
            return {
                'status': 'error',
                'error_message': f'分析过程出错: {str(e)}\n{traceback.format_exc()}',
                'processing_time': time.time() - start_time,
            }
