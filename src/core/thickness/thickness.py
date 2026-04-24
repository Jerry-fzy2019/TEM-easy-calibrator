"""
TEM 膜厚校准应用状态与业务逻辑。

本模块迁移自旧版 `tem_calibration.app.TEMCalibrationApp`，
只依赖 `numpy`/`opencv`/`matplotlib` 等科学计算库，不依赖 NiceGUI。
UI 层通过本模块提供的类和方法驱动业务流程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .calibration import calculate_scale_factor
from .config import (
    CONTRAST_ENHANCEMENT_ALPHA,
    CONTRAST_ENHANCEMENT_BETA,
    DEFAULT_SCALE_LENGTH,
    MAGNIFICATION_OPTIONS,
    NOISE_REDUCTION_KERNEL,
)
from .fft_core import FFTPeriodicityAnalyzer
from .image_io import load_image
from .image_preprocess import preprocess_image
from .profile_analysis import ProfileThicknessAnalyzer
from src.core.common.vision_utils import auto_find_roi, detect_scale_bar
from src.core.common.scale_calibration import calibrate_scale_from_points

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


@dataclass
class TEMCalibrationApp:
    """TEM 膜厚校准应用核心状态与算法逻辑（UI 无关）。"""

    current_image: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None
    scale_factor: float = 0.0
    scale_point1: Optional[Tuple[int, int]] = None
    scale_point2: Optional[Tuple[int, int]] = None
    statistics: Dict = field(default_factory=dict)
    mode: Optional[str] = None  # "scale" / "roi" / None
    current_scale_length: float = DEFAULT_SCALE_LENGTH
    current_magnification: float = MAGNIFICATION_OPTIONS[0]
    scale_status_msg: str = "等待校准比例尺..."
    process_log: List[str] = field(default_factory=list)
    scale_bbox: Optional[List[int]] = None
    preprocess_params: Dict[str, float] = field(
        default_factory=lambda: {
            "noise_kernel": NOISE_REDUCTION_KERNEL,
            "contrast_alpha": CONTRAST_ENHANCEMENT_ALPHA,
            "brightness_beta": CONTRAST_ENHANCEMENT_BETA,
        }
    )
    fft_analyzer: FFTPeriodicityAnalyzer = field(default_factory=FFTPeriodicityAnalyzer)
    profile_analyzer: Optional[ProfileThicknessAnalyzer] = None
    roi_point1: Optional[Tuple[int, int]] = None
    roi_point2: Optional[Tuple[int, int]] = None
    roi_point3: Optional[Tuple[int, int]] = None
    fft_result_image: Optional[np.ndarray] = None
    periodicity_nm: Optional[float] = None
    layer_thickness_nm: Optional[float] = None  # 单层黑色条纹厚度（通过剖面分析得到）
    profile_debug_data: Optional[Dict] = None  # 剖面分析调试数据
    mask_radius: int = 10
    auto_roi_box: Optional[Tuple[int, int, int, int]] = None

    # -------- 日志工具 --------

    def _add_log(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_symbols = {
            "INFO": "→",
            "SUCCESS": "✓",
            "WARNING": "⚠",
            "ERROR": "✗",
        }
        symbol = level_symbols.get(level, "•")
        log_entry = f"[{timestamp}] {symbol} {message}"
        self.process_log.append(log_entry)
        if len(self.process_log) > 100:
            self.process_log = self.process_log[-100:]
        print(log_entry)

    def get_process_log(self) -> str:
        if not self.process_log:
            return "等待开始计算...\n请上传图像文件开始处理。"
        return "\n".join(self.process_log[-50:])

    def clear_log(self) -> None:
        self.process_log = []
        self._add_log("日志已清空，开始新的计算流程", "INFO")

    # -------- 手动校准方法 --------
    
    def calibrate_manual(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        physical_length_nm: float,
    ) -> Dict[str, Any]:
        """手动校准比例尺（通过两点画线）。
        
        Args:
            point1: 第一个点的坐标 (x, y)
            point2: 第二个点的坐标 (x, y)
            physical_length_nm: 两点间的物理长度（纳米）
            
        Returns:
            包含校准结果的字典：
            - 'status': 'success' 或 'error'
            - 'nm_per_pixel': 计算出的分辨率
            - 'pixel_distance': 像素距离
            - 'error_message': 错误信息（如果失败）
        """
        try:
            # 统一使用通用像素比例换算函数
            result = calibrate_scale_from_points(
                point1,
                point2,
                physical_length_nm,
                force_horizontal=True,
            )
            
            if result.get("status") != "success":
                error_msg = result.get("error_message", "像素比例换算失败")
                self._add_log(error_msg, "ERROR")
                return {
                    "status": "error",
                    "error_message": error_msg,
                }
            
            nm_per_pixel = result.get("nm_per_pixel", 0.0)
            pixel_distance = result.get("pixel_distance", 0.0)
            x1, y1 = result.get("point1", point1)
            x2, y2 = result.get("point2", point2)
            
            # 更新内部状态
            self.fft_analyzer.nm_per_pixel = nm_per_pixel
            self.fft_analyzer.scale_length_nm = physical_length_nm
            self.scale_factor = nm_per_pixel
            self.current_scale_length = physical_length_nm
            self.scale_point1 = (x1, y1)
            self.scale_point2 = (x2, y2)
            
            # 保存标尺框坐标用于可视化 (x1, y1, x2, y2)
            box_height = 5
            self.scale_bbox = [
                min(x1, x2),
                min(y1, y2) - box_height // 2,
                max(x1, x2),
                max(y1, y2) + box_height // 2,
            ]
            
            self._add_log(
                f"手动校准成功: {nm_per_pixel:.4f} nm/px (距离={pixel_distance:.2f} px)",
                "SUCCESS",
            )
            
            return {
                "status": "success",
                "nm_per_pixel": nm_per_pixel,
                "pixel_distance": pixel_distance,
                "physical_length_nm": physical_length_nm,
            }
            
        except Exception as e:  # noqa: BLE001
            error_msg = f"手动校准失败: {str(e)}"
            self._add_log(error_msg, "ERROR")
            return {
                "status": "error",
                "error_message": error_msg,
            }

    # -------- 统一接口方法 --------

    def analyze_single_image(
        self,
        image_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """全自动分析单张图像，返回结果字典。
        
        执行流程：
        1. 加载图像
        2. 预处理（使用配置参数或默认值）
        3. 自动标尺识别（如果配置了标尺长度）
        4. 自动条纹检测（使用结构张量）
        5. FFT 计算
        6. 返回结果
        
        Args:
            image_path: 图像文件路径
            config: 配置字典，可包含：
                - 'noise_kernel': 降噪核大小（默认 3）
                - 'contrast_alpha': 对比度增强系数（默认 1.5）
                - 'brightness_beta': 亮度调整（默认 0）
                - 'scale_length_nm': 标尺物理长度（nm），如果提供则自动识别标尺
                - 'window_size': ROI 窗口大小（默认 512）
                - 'mask_radius': FFT 中心遮罩半径（默认 10）
                - 'sigma': 结构张量 sigma 参数（默认 1.0）
                - 'threshold': 结构张量阈值（默认 0.3）
        
        Returns:
            结果字典，包含：
            - 'status': 'success' 或 'error'
            - 'periodicity_nm': 周期性层间距（nm）
            - 'roi_coords': ROI 坐标 (x1, y1, x2, y2)
            - 'scale_factor': 比例尺因子（nm/px）
            - 'preprocess_params': 预处理参数字典
            - 'fft_stats': FFT 统计信息字典
            - 'error_message': 错误信息（如果失败）
            - 'processing_time': 处理时间（秒）
        """
        start_time = time.time()
        self.clear_log()
        
        # 解析配置
        if config is None:
            config = {}
        
        noise_kernel = config.get('noise_kernel', self.preprocess_params.get('noise_kernel', NOISE_REDUCTION_KERNEL))
        contrast_alpha = config.get('contrast_alpha', self.preprocess_params.get('contrast_alpha', CONTRAST_ENHANCEMENT_ALPHA))
        brightness_beta = config.get('brightness_beta', self.preprocess_params.get('brightness_beta', CONTRAST_ENHANCEMENT_BETA))
        scale_length_nm = config.get('scale_length_nm', self.current_scale_length)
        window_size = config.get('window_size', 512)
        mask_radius = config.get('mask_radius', self.mask_radius)
        
        try:
            # 1. 加载图像
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                error_msg = f"图像文件不存在: {image_path}"
                self._add_log(error_msg, "ERROR")
                return {
                    'status': 'error',
                    'error_message': error_msg,
                    'processing_time': time.time() - start_time,
                }
            
            self._add_log(f"开始加载图像: {image_path_obj.name}", "INFO")
            self.current_image = load_image(image_path_obj)
            
            if self.current_image is None:
                error_msg = "图像加载失败，请检查文件格式"
                self._add_log(error_msg, "ERROR")
                return {
                    'status': 'error',
                    'error_message': error_msg,
                    'processing_time': time.time() - start_time,
                }
            
            h, w = self.current_image.shape[:2]
            self._add_log(f"✓ 图像加载成功，尺寸: {w}×{h}", "SUCCESS")
            
            # 同步到 FFT 分析器
            self.fft_analyzer.original_image = self.current_image.copy()
            self.fft_analyzer.gray_image = self.current_image.copy()
            
            # 2. 预处理
            self._add_log("开始图像预处理...", "INFO")
            self.preprocess_params = {
                'noise_kernel': noise_kernel,
                'contrast_alpha': contrast_alpha,
                'brightness_beta': brightness_beta,
            }
            self.processed_image = preprocess_image(
                self.current_image,
                noise_kernel=int(noise_kernel),
                contrast_alpha=float(contrast_alpha),
                brightness_beta=int(brightness_beta),
            )
            self._add_log("✓ 预处理完成", "SUCCESS")
            
            # 3. 自动标尺识别（如果提供了标尺长度）
            scale_bar_box = None  # 保存标尺坐标用于可视化
            if scale_length_nm is not None and scale_length_nm > 0:
                self._add_log(f"开始自动标尺识别（长度={scale_length_nm} nm）...", "INFO")
                bar = detect_scale_bar(self.processed_image)
                if bar is not None:
                    x, y, cw, ch = bar
                    # 计算原图坐标（y 需要加上底部 ROI 的偏移）
                    h, w = self.processed_image.shape[:2]
                    roi_h = max(1, int(h * 0.15))
                    actual_y = h - roi_h + y
                    width_px = cw
                    if width_px > 0:
                        nm_per_pixel = calculate_scale_factor(scale_length_nm, width_px)
                        self.fft_analyzer.nm_per_pixel = nm_per_pixel
                        self.fft_analyzer.scale_length_nm = scale_length_nm
                        self.scale_factor = nm_per_pixel
                        self.current_scale_length = scale_length_nm
                        # 保存标尺坐标用于可视化 (x1, y1, x2, y2)
                        scale_bar_box = (x, actual_y, x + cw, actual_y + ch)
                        self._add_log(f"✓ 自动标尺校准成功: {nm_per_pixel:.4f} nm/px", "SUCCESS")
                    else:
                        self._add_log("标尺宽度检测异常", "WARNING")
                else:
                    self._add_log("未能自动检测到标尺", "WARNING")
            
            # 4. 自动条纹检测（使用结构张量）
            self._add_log("开始自动条纹检测...", "INFO")
            sigma = config.get('sigma', 1.0)
            threshold = config.get('threshold', 0.3)
            x1, y1, x2, y2 = auto_find_roi(
                self.processed_image, 
                window_size=window_size,
                use_structure_tensor=True,
                sigma=sigma,
                threshold=threshold
            )
            
            if x2 <= x1 or y2 <= y1:
                error_msg = "自动 ROI 选择失败"
                self._add_log(error_msg, "ERROR")
                return {
                    'status': 'error',
                    'error_message': error_msg,
                    'processing_time': time.time() - start_time,
                }
            
            self.fft_analyzer.set_roi(x1, y1, x2, y2)
            self.auto_roi_box = (x1, y1, x2, y2)
            self._add_log(f"✓ 自动 ROI 成功：({x1},{y1}) 尺寸 {x2-x1}x{y2-y1}", "SUCCESS")
            
            # 检查比例尺是否已校准
            if self.fft_analyzer.nm_per_pixel <= 0:
                error_msg = "比例尺未校准，请提供 scale_length_nm 配置"
                self._add_log(error_msg, "ERROR")
                return {
                    'status': 'error',
                    'error_message': error_msg,
                    'roi_coords': (x1, y1, x2, y2),
                    'processing_time': time.time() - start_time,
                }
            
            # 5. FFT 分析
            self._add_log("开始 FFT 分析...", "INFO")
            self.mask_radius = mask_radius
            
            fft_shifted, magnitude_log = self.fft_analyzer.compute_fft()
            self._add_log("FFT 计算完成", "SUCCESS")
            
            (peak_x, peak_y), peak_distance = self.fft_analyzer.detect_peak(
                magnitude_log, mask_radius=self.mask_radius
            )
            self._add_log(f"峰值检测完成: 位置=({peak_x:.2f}, {peak_y:.2f}), 距离={peak_distance:.2f} px", "SUCCESS")
            
            periodicity_nm = self.fft_analyzer.calculate_periodicity(peak_distance)
            self.periodicity_nm = periodicity_nm
            self._add_log(f"周期性计算完成: {periodicity_nm:.2f} nm", "SUCCESS")
            
            # 6. 灰度剖面分析（测量单层黑色条纹厚度）
            if self.fft_analyzer.nm_per_pixel > 0 and self.fft_analyzer.roi is not None:
                self._add_log("开始灰度剖面分析（单层厚度测量）...", "INFO")
                try:
                    # 初始化剖面分析器
                    if self.profile_analyzer is None:
                        self.profile_analyzer = ProfileThicknessAnalyzer(
                            self.fft_analyzer.nm_per_pixel
                        )
                    else:
                        self.profile_analyzer.nm_per_pixel = self.fft_analyzer.nm_per_pixel
                    
                    # 计算 FFT 峰值方向作为旋转角度
                    orientation_angle = None
                    if self.fft_analyzer.peak_position is not None:
                        h, w = self.fft_analyzer.fft_magnitude.shape
                        center_x, center_y = w // 2, h // 2
                        peak_x, peak_y = self.fft_analyzer.peak_position
                        # 计算角度（FFT 峰值方向与条纹方向垂直）
                        dy = peak_y - center_y
                        dx = peak_x - center_x
                        if dx != 0 or dy != 0:
                            angle_rad = np.arctan2(dy, dx)
                            orientation_angle = np.degrees(angle_rad) + 90.0
                    
                    # 执行剖面分析（传递周期用于智能过滤异常值）
                    layer_thickness, std_thickness, profile_debug = self.profile_analyzer.analyze_thickness(
                        self.fft_analyzer.roi,
                        orientation_angle=orientation_angle,
                        estimated_period_nm=periodicity_nm if periodicity_nm > 0 else None
                    )
                    
                    if layer_thickness is not None:
                        self.layer_thickness_nm = layer_thickness
                        self.profile_debug_data = profile_debug
                        self._add_log(
                            f"单层厚度测量完成: {layer_thickness:.2f} nm (标准差: {std_thickness:.2f} nm)",
                            "SUCCESS"
                        )
                    else:
                        self._add_log("灰度剖面分析未检测到有效条纹", "WARNING")
                        self.profile_debug_data = profile_debug
                except Exception as e:  # noqa: BLE001
                    self._add_log(f"灰度剖面分析失败: {str(e)}", "WARNING")
            
            # 生成可视化图像（供后续使用）
            result_image = self.fft_analyzer.visualize_results()
            self.fft_result_image = result_image
            
            # 构建结果字典
            processing_time = time.time() - start_time
            fft_stats = {
                'peak_position': {'x': round(peak_x, 2), 'y': round(peak_y, 2)},
                'peak_distance_pixels': round(peak_distance, 2),
                'nm_per_pixel': round(self.fft_analyzer.nm_per_pixel, 4),
                'scale_length_nm': round(self.fft_analyzer.scale_length_nm, 1),
            }
            
            self.statistics = {
                'periodicity_nm': round(periodicity_nm, 2),
                **fft_stats,
            }
            
            self._add_log("✓ 分析完成", "SUCCESS")
            
            # 提取 Plotly 数据
            try:
                fft_data = self.fft_analyzer.get_fft_data_for_plotly()
            except Exception as e:  # noqa: BLE001
                # 如果提取失败，返回 None，不影响主流程
                fft_data = None
                self._add_log(f"Plotly 数据提取失败: {e}", "WARNING")
            
            return {
                'status': 'success',
                'periodicity_nm': round(periodicity_nm, 2),
                'layer_thickness_nm': round(self.layer_thickness_nm, 2) if self.layer_thickness_nm is not None else None,  # 单层厚度
                'roi_coords': (x1, y1, x2, y2),
                'scale_factor': round(self.fft_analyzer.nm_per_pixel, 4),
                'preprocess_params': self.preprocess_params.copy(),
                'fft_stats': fft_stats,
                'fft_data': fft_data,  # 添加 Plotly 数据
                'profile_data': self.profile_debug_data,  # 添加剖面分析数据
                'scale_bar_box': scale_bar_box,  # 添加标尺坐标用于可视化
                'processing_time': round(processing_time, 2),
            }
            
        except Exception as e:  # noqa: BLE001
            import traceback
            error_msg = f"分析过程出错: {str(e)}"
            self._add_log(error_msg, "ERROR")
            self._add_log(traceback.format_exc(), "ERROR")
            return {
                'status': 'error',
                'error_message': error_msg,
                'processing_time': time.time() - start_time,
            }

    # -------- 兼容性方法（保留供旧代码使用，但标记为废弃） --------

    def load_image_file(
        self,
        image_file: object,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str]:
        """加载图像文件（兼容旧接口，已废弃，请使用 analyze_single_image）。"""
        self.clear_log()

        if image_file is None:
            self._add_log("错误: 请先上传图像文件", "ERROR")
            return None, None, "请先上传图像文件", self.get_process_log()

        image_path: Path | str
        if hasattr(image_file, "path"):
            image_path = Path(getattr(image_file, "path"))
        elif isinstance(image_file, (str, Path)):
            image_path = image_file
        elif hasattr(image_file, "name"):
            image_path = Path(getattr(image_file, "name"))
        else:
            image_path = Path(str(image_file))

        image_name = Path(str(image_path)).name
        self._add_log(f"开始加载图像文件: {image_name}", "INFO")
        self.current_image = load_image(image_path)

        if self.current_image is None:
            self._add_log("图像加载失败，请检查文件格式", "ERROR")
            return None, None, "图像加载失败，请检查文件格式", self.get_process_log()

        h, w = self.current_image.shape[:2]
        self._add_log(f"✓ 图像加载成功: {image_name}，尺寸: {w}×{h}", "SUCCESS")

        self.fft_analyzer.original_image = self.current_image.copy()
        self.fft_analyzer.gray_image = self.current_image.copy()

        self._add_log("开始图像预处理...", "INFO")
        self.processed_image = preprocess_image(self.current_image)
        self._add_log("✓ 预处理完成", "SUCCESS")

        msg = f"图像加载成功！\n尺寸: {w} × {h} 像素"
        return None, None, msg, self.get_process_log()

    def run_fft_analysis(self) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
        """执行完整的 FFT 分析流程并返回结果（兼容旧接口，已废弃）。

        Returns:
            Tuple[result_image_array, summary_text, stats_dict]
        """
        if self.processed_image is None:
            self._add_log("错误: 请先加载并预处理图像", "ERROR")
            return None, "请先加载并预处理图像", {}

        if self.fft_analyzer.roi is None:
            self._add_log("错误: 请先选择 ROI 区域", "ERROR")
            return None, "请先选择 ROI 区域", {}

        if self.fft_analyzer.nm_per_pixel <= 0:
            self._add_log("错误: 请先校准比例尺", "ERROR")
            return None, "请先校准比例尺", {}

        try:
            self._add_log("开始 FFT 分析...", "INFO")

            fft_shifted, magnitude_log = self.fft_analyzer.compute_fft()
            self._add_log("FFT 计算完成", "SUCCESS")

            (peak_x, peak_y), peak_distance = self.fft_analyzer.detect_peak(
                magnitude_log, mask_radius=self.mask_radius
            )
            self._add_log(f"峰值检测完成: 位置=({peak_x:.2f}, {peak_y:.2f}), 距离={peak_distance:.2f} px", "SUCCESS")

            periodicity_nm = self.fft_analyzer.calculate_periodicity(peak_distance)
            self.periodicity_nm = periodicity_nm
            self._add_log(f"周期性计算完成: {periodicity_nm:.2f} nm", "SUCCESS")

            result_image = self.fft_analyzer.visualize_results()
            self.fft_result_image = result_image

            summary = (
                f"FFT 分析完成\n"
                f"周期性层间距: {periodicity_nm:.2f} nm\n"
                f"峰值位置: ({peak_x:.2f}, {peak_y:.2f})\n"
                f"峰值距离: {peak_distance:.2f} pixels\n"
                f"比例尺分辨率: {self.fft_analyzer.nm_per_pixel:.4f} nm/pixel"
            )

            stats = {
                "periodicity_nm": round(periodicity_nm, 2),
                "peak_position": {"x": round(peak_x, 2), "y": round(peak_y, 2)},
                "peak_distance_pixels": round(peak_distance, 2),
                "nm_per_pixel": round(self.fft_analyzer.nm_per_pixel, 4),
                "scale_length_nm": round(self.fft_analyzer.scale_length_nm, 1),
            }

            self.statistics = stats
            self._add_log("✓ FFT 分析完成", "SUCCESS")

            return result_image, summary, stats

        except Exception as e:  # noqa: BLE001
            error_msg = f"FFT 分析失败: {e}"
            self._add_log(error_msg, "ERROR")
            import traceback
            self._add_log(traceback.format_exc(), "ERROR")
            return None, error_msg, {}

    # -------- 自动化辅助功能 --------

    def auto_calibrate_scale(self, physical_length_nm: float) -> str:
        """自动识别底部标尺并完成像素比例换算。"""
        if self.processed_image is None:
            msg = "请先加载并预处理图像后再自动校准比例尺"
            self._add_log(msg, "ERROR")
            return msg

        bar = detect_scale_bar(self.processed_image)
        if bar is None:
            msg = "未能自动检测到标尺条，请手动校准"
            self._add_log(msg, "WARNING")
            return msg

        x1, x2 = bar
        width_px = abs(x2 - x1)
        if width_px <= 0:
            msg = "标尺宽度检测异常，自动校准失败"
            self._add_log(msg, "ERROR")
            return msg

        nm_per_pixel = calculate_scale_factor(physical_length_nm, width_px)
        self.fft_analyzer.nm_per_pixel = nm_per_pixel
        self.fft_analyzer.scale_length_nm = physical_length_nm
        self.scale_factor = nm_per_pixel
        self.current_scale_length = physical_length_nm

        msg = (
            f"自动标尺校准成功: {physical_length_nm:.1f} nm / {width_px:.1f} px => "
            f"{nm_per_pixel:.4f} nm/px"
        )
        self._add_log(msg, "SUCCESS")
        return msg

    def auto_select_roi(self, window_size: int = 512) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        """自动寻找高频信息最强的 ROI 并设置到 FFT 分析器。"""
        if self.processed_image is None:
            msg = "请先加载并预处理图像后再自动选择 ROI"
            self._add_log(msg, "ERROR")
            return None, msg

        x1, y1, x2, y2 = auto_find_roi(self.processed_image, window_size=window_size)
        if x2 <= x1 or y2 <= y1:
            msg = "自动 ROI 选择失败，请手动框选"
            self._add_log(msg, "WARNING")
            return None, msg

        self.fft_analyzer.set_roi(x1, y1, x2, y2)
        self.auto_roi_box = (x1, y1, x2, y2)
        msg = f"自动 ROI 成功：起点({x1},{y1}) 尺寸 {x2 - x1}x{y2 - y1}"
        self._add_log(msg, "SUCCESS")
        return self.auto_roi_box, msg

    # -------- 手动测量功能（DigitalMicrograph 风格） --------

    def extract_line_profile(
        self, point1: Tuple[int, int], point2: Tuple[int, int]
    ) -> np.ndarray:
        """从两点之间提取一维灰度剖面。
        
        使用线性插值确保斜线采样准确，避免锯齿状跳变。
        
        Args:
            point1: 起点坐标 (x, y)
            point2: 终点坐标 (x, y)
            
        Returns:
            一维灰度值数组
        """
        if self.processed_image is None and self.current_image is None:
            raise ValueError("图像未加载，无法提取剖面")
        
        # 使用预处理后的图像（如果存在），否则使用原始图像
        image = self.processed_image if self.processed_image is not None else self.current_image
        
        # 确保是灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        x1, y1 = point1
        x2, y2 = point2
        
        # 计算两点间的距离（像素数）
        num_points = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        if num_points < 2:
            num_points = 2
        
        # 直接采样整数像素坐标（不使用插值）
        coords = np.linspace(0, 1, num_points)
        x_coords = np.round(x1 + coords * (x2 - x1)).astype(int)
        y_coords = np.round(y1 + coords * (y2 - y1)).astype(int)
        
        # 确保坐标在图像范围内
        h, w = image.shape
        x_coords = np.clip(x_coords, 0, w - 1)
        y_coords = np.clip(y_coords, 0, h - 1)
        
        # 直接提取像素值（不使用插值）
        profile = image[y_coords, x_coords]
        
        return profile.astype(np.float64)

    def analyze_manual_roi(
        self, roi_rect: Tuple[int, int, int, int], config: Dict
    ) -> Dict[str, Any]:
        """分析手动框选的 ROI 区域（FFT 分析）。
        
        Args:
            roi_rect: 矩形区域 (x, y, w, h) 或 (x1, y1, x2, y2)
            config: 配置字典，包含预处理参数等
            
        Returns:
            结果字典，包含 'status', 'periodicity_nm', 'fft_stats' 等
        """
        if self.processed_image is None:
            return {
                'status': 'error',
                'error_message': '图像未加载，请先上传图像',
            }
        
        if self.fft_analyzer.nm_per_pixel <= 0:
            return {
                'status': 'error',
                'error_message': '比例尺未校准，请先校准比例尺',
            }
        
        try:
            # 解析矩形坐标
            if len(roi_rect) == 4:
                # 可能是 (x, y, w, h) 或 (x1, y1, x2, y2)
                x1, y1, x2_or_w, y2_or_h = roi_rect
                # 判断是哪种格式：如果 x2_or_w < x1 或 y2_or_h < y1，则认为是 (x1, y1, x2, y2)
                if x2_or_w < x1 or y2_or_h < y1:
                    x2, y2 = x2_or_w, y2_or_h
                else:
                    x2, y2 = x1 + x2_or_w, y1 + y2_or_h
            else:
                return {
                    'status': 'error',
                    'error_message': 'ROI 坐标格式错误',
                }
            
            # 确保坐标在图像范围内
            h, w = self.processed_image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            if x2 <= x1 or y2 <= y1:
                return {
                    'status': 'error',
                    'error_message': 'ROI 区域无效（宽度或高度为 0）',
                }
            
            # 设置 ROI
            self.fft_analyzer.set_roi(x1, y1, x2, y2)
            
            # 执行 FFT 分析
            fft_shifted, magnitude_log = self.fft_analyzer.compute_fft()
            
            # 检测峰值
            mask_radius = config.get('mask_radius', self.mask_radius)
            (peak_x, peak_y), peak_distance = self.fft_analyzer.detect_peak(
                magnitude_log, mask_radius=mask_radius
            )
            
            # 计算周期性
            periodicity_nm = self.fft_analyzer.calculate_periodicity(peak_distance)
            
            # 生成可视化图像
            result_image = self.fft_analyzer.visualize_results()
            self.fft_result_image = result_image
            
            # 构建结果
            fft_stats = {
                'peak_position': {'x': round(peak_x, 2), 'y': round(peak_y, 2)},
                'peak_distance_pixels': round(peak_distance, 2),
                'nm_per_pixel': round(self.fft_analyzer.nm_per_pixel, 4),
                'roi_coords': (x1, y1, x2, y2),
            }
            
            # 提取 Plotly 数据
            try:
                fft_data = self.fft_analyzer.get_fft_data_for_plotly()
            except Exception:
                fft_data = None
            
            return {
                'status': 'success',
                'periodicity_nm': round(periodicity_nm, 2),
                'fft_stats': fft_stats,
                'fft_data': fft_data,
            }
            
        except Exception as e:  # noqa: BLE001
            import traceback
            return {
                'status': 'error',
                'error_message': f'ROI 分析失败: {str(e)}\n{traceback.format_exc()}',
            }

    def analyze_manual_line(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        peak_height_min: float = 0.2,
    ) -> Dict[str, Any]:
        """分析手动画线的灰度剖面（单层厚度测量）。

        Args:
            point1: 起点坐标 (x, y)
            point2: 终点坐标 (x, y)
            peak_height_min: 膜/峰检测最小高度（0~1），用于判断黑条纹波谷深度

        Returns:
            结果字典，包含 'status', 'layer_thickness_nm', 'profile_data' 等
        """
        if self.processed_image is None and self.current_image is None:
            return {
                'status': 'error',
                'error_message': '图像未加载，请先上传图像',
            }

        if self.fft_analyzer.nm_per_pixel <= 0:
            return {
                'status': 'error',
                'error_message': '比例尺未校准，请先校准比例尺',
            }

        try:
            profile = self.extract_line_profile(point1, point2)

            if len(profile) < 10:
                return {
                    'status': 'error',
                    'error_message': '剖面长度太短，请画一条更长的线',
                }

            if self.profile_analyzer is None:
                self.profile_analyzer = ProfileThicknessAnalyzer(
                    self.fft_analyzer.nm_per_pixel,
                    peak_height_min=peak_height_min,
                )
            else:
                self.profile_analyzer.nm_per_pixel = self.fft_analyzer.nm_per_pixel
                self.profile_analyzer.peak_height_min = peak_height_min
            
            # 使用估计的周期（如果有）来过滤异常值
            estimated_period_nm = None
            if self.periodicity_nm is not None:
                estimated_period_nm = self.periodicity_nm
            
            # 调用 1D 分析函数
            mean_thick, std_thick, debug_data = self.profile_analyzer.analyze_profile_1d(
                profile, estimated_period_nm=estimated_period_nm, invert_signal=True
            )
            
            # 将原始灰度 profile 添加到 debug_data 中（用于直接显示图像灰度）
            debug_data['profile_raw'] = profile.tolist()  # 原始灰度值
            
            if mean_thick is None:
                return {
                    'status': 'error',
                    'error_message': '未检测到有效的黑色条纹，请尝试调整画线位置',
                    'profile_data': debug_data,
                    'profile_raw': profile.tolist(),  # 即使失败也返回原始数据
                }
            
            return {
                'status': 'success',
                'layer_thickness_nm': round(mean_thick, 2),
                'std_thickness_nm': round(std_thick, 2) if std_thick is not None else None,
                'profile_data': debug_data,
                'profile_raw': profile.tolist(),  # 原始灰度值，用于直接显示
            }
            
        except Exception as e:  # noqa: BLE001
            import traceback
            return {
                'status': 'error',
                'error_message': f'Line Profile 分析失败: {str(e)}\n{traceback.format_exc()}',
            }





