"""
FFT 周期性分析核心实现。

本文件迁移自旧版 `tem_calibration.fft_analysis.FFTPeriodicityAnalyzer`，
不包含任何 UI 逻辑，仅负责频谱计算与结果可视化所需的数据生成。
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import warnings

# 禁用 matplotlib 字体警告（必须在导入 matplotlib 之前）
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, message=".*font.*")

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import peak_local_max

matplotlib.use("Agg")  # 非交互式后端，便于在打包环境中使用

# 先导入 warnings 并配置警告过滤器
import warnings
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, message=".*font.*")

# 配置中文字体支持
import platform
if platform.system() == "Windows":
    # Windows 系统字体路径
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/msyh.ttc",         # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
    ]
    # 尝试设置中文字体
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                from matplotlib.font_manager import FontProperties
                prop = FontProperties(fname=font_path)
                matplotlib.rcParams["font.family"] = prop.get_name()
                matplotlib.rcParams["font.sans-serif"] = [prop.get_name(), "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
                break
            except Exception:
                continue
    else:
        # 如果找不到字体文件，使用系统默认配置
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
else:
    # Linux/Mac 系统
    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"]

matplotlib.rcParams["axes.unicode_minus"] = False


class FFTPeriodicityAnalyzer:
    """使用 2D FFT 分析 TEM 图像周期性层间距。"""

    def __init__(self) -> None:
        self.original_image: Optional[np.ndarray] = None
        self.gray_image: Optional[np.ndarray] = None
        self.roi: Optional[np.ndarray] = None
        self.roi_coords: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
        self.scale_point1: Optional[Tuple[int, int]] = None
        self.scale_point2: Optional[Tuple[int, int]] = None
        self.nm_per_pixel: float = 0.0
        self.scale_length_nm: float = 50.0
        self.periodicity_nm: Optional[float] = None
        self.fft_magnitude: Optional[np.ndarray] = None
        self.fft_shifted: Optional[np.ndarray] = None  # 保存复数形式的 FFT 结果，用于 IFFT 重构
        # 记录 FFT 计算时使用的正方形边长（用于后续周期性计算）
        self._fft_calc_size: Optional[int] = None
        # 峰位置允许为浮点坐标（亚像素）
        self.peak_position: Optional[Tuple[float, float]] = None

    # -------- 图像与比例尺设置 --------

    def load_image(self, image_path: str | Path | object) -> Optional[np.ndarray]:
        """加载图像并转换为灰度图。"""
        try:
            path = Path(str(image_path))

            # TIF 优先尝试 tifffile
            if path.suffix.lower() in [".tif", ".tiff"]:
                try:
                    import tifffile

                    img = tifffile.imread(str(path))
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                except Exception:  # noqa: BLE001
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                return None

            self.original_image = img.copy()
            self.gray_image = img.copy()
            return self.gray_image
        except Exception as e:  # noqa: BLE001
            print(f"加载图像失败: {e}")
            return None

    def set_scale_points(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        scale_length_nm: float,
    ) -> float:
        """设置比例尺两点并计算 nm_per_pixel。"""
        self.scale_point1 = point1
        self.scale_point2 = point2
        self.scale_length_nm = scale_length_nm

        pixel_distance = float(
            np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        )
        self.nm_per_pixel = scale_length_nm / pixel_distance if pixel_distance > 0 else 0.0
        return self.nm_per_pixel

    def set_roi(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """设置感兴趣区域（ROI）。"""
        if self.gray_image is None:
            raise ValueError("请先加载图像")

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        h, w = self.gray_image.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        self.roi_coords = (x1, y1, x2, y2)
        self.roi = self.gray_image[y1:y2, x1:x2].copy()
        return self.roi

    # -------- FFT 计算与峰值检测 --------

    @staticmethod
    def _apply_hanning_window(image: np.ndarray) -> np.ndarray:
        """对图像应用 Hanning 窗以减少边缘效应。"""
        h, w = image.shape
        hanning_y = np.hanning(h)
        hanning_x = np.hanning(w)
        hanning_2d = np.outer(hanning_y, hanning_x)
        windowed = image.astype(np.float64) * hanning_2d
        return windowed.astype(np.uint8)

    def compute_fft(self, roi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算 2D FFT 并返回移位结果与对数幅度谱（强制使用正方形 ROI）。"""
        if roi is None:
            if self.roi is None:
                raise ValueError("请先设置 ROI 区域")
            roi = self.roi

        # 强制裁剪为中心正方形，保证 Nx = Ny，使 d = N / R 公式成立
        h, w = roi.shape
        min_side = min(h, w)
        start_y = (h - min_side) // 2
        start_x = (w - min_side) // 2
        square_roi = roi[start_y : start_y + min_side, start_x : start_x + min_side]

        windowed_roi = self._apply_hanning_window(square_roi)
        fft_result = np.fft.fft2(windowed_roi.astype(np.float64))
        fft_shifted = np.fft.fftshift(fft_result)

        magnitude = np.abs(fft_shifted)
        magnitude_log = 20 * np.log(magnitude + 1.0)
        self.fft_magnitude = magnitude_log
        self.fft_shifted = fft_shifted  # 保存复数形式的 FFT 结果
        # 记录参与 FFT 计算的有效边长
        self._fft_calc_size = min_side
        return fft_shifted, magnitude_log

    def detect_peak(
        self,
        magnitude_spectrum: np.ndarray,
        mask_radius: Optional[int] = None,
    ) -> Tuple[Tuple[float, float], float]:
        """检测 FFT 幅度谱中的主峰（衍射点，含亚像素细化）。"""
        h, w = magnitude_spectrum.shape
        center_x, center_y = w // 2, h // 2

        # 动态计算中心遮罩半径：默认取边长的约 3%，并保证不小于 3 像素
        if mask_radius is None:
            mask_radius = int(min(h, w) * 0.03)
            mask_radius = max(3, mask_radius)

        mask = np.ones((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), mask_radius, 0, -1)

        masked = magnitude_spectrum.copy()
        masked[mask == 0] = 0

        # 粗定位：整数像素最大值
        _, _, _, max_loc = cv2.minMaxLoc(masked)
        peak_x_int, peak_y_int = max_loc

        # 亚像素细化：使用 5x5 邻域的质心法
        win_size = 2
        x_start = max(0, peak_x_int - win_size)
        x_end = min(w, peak_x_int + win_size + 1)
        y_start = max(0, peak_y_int - win_size)
        y_end = min(h, peak_y_int + win_size + 1)

        roi_patch = masked[y_start:y_end, x_start:x_end]
        moments = cv2.moments(roi_patch)
        if moments["m00"] != 0:
            offset_x = moments["m10"] / moments["m00"]
            offset_y = moments["m01"] / moments["m00"]
            peak_x = x_start + offset_x
            peak_y = y_start + offset_y
        else:
            peak_x, peak_y = float(peak_x_int), float(peak_y_int)

        self.peak_position = (peak_x, peak_y)

        distance = float(np.sqrt((peak_x - center_x) ** 2 + (peak_y - center_y) ** 2))
        return (peak_x, peak_y), distance

    def detect_multiple_peaks(
        self,
        magnitude_spectrum: np.ndarray,
        min_distance: int = 10,
        threshold_rel: float = 0.1,
        exclude_center_radius: Optional[int] = None,
        max_peaks: int = 10,
    ) -> List[Dict[str, Any]]:
        """检测 FFT 幅度谱中的多个衍射峰（多峰检测）。
        
        Args:
            magnitude_spectrum: FFT 对数幅度谱
            min_distance: 峰与峰之间的最小像素距离（默认 10）
            threshold_rel: 相对阈值，过滤掉噪音（默认 0.1，即最大值的 10%）
            exclude_center_radius: 中心屏蔽半径（像素），用于掩盖 DC 分量。
                                  如果为 None，则自动计算为边长的 3%
            max_peaks: 最大返回峰值数量（默认 10）
            
        Returns:
            峰信息列表，每个元素包含：
            - 'position': (x, y) 以图像中心为原点的相对坐标（浮点数，支持亚像素）
            - 'distance_px': 距离中心的像素距离
            - 'intensity': 峰值强度（对数幅度值）
        """
        h, w = magnitude_spectrum.shape
        center_x, center_y = w // 2, h // 2
        
        # 自动计算中心屏蔽半径（如果未提供）
        if exclude_center_radius is None:
            exclude_center_radius = int(min(h, w) * 0.03)
            exclude_center_radius = max(3, exclude_center_radius)
        
        # 创建中心屏蔽 Mask
        masked_spectrum = magnitude_spectrum.copy()
        mask = np.ones((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), exclude_center_radius, 0, -1)
        masked_spectrum[mask == 0] = 0
        
        # 计算相对阈值（基于屏蔽后的最大值）
        max_val = np.max(masked_spectrum)
        threshold_abs = max_val * threshold_rel
        
        # 使用 peak_local_max 检测局部极大值
        # peak_local_max 返回的是 (row, col) 格式，即 (y, x) 格式
        peaks_coords = peak_local_max(
            masked_spectrum,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            num_peaks=max_peaks,
        )
        
        # 转换为以中心为原点的相对坐标，并进行亚像素优化
        peaks_info = []
        
        for peak_row, peak_col in peaks_coords:
            # peak_local_max 返回的是整数坐标 (row, col) = (y, x)
            peak_y_int, peak_x_int = int(peak_row), int(peak_col)
            
            # 亚像素优化：使用质心法
            win_size = 2
            x_start = max(0, peak_x_int - win_size)
            x_end = min(w, peak_x_int + win_size + 1)
            y_start = max(0, peak_y_int - win_size)
            y_end = min(h, peak_y_int + win_size + 1)
            
            roi_patch = masked_spectrum[y_start:y_end, x_start:x_end]
            moments = cv2.moments(roi_patch)
            
            if moments["m00"] != 0:
                offset_x = moments["m10"] / moments["m00"]
                offset_y = moments["m01"] / moments["m00"]
                peak_x = x_start + offset_x
                peak_y = y_start + offset_y
            else:
                peak_x, peak_y = float(peak_x_int), float(peak_y_int)
            
            # 计算距离中心的像素距离
            dx = peak_x - center_x
            dy = peak_y - center_y
            distance_px = float(np.sqrt(dx**2 + dy**2))
            
            # 获取峰值强度（使用亚像素位置的双线性插值，或直接使用整数位置的值）
            if 0 <= int(peak_y) < h and 0 <= int(peak_x) < w:
                intensity = float(masked_spectrum[int(peak_y), int(peak_x)])
            else:
                intensity = 0.0
            
            peaks_info.append({
                'position': (peak_x, peak_y),  # 绝对坐标 (x, y)
                'distance_px': distance_px,
                'intensity': intensity,
            })
        
        # 按距离排序（从近到远）
        peaks_info.sort(key=lambda p: p['distance_px'])
        
        return peaks_info

    def calculate_periodicity(
        self,
        peak_distance_pixels: float,
        roi_size: Optional[int] = None,
    ) -> float:
        """根据峰值位置与 ROI 尺寸计算周期性层间距（纳米）。"""
        if peak_distance_pixels <= 1e-5:
            return 0.0

        if roi_size is None:
            # 优先使用 FFT 计算时记录的正方形边长
            if self._fft_calc_size is not None:
                roi_size = self._fft_calc_size
            elif self.roi is not None:
                roi_size = min(self.roi.shape[:2])
            else:
                raise ValueError("请先设置 ROI 区域或执行 FFT 计算")

        d_pixels = roi_size / peak_distance_pixels
        if self.nm_per_pixel <= 0:
            raise ValueError("请先校准比例尺（nm_per_pixel <= 0）")

        d_nm = d_pixels * self.nm_per_pixel
        self.periodicity_nm = d_nm
        return d_nm

    # -------- Plotly 数据提取 --------

    def get_fft_data_for_plotly(self) -> Dict[str, Any]:
        """提取 FFT 数据供 Plotly 可视化使用。
        
        Returns:
            包含以下键的字典：
            - 'fft_magnitude': 2D FFT 幅度谱数组
            - 'roi': ROI 区域图像
            - 'profile_data': 径向 Profile 曲线数据 (distance, intensity)
            - 'peak_position': 峰值位置 (x, y)
            - 'peak_distance': 峰值距离（像素）
            - 'center': FFT 中心位置 (x, y)
        """
        if self.roi is None or self.fft_magnitude is None:
            raise ValueError("请先执行 FFT 分析")
        
        h, w = self.fft_magnitude.shape
        center_x, center_y = w // 2, h // 2
        
        # 计算径向 Profile（从中心到峰值的径向强度分布）
        profile_data = None
        peak_distance = None
        
        if self.peak_position is not None:
            peak_x, peak_y = self.peak_position
            # 计算从中心到峰值的距离
            peak_distance = float(np.sqrt((peak_x - center_x)**2 + (peak_y - center_y)**2))
            max_distance = int(peak_distance) + 20
            
            # 创建径向采样
            distances = []
            intensities = []
            
            for r in range(0, max_distance, 1):
                # 在径向方向上采样多个角度，取平均值
                angles = np.linspace(0, 2 * np.pi, 36)  # 每10度采样一次
                sample_intensities = []
                
                for angle in angles:
                    x = int(center_x + r * np.cos(angle))
                    y = int(center_y + r * np.sin(angle))
                    
                    if 0 <= x < w and 0 <= y < h:
                        sample_intensities.append(float(self.fft_magnitude[y, x]))
                
                if sample_intensities:
                    distances.append(float(r))
                    intensities.append(float(np.mean(sample_intensities)))
            
            profile_data = {
                'distance': distances,
                'intensity': intensities
            }
        
        return {
            'fft_magnitude': self.fft_magnitude.copy(),
            'roi': self.roi.copy() if self.roi is not None else None,
            'profile_data': profile_data,
            'peak_position': self.peak_position,
            'peak_distance': peak_distance,
            'center': (center_x, center_y),
            'nm_per_pixel': self.nm_per_pixel,
            'periodicity_nm': self.periodicity_nm,
        }

    def reconstruct_lattice_image(
        self,
        peak_radius: int = 3,
        dc_radius: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """使用 IFFT 重构晶格条纹图像，用于验证算法提取的信号是否正确。
        
        原理：只保留 FFT 频谱中的中心直流分量(DC)和检测到的主衍射峰，滤除其他噪声，
        然后执行逆傅里叶变换，得到"提纯"后的条纹图像。
        
        Args:
            peak_radius: 峰值周围的保留半径（像素），默认 3
            dc_radius: 中心直流分量的保留半径，如果为 None 则使用 mask_radius
            
        Returns:
            重构的晶格图像（灰度图），如果 FFT 未计算则返回 None
        """
        if self.fft_shifted is None or self.roi is None:
            return None
        
        h, w = self.fft_shifted.shape
        center_x, center_y = w // 2, h // 2
        
        # 创建掩膜：只保留 DC 和峰值
        mask = np.ones((h, w), dtype=np.float64)
        
        # 先全部置零
        mask.fill(0.0)
        
        # 保留中心直流分量
        if dc_radius is None:
            dc_radius = max(3, int(min(h, w) * 0.03))
        
        cv2.circle(mask, (center_x, center_y), dc_radius, 1.0, -1)
        
        # 保留检测到的主峰
        if self.peak_position is not None:
            peak_x, peak_y = self.peak_position
            peak_x_int = int(round(peak_x))
            peak_y_int = int(round(peak_y))
            
            # 在峰值位置创建圆形掩膜
            cv2.circle(mask, (peak_x_int, peak_y_int), peak_radius, 1.0, -1)
            
            # 如果有对称峰（对于周期性结构，通常会有对称的峰）
            # 计算对称位置
            sym_x = 2 * center_x - peak_x_int
            sym_y = 2 * center_y - peak_y_int
            if 0 <= sym_x < w and 0 <= sym_y < h:
                cv2.circle(mask, (sym_x, sym_y), peak_radius, 1.0, -1)
        
        # 应用掩膜到 FFT 频谱
        fft_filtered = self.fft_shifted * mask
        
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

    # -------- 可视化辅助（返回 numpy 数组，供 UI 层显示） --------

    def visualize_results(self) -> np.ndarray:
        """创建 3 子图可视化结果图像并以 RGB 数组返回。"""
        if self.roi is None or self.fft_magnitude is None:
            raise ValueError("请先执行 FFT 分析")

        # 获取中文字体属性（如果可用）
        title_font = None
        try:
            from matplotlib.font_manager import FontProperties
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simsun.ttc",
            ]
            for font_path in font_paths:
                if Path(font_path).exists():
                    try:
                        title_font = FontProperties(fname=font_path, size=12)
                        break
                    except Exception:
                        continue
        except Exception:
            pass

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 子图 1：ROI
        axes[0].imshow(self.roi, cmap="gray")
        if title_font:
            axes[0].set_title("选择的 ROI 区域", fontproperties=title_font)
        else:
            axes[0].set_title("ROI Region", fontsize=12)
        axes[0].axis("off")

        # 子图 2：FFT 幅度谱
        h, w = self.fft_magnitude.shape
        center_x, center_y = w // 2, h // 2

        im = axes[1].imshow(self.fft_magnitude, cmap="hot")
        if title_font:
            axes[1].set_title("FFT 幅度谱", fontproperties=title_font)
        else:
            axes[1].set_title("FFT Magnitude Spectrum", fontsize=12)

        if title_font:
            axes[1].plot(center_x, center_y, "b+", markersize=15, markeredgewidth=2, label="中心(DC)")
            if self.peak_position is not None:
                peak_x, peak_y = self.peak_position
                axes[1].plot(peak_x, peak_y, "rx", markersize=15, markeredgewidth=2, label="衍射峰")
                axes[1].plot(
                    [center_x, peak_x],
                    [center_y, peak_y],
                    "g--",
                    linewidth=2,
                    alpha=0.7,
                )
        else:
            axes[1].plot(center_x, center_y, "b+", markersize=15, markeredgewidth=2, label="Center (DC)")
            if self.peak_position is not None:
                peak_x, peak_y = self.peak_position
                axes[1].plot(peak_x, peak_y, "rx", markersize=15, markeredgewidth=2, label="Diffraction Peak")
                axes[1].plot(
                    [center_x, peak_x],
                    [center_y, peak_y],
                    "g--",
                    linewidth=2,
                    alpha=0.7,
                )
        axes[1].legend()
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])

        # 子图 3：文本结果
        axes[2].axis("off")
        
        # 根据字体可用性选择文本语言
        if title_font:
            result_text = "FFT 分析结果\n" + "=" * 30 + "\n\n"
            result_text += f"比例尺长度: {self.scale_length_nm:.1f} nm\n"
            result_text += f"像素分辨率: {self.nm_per_pixel:.4f} nm/pixel\n\n"
            
            if self.peak_position is not None:
                peak_x, peak_y = self.peak_position
                distance = float(
                    np.sqrt((peak_x - center_x) ** 2 + (peak_y - center_y) ** 2)
                )
                result_text += f"峰值位置: ({peak_x:.2f}, {peak_y:.2f})\n"
                result_text += f"峰值距离: {distance:.2f} pixels\n\n"
            
            if self.periodicity_nm is not None:
                result_text += "平均周期性层间距:\n"
                result_text += f"{self.periodicity_nm:.2f} nm\n"
            else:
                result_text += "未计算周期性"
            
            axes[2].text(
                0.1,
                0.5,
                result_text,
                fontsize=12,
                verticalalignment="center",
                fontproperties=title_font,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            # 使用英文避免字体警告
            result_text = "FFT Analysis Results\n" + "=" * 30 + "\n\n"
            result_text += f"Scale Length: {self.scale_length_nm:.1f} nm\n"
            result_text += f"Pixel Resolution: {self.nm_per_pixel:.4f} nm/pixel\n\n"
            
            if self.peak_position is not None:
                peak_x, peak_y = self.peak_position
                distance = float(
                    np.sqrt((peak_x - center_x) ** 2 + (peak_y - center_y) ** 2)
                )
                result_text += f"Peak Position: ({peak_x:.2f}, {peak_y:.2f})\n"
                result_text += f"Peak Distance: {distance:.2f} pixels\n\n"
            
            if self.periodicity_nm is not None:
                result_text += "Average Periodicity:\n"
                result_text += f"{self.periodicity_nm:.2f} nm\n"
            else:
                result_text += "Periodicity not calculated"
            
            axes[2].text(
                0.1,
                0.5,
                result_text,
                fontsize=12,
                verticalalignment="center",
                family="sans-serif",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        # 在绘制前再次设置警告过滤器，确保捕获所有警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*font.*")
            fig.canvas.draw()
        
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape((height, width, 4))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        plt.close(fig)

        return img_rgb

    # -------- 标注辅助（仍然返回 numpy 数组，由 UI 层显示） --------

    def _put_chinese_text(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_size: int = 20,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """在 OpenCV 图像上绘制中文文字（使用 PIL）。"""
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_image)

        font = ImageFont.load_default()
        try:
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simsun.ttc",
            ]
            for font_path in font_paths:
                if Path(font_path).exists():
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()

        rgb_color = (color[2], color[1], color[0])
        x, y = position
        draw.text(
            (x, y),
            text,
            font=font,
            fill=rgb_color,
            stroke_width=max(1, thickness // 2),
        )

        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return img_bgr


