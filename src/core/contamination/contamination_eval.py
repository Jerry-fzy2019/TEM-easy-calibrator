"""
污染率表征核心模块（基于轮廓的形态测量法）。

实现不规则圆形区域（如积碳斑或孔洞）的生长测量：
- 使用轮廓提取法自动识别目标区域
- 计算面积和等效直径
- 追踪随时间的变化趋势
"""

from __future__ import annotations

import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class ContaminationAnalyzer:
    """污染率分析器（基于轮廓提取）。"""

    def __init__(self) -> None:
        """初始化分析器。"""
        pass

    def _load_image(self, path: Path) -> np.ndarray:
        """读取单张图像，强制灰度模式。
        
        Args:
            path: 图像路径
            
        Returns:
            灰度图像数组
            
        Raises:
            ValueError: 当文件不存在或读取失败时抛出
        """
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            raise ValueError(f"无法读取图像文件: {path}")
        return img

    def _to_path_list(self, image_path: str | Path | Sequence[str | Path]) -> List[Path]:
        """将输入转换为路径列表，确保为序列。"""
        if isinstance(image_path, (str, Path)):
            raise ValueError("请提供包含多帧的图像序列路径列表，而非单个文件。")
        try:
            paths = [Path(p) for p in image_path]
        except Exception as exc:  # noqa: BLE001
            raise ValueError("image_path 必须是路径序列。") from exc
        if len(paths) < 1:
            raise ValueError("至少需要一帧图像。")
        return paths

    def _preprocess_frame(
        self,
        image: np.ndarray,
        target_type: str = "dark",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """预处理单帧图像，提取目标区域。
        
        Args:
            image: 输入灰度图像
            target_type: 'dark' 表示检测黑斑（亮背景），'bright' 表示检测亮孔（黑背景）
            
        Returns:
            (二值化图像, 原始图像副本)
        """
        # 1. 降噪（尽量保留细节边缘，核稍小一点）
        denoised = cv2.medianBlur(image, 3)
        
        # 2. Otsu 自动阈值
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. 极性判断与调整
        # 检查图像中心区域的像素值，判断目标区域是黑还是白
        h, w = binary.shape
        center_roi = binary[h//4:h*3//4, w//4:w*3//4]
        center_mean = np.mean(center_roi)
        
        # 如果目标是黑斑（dark），但中心是白的，需要反转
        # 如果目标是亮孔（bright），但中心是黑的，需要反转
        if target_type == "dark" and center_mean > 127:
            binary = cv2.bitwise_not(binary)
        elif target_type == "bright" and center_mean < 127:
            binary = cv2.bitwise_not(binary)
        
        # 4. 形态学优化：填充小孔洞，同时尽量贴合真实边缘（核缩小、迭代次数减小）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return binary, image.copy()

    def _find_largest_contour(
        self,
        binary: np.ndarray,
        min_area: int = 100,
    ) -> Optional[Tuple[np.ndarray, float, float, float]]:
        """找到面积最大的轮廓。
        
        Args:
            binary: 二值化图像
            min_area: 最小轮廓面积（像素），用于过滤噪点
            
        Returns:
            (轮廓点集, 面积, 等效直径, 圆形度) 或 None
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # 找到面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < min_area:
            return None
        
        # 计算等效圆直径
        equivalent_diameter = 2 * np.sqrt(area / np.pi)
        
        # 计算圆形度 (4π*Area / Perimeter^2)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0.0
        
        return largest_contour, area, equivalent_diameter, circularity

    def analyze_single_image(
        self,
        image_path: str | Path | List[str | Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """分析图像序列中不规则圆形区域的生长。
        
        Args:
            image_path: 图像序列路径列表
            config: 配置字典，可包含:
                - 'target_type': 'dark' 或 'bright'，默认 'dark'
                - 'min_area': 最小轮廓面积（像素），默认 100
                - 'show_overlay': 是否生成叠加图像，默认 True
                
        Returns:
            结果字典，包含:
                - 'status': 'success' 或 'error'
                - 'growth_curve': 生长曲线数据列表
                - 'overlay_images': 叠加了轮廓的代表性图像（可选）
                - 'processing_time': 处理时间（秒）
                - 'error_message': 错误信息（如果失败）
        """
        start_time = time.time()
        config = config or {}
        
        target_type = config.get("target_type", "dark")
        min_area = int(config.get("min_area", 100))
        show_overlay = config.get("show_overlay", True)
        nm_per_pixel = float(config.get("nm_per_pixel", 0.0) or 0.0)
        
        try:
            paths = self._to_path_list(image_path)
            
            growth_curve: List[Dict[str, Any]] = []
            overlay_images: List[np.ndarray] = []
            
            # 逐帧处理
            for frame_idx, path in enumerate(paths):
                img = self._load_image(path)
                binary, original = self._preprocess_frame(img, target_type=target_type)
                
                result = self._find_largest_contour(binary, min_area=min_area)
                
                if result is None:
                    # 未检测到目标
                    growth_curve.append({
                        "frame": frame_idx,
                        "area_px": 0.0,
                        "diameter_px": 0.0,
                        "circularity": 0.0,
                    })
                    continue
                
                contour, area, diameter, circularity = result
                
                growth_curve.append({
                    "frame": frame_idx,
                    "area_px": float(area),
                    "diameter_px": float(diameter),
                    "circularity": float(circularity),
                })
                
                # 生成叠加图像（首帧、中间、尾帧）
                if show_overlay:
                    total_frames = len(paths)
                    if (
                        frame_idx == 0
                        or frame_idx == total_frames // 2
                        or frame_idx == total_frames - 1
                    ):
                        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
                        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
                        # 标注等效直径和实际面积（像素 & 物理单位），避免使用上标字符以防显示为 "???"
                        text_px = f"Frame {frame_idx}: D={diameter:.1f}px, A={area:.0f}px^2"
                        cv2.putText(
                            overlay,
                            text_px,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        if nm_per_pixel > 0:
                            d_nm = diameter * nm_per_pixel
                            area_nm2 = area * (nm_per_pixel ** 2)
                            text_nm = f"D={d_nm:.2f}nm, A={area_nm2:.1f}nm^2"
                            cv2.putText(
                                overlay,
                                text_nm,
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )
                        overlay_images.append(overlay)
            
            return {
                "status": "success",
                "growth_curve": growth_curve,
                "overlay_images": overlay_images if show_overlay else [],
                "processing_time": round(time.time() - start_time, 3),
            }
        
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "processing_time": round(time.time() - start_time, 3),
            }


# 保留旧接口以兼容
def evaluate_contamination(image: np.ndarray) -> Dict[str, Any]:
    """占位函数：对单张 TEM 图像进行污染率表征（已废弃，请使用 ContaminationAnalyzer）。"""
    raise NotImplementedError("污染率表征算法已迁移至 ContaminationAnalyzer.analyze_single_image")
