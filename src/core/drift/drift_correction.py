"""
图像漂移 / 畸变校准核心模块。

实现多帧 TEM 序列之间的平移估计与亚像素配准。
"""

from __future__ import annotations

import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation


class DriftAnalyzer:
    """图像漂移/畸变分析器。

    基于相位相关（Phase Correlation）的序列漂移估计与校正。
    读取 8/16-bit TEM 图像，支持亚像素精度漂移估计，并返回校正后的序列。
    """

    def __init__(self) -> None:
        """初始化分析器。"""
        # 预留未来配置或缓存位置
        pass

    def _load_image(self, path: Path) -> np.ndarray:
        """读取单张图像，兼容 16-bit。

        Args:
            path: 图像路径

        Returns:
            读取的图像数组

        Raises:
            ValueError: 当文件不存在或读取失败时抛出，错误消息包含文件名。
        """
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            raise ValueError(f"无法读取图像文件: {path}")
        return img

    def _to_path_list(self, image_path: str | Path | Sequence[str | Path]) -> List[Path]:
        """将输入转换为路径列表，确保为序列。"""
        if isinstance(image_path, (str, Path)):
            # 单文件不允许，提示需要序列
            raise ValueError("请提供包含多帧的图像序列路径列表，而非单个文件。")
        try:
            paths = [Path(p) for p in image_path]
        except Exception as exc:  # noqa: BLE001
            raise ValueError("image_path 必须是路径序列。") from exc
        if len(paths) < 2:
            raise ValueError("漂移校准需要至少两帧图像。")
        return paths

    def analyze_single_image(
        self,
        image_path: str | Path | List[str | Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """分析图像序列的漂移/畸变并返回校正结果。

        流程：
            1. 校验并读取序列，第一帧作为参考。
            2. 对后续帧使用相位相关计算相对参考的漂移 (dy, dx)。
            3. 使用 scipy.ndimage.shift 反向平移得到校正图像。

        Args:
            image_path: 图像序列路径列表（必须至少两帧）
            config: 配置字典，可包含:
                - 'upsample_factor': int，亚像素放大倍率，默认 100

        Returns:
            包含以下键的结果字典：
                - 'status': 'success' or 'error'
                - 'drift_vectors': List[Tuple[float, float]]，每帧相对参考的 (dy, dx)，首帧为 (0,0)
                - 'corrected_images': List[np.ndarray]，校正后的图像序列
                - 'processing_time': float，耗时（秒）
                - 'error_message': 若失败则包含错误信息
                - 'traceback': 若失败则包含详细 traceback
        """
        start_time = time.time()
        config = config or {}
        upsample_factor = int(config.get("upsample_factor", 100))

        try:
            paths = self._to_path_list(image_path)

            # 读取全部帧
            images: List[np.ndarray] = [self._load_image(p) for p in paths]
            ref = images[0]

            drift_vectors: List[Tuple[float, float]] = [(0.0, 0.0)]
            corrected_images: List[np.ndarray] = [ref]

            # 后续帧漂移估计与校正
            for idx, img in enumerate(images[1:], start=1):
                shift, _, _ = phase_cross_correlation(
                    ref, img, upsample_factor=upsample_factor
                )
                dy, dx = float(shift[0]), float(shift[1])
                drift_vectors.append((dy, dx))

                # 反向平移校正（将 moving 图对齐到 ref）
                # nd_shift 的 shift 维度需与输入一致，彩色图为 (H, W, C)
                if img.ndim == 2:
                    shift_tuple = (-dy, -dx)
                else:
                    shift_tuple = (-dy, -dx) + (0,) * (img.ndim - 2)
                corrected = nd_shift(img, shift=shift_tuple, order=3, mode="nearest")
                corrected_images.append(corrected)

            return {
                "status": "success",
                "drift_vectors": drift_vectors,
                "corrected_images": corrected_images,
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
def estimate_drift(frames: List[np.ndarray]) -> Dict[str, Any]:
    """占位函数：估计多帧 TEM 图像的漂移/畸变（已废弃，请使用 DriftAnalyzer）。"""
    raise NotImplementedError("图像漂移/畸变校准算法已迁移至 DriftAnalyzer.analyze_single_image")


def correct_drift(frames: List[np.ndarray], drift_field: Any) -> List[np.ndarray]:
    """占位函数：根据估计的漂移场对图像进行校正（已废弃，请使用 DriftAnalyzer）。"""
    raise NotImplementedError("图像漂移/畸变校正算法已迁移至 DriftAnalyzer.analyze_single_image")
