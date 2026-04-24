"""
自动识别标尺与自动寻找 ROI 的视觉辅助函数。

仅依赖 numpy 与 OpenCV，供 core 层调用，UI 不直接参与图像处理。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def detect_scale_bar(image: np.ndarray) -> Optional[Tuple[int, int]]:
    """自动检测 TEM 图像底部标尺，返回 (start_x, end_x)。支持黑底白字/白底黑字。"""
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    roi_h = max(1, int(h * 0.15))
    bottom_roi = gray[h - roi_h : h, :]

    candidates = []
    # 兼容黑白反色标尺，尝试两种阈值模式
    for method in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, binary = cv2.threshold(bottom_roi, 0, 255, method + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch < 5 or cw < 20:
                continue
            aspect = cw / float(ch)
            # 过滤：长宽比适中，且不占满整幅
            if 3 < aspect < 50 and cw < w * 0.9:
                candidates.append((x, y, cw, ch))

    if not candidates:
        return None

    # 取最宽的候选作为标尺
    x, y, cw, ch = max(candidates, key=lambda t: t[2])
    return x, x + cw


def auto_find_roi(image: np.ndarray, window_size: int = 512) -> Tuple[int, int, int, int]:
    """基于清晰度（拉普拉斯方差）自动寻找最佳 ROI，使用降采样加速。"""
    if image is None or image.size == 0:
        return 0, 0, 0, 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]

    # 过大图像先降采样，提高速度
    scale = 1.0
    work_img = gray
    if max(h, w) > 1000:
        scale = 1000.0 / max(h, w)
        work_img = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

    scaled_h, scaled_w = work_img.shape
    scaled_win = max(32, min(int(window_size * scale), scaled_h, scaled_w))
    if scaled_win <= 1:
        return 0, 0, 0, 0

    step = max(1, scaled_win // 2)
    best_score = -1.0
    best_coords = (0, 0, scaled_win, scaled_win)

    for y in range(0, scaled_h - scaled_win + 1, step):
        for x in range(0, scaled_w - scaled_win + 1, step):
            patch = work_img[y : y + scaled_win, x : x + scaled_win]
            score = cv2.Laplacian(patch, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_coords = (x, y, x + scaled_win, y + scaled_win)

    bx, by, bx2, by2 = best_coords
    # 映射回原图，保持原窗口大小
    final_x1 = int(bx / scale)
    final_y1 = int(by / scale)
    final_x2 = min(w, final_x1 + window_size)
    final_y2 = min(h, final_y1 + window_size)
    return final_x1, final_y1, final_x2, final_y2

