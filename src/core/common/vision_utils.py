"""
通用视觉工具函数：结构张量、自动标尺识别等。

使用 scikit-image 实现基于结构张量的条纹检测算法，
替代原有的基于拉普拉斯方差的 ROI 选择方法。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.feature import structure_tensor


def detect_scale_bar(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """自动检测 TEM 图像底部标尺，返回 (start_x, y, width, height)。支持黑底白字/白底黑字。
    
    Args:
        image: 输入图像（灰度或彩色）
        
    Returns:
        标尺的坐标信息 (start_x, y, width, height)，如果未检测到则返回 None
        其中 y 是相对于底部 ROI 的坐标，需要加上 (h - roi_h) 才是原图坐标
    """
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
    # 返回 (start_x, y_in_roi, width, height)，y 需要加上 (h - roi_h) 才是原图坐标
    return x, y, cw, ch


def detect_lattice_regions(
    image: np.ndarray, 
    sigma: float = 1.0, 
    threshold: float = 0.3,
    window_size: int = 512
) -> Tuple[int, int, int, int]:
    """基于结构张量（Structure Tensor）自动检测条纹最清晰的区域。
    
    使用结构张量计算局部相干性（Coherency），定位条纹最清晰的 ROI 区域。
    这是对原有基于拉普拉斯方差方法的升级。
    
    Args:
        image: 输入灰度图像
        sigma: 高斯平滑参数，控制结构张量的尺度（默认 1.0）
        threshold: 相干性阈值，用于过滤低质量区域（默认 0.3）
        window_size: 目标 ROI 窗口大小（默认 512）
        
    Returns:
        ROI 坐标 (x1, y1, x2, y2)
    """
    if image is None or image.size == 0:
        return 0, 0, 0, 0

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    
    # 过大图像先降采样，提高速度
    scale = 1.0
    work_img = gray
    if max(h, w) > 1000:
        scale = 1000.0 / max(h, w)
        work_img = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        sigma = sigma * scale  # 调整 sigma 以适应缩放
    
    scaled_h, scaled_w = work_img.shape
    
    # 计算结构张量
    # structure_tensor 返回 (Axx, Axy, Ayy) 三个分量
    Axx, Axy, Ayy = structure_tensor(work_img, sigma=sigma, mode='constant')
    
    # 手动计算特征值（结构张量的特征值公式）
    # lambda = 0.5 * (Axx + Ayy ± sqrt((Axx - Ayy)^2 + 4*Axy^2))
    trace = Axx + Ayy
    det = Axx * Ayy - Axy * Axy
    discriminant = trace * trace - 4 * det
    discriminant = np.maximum(discriminant, 0)  # 确保非负
    
    lambda1 = 0.5 * (trace + np.sqrt(discriminant))  # 较大特征值
    lambda2 = 0.5 * (trace - np.sqrt(discriminant))  # 较小特征值
    
    # 计算相干性（Coherency）
    # Coherency = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
    eps = 1e-10
    coherency = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
    
    # 应用阈值过滤
    coherency_masked = np.where(coherency > threshold, coherency, 0)
    
    # 计算目标窗口大小（缩放后）
    scaled_win = max(32, min(int(window_size * scale), scaled_h, scaled_w))
    if scaled_win <= 1:
        return 0, 0, 0, 0
    
    # 滑动窗口搜索最佳区域
    step = max(1, scaled_win // 2)
    best_score = -1.0
    best_coords = (0, 0, scaled_win, scaled_win)
    
    for y in range(0, scaled_h - scaled_win + 1, step):
        for x in range(0, scaled_w - scaled_win + 1, step):
            patch = coherency_masked[y : y + scaled_win, x : x + scaled_win]
            # 使用平均相干性作为评分
            score = np.mean(patch)
            if score > best_score:
                best_score = score
                best_coords = (x, y, x + scaled_win, y + scaled_win)
    
    bx, by, bx2, by2 = best_coords
    
    # 映射回原图坐标
    final_x1 = int(bx / scale)
    final_y1 = int(by / scale)
    final_x2 = min(w, final_x1 + window_size)
    final_y2 = min(h, final_y1 + window_size)
    
    return final_x1, final_y1, final_x2, final_y2


def auto_find_roi(
    image: np.ndarray, 
    window_size: int = 512,
    use_structure_tensor: bool = True,
    sigma: float = 1.0,
    threshold: float = 0.3
) -> Tuple[int, int, int, int]:
    """自动寻找最佳 ROI 区域。
    
    优先使用结构张量方法，如果失败则回退到拉普拉斯方差方法。
    
    Args:
        image: 输入图像
        window_size: 目标窗口大小
        use_structure_tensor: 是否优先使用结构张量方法（默认 True）
        sigma: 结构张量参数（仅当 use_structure_tensor=True 时使用）
        threshold: 结构张量阈值（仅当 use_structure_tensor=True 时使用）
        
    Returns:
        ROI 坐标 (x1, y1, x2, y2)
    """
    if image is None or image.size == 0:
        return 0, 0, 0, 0
    
    # 优先使用结构张量方法
    if use_structure_tensor:
        try:
            result = detect_lattice_regions(image, sigma=sigma, threshold=threshold, window_size=window_size)
            # 验证结果有效性
            x1, y1, x2, y2 = result
            if x2 > x1 and y2 > y1:
                return result
        except Exception:
            # 如果结构张量方法失败，回退到拉普拉斯方法
            pass
    
    # 回退到原有的拉普拉斯方差方法
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


def draw_debug_overlay(
    image: np.ndarray,
    box: Optional[Tuple[int, int, int, int]] = None,
    label: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """在图像上绘制调试覆盖层（检测框和标签）。
    
    Args:
        image: 输入图像（灰度或彩色）
        box: 检测框坐标 (x, y, width, height) 或 (x1, y1, x2, y2)，如果为 None 则不绘制
        label: 标签文字，如果为 None 则不绘制
        color: 框和文字的颜色 (B, G, R)
        thickness: 线条粗细
        
    Returns:
        绘制了覆盖层的图像副本
    """
    # 创建图像副本
    if len(image.shape) == 2:
        img_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_copy = image.copy()
    
    # 绘制检测框
    if box is not None:
        if len(box) == 4:
            # 判断是 (x, y, w, h) 还是 (x1, y1, x2, y2)
            if box[2] < 1000 and box[3] < 1000:  # 假设宽度和高度不会超过 1000
                # 可能是 (x, y, w, h) 格式
                x, y, w, h = box
                x1, y1 = x, y
                x2, y2 = x + w, y + h
            else:
                # (x1, y1, x2, y2) 格式
                x1, y1, x2, y2 = box
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签
            if label is not None:
                # 计算文字位置（框的上方）
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                
                # 文字背景
                text_x = x1
                text_y = max(y1 - 5, text_height + 5)
                cv2.rectangle(
                    img_copy,
                    (text_x, text_y - text_height - 5),
                    (text_x + text_width + 5, text_y + baseline),
                    (0, 0, 0),
                    -1
                )
                
                # 绘制文字
                cv2.putText(
                    img_copy,
                    label,
                    (text_x + 2, text_y),
                    font,
                    font_scale,
                    color,
                    text_thickness,
                    cv2.LINE_AA
                )
    
    return img_copy
