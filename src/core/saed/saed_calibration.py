"""
SAED 选区衍射校准核心模块。

实现衍射斑点/环的检测与索引、像平面到倒易空间的几何映射。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import sys
import string

import cv2
import numpy as np
from skimage.feature import blob_log

# 添加项目根目录到路径（用于导入其他模块）
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.thickness.image_io import load_image


class SAEDAnalyzer:
    """SAED 选区衍射分析器。"""
    
    def __init__(self):
        """初始化分析器。"""
        pass
    
    def detect_beam_center(self, image: np.ndarray) -> Tuple[float, float]:
        """寻找透射斑中心（选择图像中心最近的亮斑）。
        
        Args:
            image: 输入 SAED 图像（灰度图）
            
        Returns:
            透射斑中心坐标 (center_x, center_y)
        """
        h, w = image.shape
        image_center = (w / 2, h / 2)
        
        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 找到所有亮斑（使用阈值）
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        threshold = max_val * 0.7  # 降低阈值以找到更多候选亮斑
        
        # 创建二值化图像
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # 查找所有连通域（亮斑）
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果没有找到亮斑，回退到最大值位置
            return (float(max_loc[0]), float(max_loc[1]))
        
        # 计算每个连通域的质心，并找到距离图像中心最近的
        best_center = None
        min_distance = float('inf')
        
        for contour in contours:
            # 计算质心
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                
                # 计算到图像中心的距离
                distance = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_center = (cx, cy)
        
        if best_center is None:
            # 如果所有连通域都没有有效质心，回退到最大值位置
            return (float(max_loc[0]), float(max_loc[1]))
        
        return (float(best_center[0]), float(best_center[1]))
    
    def detect_diffraction_spots(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
        mask_radius: int = 20,
        min_sigma: float = 2.0,
        max_sigma: float = 20.0,
        threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """检测衍射斑点（使用 Laplacian of Gaussian）。
        
        Args:
            image: 输入 SAED 图像（灰度图）
            center: 透射斑中心坐标 (center_x, center_y)
            mask_radius: 中心屏蔽半径（像素），用于排除透射斑
            min_sigma: LoG 检测的最小 sigma 值（默认 2.0）
            max_sigma: LoG 检测的最大 sigma 值（默认 20.0）
            threshold: LoG 检测的阈值（默认 0.1）
            
        Returns:
            衍射斑点列表，每项包含：
            - 'x': 斑点 x 坐标
            - 'y': 斑点 y 坐标
            - 'r_pix': 距离中心的像素距离
            - 'intensity': 斑点强度
            列表按强度降序排列
        """
        center_x, center_y = center
        
        # 使用 blob_log 检测斑点
        # blob_log 返回 (y, x, sigma) 格式
        blobs = blob_log(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
            num_sigma=10,  # sigma 采样数量
        )
        
        spots = []
        h, w = image.shape
        
        for blob in blobs:
            # blob_log 返回 (y, x, sigma)
            y, x, sigma = blob
            
            # 检查是否在图像范围内
            if not (0 <= x < w and 0 <= y < h):
                continue
            
            # 计算距离中心的像素距离
            dx = x - center_x
            dy = y - center_y
            r_pix = np.sqrt(dx**2 + dy**2)
            
            # 中心屏蔽：如果距离小于 mask_radius，则排除（位于透射斑内部）
            if r_pix < mask_radius:
                continue
            
            # 获取该位置的强度值
            x_int = int(round(x))
            y_int = int(round(y))
            if 0 <= x_int < w and 0 <= y_int < h:
                intensity = float(image[y_int, x_int])
            else:
                intensity = 0.0
            
            spots.append({
                'x': float(x),
                'y': float(y),
                'r_pix': float(r_pix),
                'intensity': intensity,
            })
        
        # 按强度降序排列
        spots.sort(key=lambda s: s['intensity'], reverse=True)
        
        return spots
    
    def _detect_nearest_ring_spots(self, spots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测中心最近邻的一圈斑点。
        
        先计算所有斑点的距离，按距离排序。
        当出现一个距离是已有最小距离的两倍或更大时，判断为下一圈。
        最近一圈就是该点之前的所有斑点。
        
        Args:
            spots: 所有检测到的斑点列表
            
        Returns:
            最近邻一圈斑点的列表
        """
        if not spots:
            return []
        
        # 按距离中心排序
        sorted_by_distance = sorted(spots, key=lambda s: s.get('r_pix', 0))
        
        if len(sorted_by_distance) == 0:
            return []
        
        # 如果只有一个斑点，直接返回
        if len(sorted_by_distance) == 1:
            return sorted_by_distance
        
        # 找到最小距离（第一个斑点的距离）
        min_distance = sorted_by_distance[0].get('r_pix', 0)
        
        # 如果最小距离为0或无效，至少返回前2个斑点
        if min_distance <= 0 or min_distance < 1.0:
            return sorted_by_distance[:min(2, len(sorted_by_distance))]
        
        # 遍历所有斑点，找到第一个距离 >= 最小距离两倍的斑点
        # 这个点之前的所有斑点就是最近一圈
        nearest_ring_end = len(sorted_by_distance)  # 默认返回所有（如果所有斑点都在同一圈）
        
        for i in range(1, len(sorted_by_distance)):
            current_distance = sorted_by_distance[i].get('r_pix', 0)
            
            # 跳过无效的距离值
            if current_distance <= 0:
                continue
            
            # 如果当前距离 >= 最小距离的两倍，说明进入下一圈
            if current_distance >= min_distance * 2.0:
                nearest_ring_end = i
                break
        
        # 返回最近一圈的斑点
        nearest_ring = sorted_by_distance[:nearest_ring_end]
        
        # 确保至少返回1个斑点（正常情况下不应该为空，但作为保护措施）
        if len(nearest_ring) == 0:
            nearest_ring = sorted_by_distance[:1]
        
        # 额外保护：如果返回的列表为空（不应该发生），至少返回前几个斑点
        if len(nearest_ring) == 0 and len(sorted_by_distance) > 0:
            nearest_ring = sorted_by_distance[:min(3, len(sorted_by_distance))]
        
        return nearest_ring
    
    def analyze_single_image(
        self,
        image_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """分析单张 SAED 衍射图。
        
        Args:
            image_path: 图像文件路径
            config: 配置字典，可包含：
                - 'calibration_factor': 标定系数 k (nm⁻¹/pixel)，如果未提供则使用默认值 1.0
                - 'scale_bar_val': 标尺值（可选，用于计算 calibration_factor）
                - 'mask_radius': 中心屏蔽半径（像素，默认 20）
                - 'min_sigma': LoG 检测的最小 sigma（默认 2.0）
                - 'max_sigma': LoG 检测的最大 sigma（默认 20.0）
                - 'threshold': LoG 检测的阈值（默认 0.1）
            
        Returns:
            结果字典，包含：
            - 'status': 'success' 或 'error'
            - 'center': 透射斑中心坐标 (center_x, center_y)
            - 'calibration_factor': 标定因子 k (nm⁻¹/pixel)
            - 'is_calibrated': 是否已标定（bool）
            - 'spots': 衍射斑点列表，每项包含：
                - 'x': 斑点 x 坐标
                - 'y': 斑点 y 坐标
                - 'r_pix': 距离中心的像素距离
                - 'd_spacing_nm': 晶面间距（nm），如果未标定则为 0
                - 'angle_deg': 相对于中心的角度（度，0-360）
                - 'intensity': 斑点强度
            - 'top_spots': 前 N 个最强斑点的列表
            - 'error_message': 错误信息（如果失败）
            - 'processing_time': 处理时间（秒）
        """
        start_time = time.time()
        
        if config is None:
            config = {}
        
        try:
            # 1. 加载图像
            image = load_image(image_path)
            if image is None:
                return {
                    'status': 'error',
                    'error_message': '图像加载失败，请检查文件格式',
                    'processing_time': time.time() - start_time,
                }
            
            # 确保是灰度图
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 为避免将图像下方的比例尺条纹/标注区域误算入斑点检测，
            # 默认裁掉图像底部 1/10 的高度，仅使用上方 90% 进行中心与斑点分析。
            try:
                h, w = image.shape[:2]
                if h > 10:
                    crop_h = int(h * 0.9)
                    if crop_h < h:
                        image = image[:crop_h, :]
            except Exception:
                # 出现任何异常时，回退为使用完整图像
                pass
            
            # 2. 中心定位
            manual_center = config.get('manual_center')
            if manual_center is not None and len(manual_center) == 2:
                # 使用手动指定的中心
                center_x, center_y = float(manual_center[0]), float(manual_center[1])
            else:
                # 自动检测中心
                center = self.detect_beam_center(image)
                center_x, center_y = center
            
            # 3. 斑点检测
            mask_radius = config.get('mask_radius', 20)
            min_sigma = config.get('min_sigma', 2.0)
            max_sigma = config.get('max_sigma', 20.0)
            threshold = config.get('threshold', 0.1)
            
            spots_raw = self.detect_diffraction_spots(
                image,
                center,
                mask_radius=mask_radius,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                threshold=threshold,
            )
            
            if not spots_raw:
                return {
                    'status': 'error',
                        'error_message': '未检测到任何衍射斑点',
                    'processing_time': time.time() - start_time,
                    }
            
            # 4. 获取标定系数
            calibration_factor = config.get('calibration_factor')
            is_calibrated = True
            
            if calibration_factor is None or calibration_factor <= 0:
                # 如果未提供标定系数，使用默认值 1.0 并标记为未标定
                calibration_factor = 1.0
                is_calibrated = False
            
            # 5. 计算物理量（d-spacing 和角度）
            spots = []
            for spot in spots_raw:
                x = spot['x']
                y = spot['y']
                r_pix = spot['r_pix']
                intensity = spot['intensity']
                
                # 计算 d-spacing
                d_spacing_nm = 0.0
                if is_calibrated and r_pix > 1e-5:
                    try:
                        # d = 1.0 / (r_pix * k)
                        reciprocal_length = r_pix * calibration_factor
                        if reciprocal_length > 1e-10:
                            d_spacing_nm = 1.0 / reciprocal_length
                    except ZeroDivisionError:
                        d_spacing_nm = 0.0
                
                # 计算角度（相对于中心，0-360 度）
                dx = x - center_x
                dy = y - center_y
                
                # 使用 atan2 计算角度（弧度），然后转换为度
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)
                
                # 标准化到 0-360 度范围
                if angle_deg < 0:
                    angle_deg += 360.0
                
                spots.append({
                    'x': round(x, 2),
                    'y': round(y, 2),
                    'r_pix': round(r_pix, 2),
                    'd_spacing_nm': round(d_spacing_nm, 4) if d_spacing_nm > 0 else 0.0,
                    'angle_deg': round(angle_deg, 2),
                    'intensity': round(intensity, 2),
                })
            
            # 6. 检测中心最近邻的一圈斑点
            if not spots:
                nearest_ring_spots = []
            else:
                nearest_ring_spots = self._detect_nearest_ring_spots(spots)
                # 确保返回的列表不为空（如果spots不为空）
                if not nearest_ring_spots and len(spots) > 0:
                    # 如果检测失败，至少返回前几个最近的斑点
                    sorted_spots = sorted(spots, key=lambda s: s.get('r_pix', 0))
                    nearest_ring_spots = sorted_spots[:min(3, len(sorted_spots))]
            
            # 7. 按距离对最近邻一圈斑点分组（a,b,c...），像素差小于阈值视为同一长度
            distance_groups: List[Dict[str, Any]] = []
            if len(nearest_ring_spots) > 0:
                # 阈值（像素），可通过 config['distance_group_threshold_pix'] 调整
                threshold_pix = float(config.get('distance_group_threshold_pix', 5.0))
                
                # 按距离从小到大排序
                sorted_by_r = sorted(nearest_ring_spots, key=lambda s: s.get('r_pix', 0.0))
                
                letters = string.ascii_lowercase
                current_group_index = -1
                current_label = None
                
                for spot in sorted_by_r:
                    r = float(spot.get('r_pix', 0.0))
                    if r <= 0:
                        continue
                    
                    if not distance_groups:
                        # 第一个分组
                        current_group_index = 0
                        current_label = letters[current_group_index]
                        distance_groups.append({
                            'label': current_label,
                            'spots': [spot],
                            'mean_r_pix': r,
                        })
                        spot['group_label'] = current_label
                        spot['group_member_index'] = 1
                    else:
                        current_group = distance_groups[-1]
                        mean_r = float(current_group['mean_r_pix'])
                        
                        if abs(r - mean_r) <= threshold_pix:
                            # 归入当前分组
                            current_group['spots'].append(spot)
                            # 更新当前分组平均半径
                            rs = [float(s.get('r_pix', 0.0)) for s in current_group['spots'] if float(s.get('r_pix', 0.0)) > 0]
                            current_group['mean_r_pix'] = float(np.mean(rs)) if rs else mean_r
                            spot['group_label'] = current_group['label']
                            spot['group_member_index'] = len(current_group['spots'])
                        else:
                            # 新建一个分组，使用下一个字母
                            current_group_index += 1
                            if current_group_index >= len(letters):
                                # 超过 z 后继续使用 z
                                current_group_index = len(letters) - 1
                            current_label = letters[current_group_index]
                            distance_groups.append({
                                'label': current_label,
                                'spots': [spot],
                                'mean_r_pix': r,
                            })
                            spot['group_label'] = current_label
                            spot['group_member_index'] = 1
                
                # 计算每组的平均 d-spacing 和平均角度
                for group in distance_groups:
                    group_spots = group['spots']
                    rs = [float(s.get('r_pix', 0.0)) for s in group_spots if float(s.get('r_pix', 0.0)) > 0]
                    ds = [float(s.get('d_spacing_nm', 0.0)) for s in group_spots if float(s.get('d_spacing_nm', 0.0)) > 0]
                    angles = [float(s.get('angle_deg', 0.0)) for s in group_spots]
                    
                    group['mean_r_pix'] = float(np.mean(rs)) if rs else 0.0
                    group['mean_d_spacing_nm'] = float(np.mean(ds)) if ds else 0.0
                    group['mean_angle_deg'] = float(np.mean(angles)) if angles else 0.0
            
            # 8. 计算最近邻一圈斑点的角度差和距离比值（按 a,b,c... 分组）
            angle_differences: List[float] = []
            distance_ratios: List[float] = []
            
            # 角度差：使用分组后的平均角度，按角度排序计算相邻组之间的角度差
            if len(distance_groups) >= 2:
                sorted_groups = sorted(
                    [g for g in distance_groups if g.get('mean_angle_deg', None) is not None],
                    key=lambda g: g.get('mean_angle_deg', 0.0)
                )
                
                for i in range(len(sorted_groups)):
                    current_angle = float(sorted_groups[i].get('mean_angle_deg', 0.0))
                    next_angle = float(sorted_groups[(i + 1) % len(sorted_groups)].get('mean_angle_deg', 0.0))
                    
                    angle_diff = next_angle - current_angle
                    if angle_diff < 0:
                        angle_diff += 360.0
                    
                    angle_differences.append(round(angle_diff, 2))
            
            # 距离比值：只在不同长度分组（a,b,c...）之间计算“较小 / 较大”的比值
            if len(distance_groups) >= 2:
                for i in range(len(distance_groups)):
                    for j in range(i + 1, len(distance_groups)):
                        r1 = float(distance_groups[i].get('mean_r_pix', 0.0))
                        r2 = float(distance_groups[j].get('mean_r_pix', 0.0))
                        if r1 > 1e-5 and r2 > 1e-5:
                            r_min = min(r1, r2)
                            r_max = max(r1, r2)
                            ratio = r_min / r_max
                            distance_ratios.append(round(ratio, 4))
            
            return {
                'status': 'success',
                'center': (round(center_x, 2), round(center_y, 2)),
                'calibration_factor': round(calibration_factor, 6),
                'is_calibrated': is_calibrated,
                'spots': spots,
                'nearest_ring_spots': nearest_ring_spots,  # 最近邻一圈斑点
                'distance_groups': distance_groups,        # 距离分组（a,b,c...）
                'angle_differences': angle_differences,  # 角度差列表
                'distance_ratios': distance_ratios,  # 距离比值列表
                'processing_time': round(time.time() - start_time, 2),
            }
        
        except Exception as e:
            import traceback
            return {
                'status': 'error',
                'error_message': f'分析过程出错: {str(e)}\n{traceback.format_exc()}',
                'processing_time': time.time() - start_time,
            }


# 保留旧接口以兼容
def calibrate_saed_pattern(image: np.ndarray) -> Dict[str, Any]:
    """占位函数：对 SAED 选区衍射图进行标定（已废弃，请使用 SAEDAnalyzer）。"""
    raise NotImplementedError("SAED 选区衍射校准算法尚未实现，请使用 SAEDAnalyzer.analyze_single_image")
