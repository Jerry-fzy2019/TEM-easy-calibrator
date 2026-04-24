"""
模块化的像素比例换算弹窗组件。

支持两种模式：
1. 直接输入：用户直接输入 nm/px 值
2. 手动校准：用户在图像上画线，输入物理长度

对于 dm3 文件，自动使用 hyperspy 读取 scale 元数据并填入输入框。
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.common.scale_calibration import calibrate_scale_from_points

# 尝试导入 hyperspy
try:
    import hyperspy.api as hs
    HYPERSPY_AVAILABLE = True
except ImportError:
    HYPERSPY_AVAILABLE = False


def extract_scale_from_dm3(file_path: str | Path) -> Optional[float]:
    """从 dm3 文件中提取 scale 信息（nm/px）。
    
    Args:
        file_path: dm3 文件路径
        
    Returns:
        如果成功提取，返回 nm/px 值；否则返回 None
    """
    if not HYPERSPY_AVAILABLE:
        return None
    
    try:
        # 使用 hyperspy 加载 dm3 文件
        signal = hs.load(str(file_path))
        
        # 如果返回列表，取第一个
        if isinstance(signal, list):
            signal = signal[0]
        
        # 获取 axes_manager 中的 scale 信息
        if hasattr(signal, 'axes_manager') and signal.axes_manager is not None:
            # 遍历所有轴，查找 scale 信息
            for axis in signal.axes_manager:
                if hasattr(axis, 'scale') and axis.scale is not None:
                    # scale 单位通常是 nm，转换为 nm/px
                    scale_nm = float(axis.scale)
                    if scale_nm > 0:
                        return scale_nm
        
        # 尝试从 metadata 中获取
        if hasattr(signal, 'metadata') and signal.metadata is not None:
            # 查找常见的 scale 字段
            metadata = signal.metadata
            if hasattr(metadata, 'get_item'):
                # 尝试不同的可能字段名
                for key in ['scale', 'Scale', 'SCALE', 'pixel_size', 'PixelSize']:
                    try:
                        value = metadata.get_item(key)
                        if value is not None:
                            scale_nm = float(value)
                            if scale_nm > 0:
                                return scale_nm
                    except (AttributeError, ValueError, TypeError):
                        continue
        
        return None
    except Exception as e:
        print(f"[DEBUG] 提取 dm3 scale 失败: {e}")
        return None


def scale_calibration_dialog(
    image: Optional[np.ndarray],
    image_file_path: Optional[str | Path] = None,
    current_nm_per_pixel: Optional[float] = None,
    session_state_key: str = "scale_calibration",
    on_calibrated: Optional[Callable[[float], None]] = None,
    input_label: str = "像素分辨率 (nm/px)",
    input_help: str = "每像素对应的纳米数",
    input_format: str = "%.6f",
) -> Tuple[Optional[float], Dict[str, Any]]:
    """像素比例换算弹窗组件。
    
    Args:
        image: 当前显示的图像（用于手动校准画线）
        image_file_path: 图像文件路径（用于提取 dm3 元数据）
        current_nm_per_pixel: 当前的 nm/px 值（用于初始化输入框）
        session_state_key: session state 的 key，用于存储状态
        on_calibrated: 校准成功后的回调函数，参数为 nm_per_pixel
        
    Returns:
        (nm_per_pixel, info_dict)
        - nm_per_pixel: 校准后的 nm/px 值，如果未校准则返回 None
        - info_dict: 包含状态信息的字典，如 {'mode': 'input'|'manual', 'auto_filled': bool}
    """
    info_dict = {'mode': None, 'auto_filled': False}
    
    # 初始化 session state
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = {
            'mode': 'input',
            'nm_per_pixel': current_nm_per_pixel if current_nm_per_pixel else 0.1,
            'manual_scale_length': 50.0,
            'manual_points': None,
        }
    
    state = st.session_state[session_state_key]
    
    # 检查是否是 dm3 文件，如果是则自动提取 scale
    auto_filled_value = None
    if image_file_path:
        file_path = Path(image_file_path)
        if file_path.suffix.lower() == '.dm3':
            auto_filled_value = extract_scale_from_dm3(file_path)
            if auto_filled_value is not None:
                info_dict['auto_filled'] = True
                # 自动填入到输入框
                if state['mode'] == 'input':
                    state['nm_per_pixel'] = auto_filled_value
    
    # 使用弹窗（modal）
    with st.expander("📏 像素比例换算", expanded=False):
        # 模式选择
        mode = st.radio(
            "校准模式",
            ["直接输入", "手动校准"],
            index=0 if state['mode'] == 'input' else 1,
            key=f"{session_state_key}_mode",
            help="选择像素比例设置方式：直接输入或手动画线换算"
        )
        
        state['mode'] = mode
        info_dict['mode'] = mode
        
        if mode == "直接输入":
            # 直接输入模式
            if auto_filled_value is not None:
                st.success(f"✅ 已从 dm3 文件自动提取像素比例: {auto_filled_value:.6f} nm/px")
            
            nm_per_pixel = st.number_input(
                input_label,
                min_value=0.000001,
                value=state['nm_per_pixel'],
                step=0.0001,
                format=input_format,
                key=f"{session_state_key}_input",
                help=input_help
            )
            
            state['nm_per_pixel'] = nm_per_pixel
            
            if st.button("✅ 确认", type="primary", key=f"{session_state_key}_confirm_input"):
                if nm_per_pixel > 0:
                    if on_calibrated:
                        on_calibrated(nm_per_pixel)
                    st.success(f"✅ 像素比例已设置: {nm_per_pixel:.6f} nm/px")
                    st.rerun()
                else:
                    st.error("❌ 请输入有效的像素比例值")
            
            return nm_per_pixel if nm_per_pixel > 0 else None, info_dict
        
        else:
            # 手动校准模式
            if image is None:
                st.warning("⚠️ 请先上传图像")
                return None, info_dict
            
            st.info("👆 在图像上画一条线覆盖标尺，然后输入物理长度（按原图像素换算，与显示缩放无关）")
            
            # 统一将图像转换为 8 位、对比度适中的显示图，避免某些 HRTEM 原图过暗或 16bit 看起来是“空白”
            img_np = np.asarray(image)
            h, w = img_np.shape[:2] if len(img_np.shape) == 2 else img_np.shape[:2]

            # 如果不是 uint8，则做一次线性拉伸到 [0, 255]
            if img_np.dtype != np.uint8:
                img_float = img_np.astype(np.float32)
                v_min = float(img_float.min())
                v_max = float(img_float.max())
                if v_max > v_min:
                    img_norm = (img_float - v_min) / (v_max - v_min)
                    img_8u = (img_norm * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_8u = np.zeros_like(img_float, dtype=np.uint8)
            else:
                img_8u = img_np

            # 计算显示尺寸并生成与 canvas 同尺寸的缩放图，避免前端缩放导致比例尺错误
            max_display_size = 400
            scale_display = min(1.0, max_display_size / max(w, h))
            display_width = int(w * scale_display)
            display_height = int(h * scale_display)

            # 始终转换为 RGB 或灰度，再给 canvas，当 PNG 带 alpha 通道时强制丢弃 alpha，避免前端兼容性问题
            if len(img_8u.shape) == 2:
                pil_full = Image.fromarray(img_8u, mode='L').convert('RGB')
            else:
                # 只取前三个通道作为 RGB
                if img_8u.shape[2] >= 3:
                    pil_full = Image.fromarray(img_8u[:, :, :3], mode='RGB')
                else:
                    pil_full = Image.fromarray(img_8u[:, :, 0], mode='L').convert('RGB')
            pil_image = pil_full.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # 创建 canvas（背景图为缩放图，坐标换算时按 原图尺寸/显示尺寸 得到原图像素）
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=pil_image,
                update_streamlit=True,
                drawing_mode="line",
                point_display_radius=0,
                key=f"{session_state_key}_canvas",
                width=display_width,
                height=display_height,
            )
            
            # 处理画线结果：canvas 坐标为显示图坐标，必须按比例换算为原图像素
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                if objects:
                    last_obj = objects[-1]
                    if last_obj.get("type") == "line":
                        canvas_x1 = float(last_obj.get("x1", 0))
                        canvas_y1 = float(last_obj.get("y1", 0))
                        canvas_x2 = float(last_obj.get("x2", 0))
                        canvas_y2 = float(last_obj.get("y2", 0))
                        scale_x = w / display_width
                        scale_y = h / display_height
                        x1_orig = canvas_x1 * scale_x
                        y1_orig = canvas_y1 * scale_y
                        x2_orig = canvas_x2 * scale_x
                        y2_orig = canvas_y2 * scale_y
                        # 水平方向像素距离（原图像素）
                        pixel_distance = abs(x2_orig - x1_orig)
                        point1 = (int(round(x1_orig)), int(round(y1_orig)))
                        point2 = (int(round(x2_orig)), int(round(y2_orig)))
                        
                        st.info(f"已画线（按原图换算）：像素距离 **{pixel_distance:.2f} px**（原图），标尺长度请填纳米数")
                        
                        # 输入物理长度
                        physical_length = st.number_input(
                            "标尺物理长度 (nm)",
                            min_value=0.1,
                            value=state['manual_scale_length'],
                            step=0.1,
                            key=f"{session_state_key}_physical_length",
                        )
                        state['manual_scale_length'] = physical_length
                        
                        if st.button("✅ 开始换算", type="primary", key=f"{session_state_key}_confirm_manual"):
                            if pixel_distance > 1e-6:
                                result = calibrate_scale_from_points(
                                    point1, point2, physical_length, force_horizontal=True,
                                    pixel_distance_override=pixel_distance,
                                )
                                
                                if result.get('status') == 'success':
                                    nm_per_pixel = result.get('nm_per_pixel')
                                    state['nm_per_pixel'] = nm_per_pixel
                                    state['manual_points'] = (point1, point2)
                                    
                                    if on_calibrated:
                                        on_calibrated(nm_per_pixel)
                                    
                                    st.success(f"✅ 校准成功！分辨率: {nm_per_pixel:.6f} nm/px")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result.get('error_message', '校准失败')}")
                            else:
                                st.warning("⚠️ 请先画一条有效的线")
            
            # 返回当前存储的值（如果有）
            return state.get('nm_per_pixel') if state.get('nm_per_pixel', 0) > 0 else None, info_dict
