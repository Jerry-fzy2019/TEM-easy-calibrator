"""
Streamlit UI 工具函数。
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw
import streamlit as st


def draw_roi_box(
    image: np.ndarray,
    roi_coords: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """在图像上绘制 ROI 框。
    
    Args:
        image: 输入图像
        roi_coords: ROI 坐标 (x1, y1, x2, y2)
        color: 框的颜色 (B, G, R)
        thickness: 线条粗细
        
    Returns:
        绘制了 ROI 框的图像
    """
    img_copy = image.copy()
    if len(img_copy.shape) == 2:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
    elif img_copy.shape[2] == 4:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGBA2RGB)
    
    x1, y1, x2, y2 = roi_coords
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    return img_copy


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """将 numpy 数组转换为 PIL Image。
    
    Args:
        image: numpy 数组图像
        
    Returns:
        PIL Image 对象
    """
    if len(image.shape) == 2:
        return Image.fromarray(image, mode='L')
    elif image.shape[2] == 3:
        return Image.fromarray(image, mode='RGB')
    elif image.shape[2] == 4:
        return Image.fromarray(image, mode='RGBA')
    else:
        raise ValueError(f"不支持的图像形状: {image.shape}")


def format_result_summary(result: dict) -> str:
    """格式化分析结果为可读的文本摘要。
    
    Args:
        result: analyze_single_image 返回的结果字典
        
    Returns:
        格式化的文本字符串
    """
    if result.get('status') != 'success':
        return f"❌ 分析失败: {result.get('error_message', '未知错误')}"
    
    summary = f"""
## ✅ 分析成功

### 主要结果
- **周期性层间距**: {result.get('periodicity_nm', 'N/A')} nm
- **比例尺因子**: {result.get('scale_factor', 'N/A')} nm/px

### ROI 信息
- **坐标**: {result.get('roi_coords', 'N/A')}

### FFT 统计
"""
    fft_stats = result.get('fft_stats', {})
    if fft_stats:
        peak_pos = fft_stats.get('peak_position', {})
        summary += f"- **峰值位置**: ({peak_pos.get('x', 'N/A')}, {peak_pos.get('y', 'N/A')})\n"
        summary += f"- **峰值距离**: {fft_stats.get('peak_distance_pixels', 'N/A')} px\n"
    
    summary += f"\n### 处理信息\n"
    summary += f"- **处理时间**: {result.get('processing_time', 'N/A')} 秒\n"
    
    preprocess = result.get('preprocess_params', {})
    if preprocess:
        summary += f"- **预处理参数**: 降噪核={preprocess.get('noise_kernel', 'N/A')}, "
        summary += f"对比度={preprocess.get('contrast_alpha', 'N/A')}, "
        summary += f"亮度={preprocess.get('brightness_beta', 'N/A')}\n"
    
    return summary


def create_fft_heatmap_plotly(
    fft_magnitude: np.ndarray,
    peak_position: Optional[Tuple[float, float]] = None,
    center: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """使用 Plotly 创建交互式 FFT 热力图。
    
    Args:
        fft_magnitude: 2D FFT 幅度谱数组
        peak_position: 峰值位置 (x, y)，可选
        center: FFT 中心位置 (x, y)，可选
        
    Returns:
        Plotly Figure 对象
    """
    h, w = fft_magnitude.shape
    if center is None:
        center = (w // 2, h // 2)
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=fft_magnitude,
        colorscale='Hot',
        colorbar=dict(title="Magnitude (dB)"),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Magnitude: %{z:.2f} dB<extra></extra>',
    ))
    
    # 标记峰值位置
    if peak_position is not None:
        peak_x, peak_y = peak_position
        fig.add_trace(go.Scatter(
            x=[peak_x],
            y=[peak_y],
            mode='markers',
            marker=dict(
                size=15,
                color='cyan',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            name='Peak',
            hovertemplate=f'Peak Position<br>X: {peak_x:.2f}<br>Y: {peak_y:.2f}<extra></extra>',
        ))
    
    # 标记中心位置
    center_x, center_y = center
    fig.add_trace(go.Scatter(
        x=[center_x],
        y=[center_y],
        mode='markers',
        marker=dict(
            size=10,
            color='yellow',
            symbol='circle',
            line=dict(width=2, color='black')
        ),
        name='Center',
        hovertemplate=f'Center<br>X: {center_x}<br>Y: {center_y}<extra></extra>',
    ))
    
    # 更新布局（支持 Dark Mode）
    fig.update_layout(
        title="FFT Magnitude Spectrum",
        xaxis_title="X (pixels)",
        yaxis_title="Y (pixels)",
        width=600,
        height=500,
        template='plotly',  # 使用默认模板，自动适配 Dark Mode
        plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            font=dict(size=11),
        ),
        # 不设置 font.color，让 Plotly 使用模板默认值（自动适配 Dark Mode）
    )
    
    # 反转 Y 轴以匹配图像坐标系
    fig.update_yaxes(autorange="reversed")
    
    return fig


def create_fft_profile_plotly(
    profile_data: Dict[str, list],
    peak_distance: Optional[float] = None,
) -> go.Figure:
    """创建径向 Profile 曲线。
    
    Args:
        profile_data: 包含 'distance' 和 'intensity' 列表的字典
        peak_distance: 峰值距离（像素），可选
        
    Returns:
        Plotly Figure 对象
    """
    if profile_data is None or not profile_data.get('distance'):
        # 返回空图表（支持 Dark Mode）
        fig = go.Figure()
        fig.update_layout(
            title="FFT Radial Profile",
            xaxis_title="Distance from Center (pixels)",
            yaxis_title="Intensity (dB)",
            width=600,
            height=400,
            template='plotly',  # 使用默认模板，自动适配 Dark Mode
            plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
            paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
            # 不设置 font.color，让 Plotly 使用模板默认值（自动适配 Dark Mode）
        )
        return fig
    
    distances = profile_data['distance']
    intensities = profile_data['intensity']
    
    # 创建曲线
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=distances,
        y=intensities,
        mode='lines+markers',
        name='Radial Profile',
        line=dict(width=2, color='blue'),
        marker=dict(size=4),
        hovertemplate='Distance: %{x:.1f} px<br>Intensity: %{y:.2f} dB<extra></extra>',
    ))
    
    # 标记峰值位置
    if peak_distance is not None:
        # 找到最接近峰值距离的数据点
        closest_idx = min(range(len(distances)), key=lambda i: abs(distances[i] - peak_distance))
        peak_intensity = intensities[closest_idx]
        
        fig.add_trace(go.Scatter(
            x=[peak_distance],
            y=[peak_intensity],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            name='Peak',
            hovertemplate=f'Peak Distance<br>Distance: {peak_distance:.2f} px<br>Intensity: {peak_intensity:.2f} dB<extra></extra>',
        ))
    
    # 更新布局（支持 Dark Mode）
    fig.update_layout(
        title="FFT Radial Profile",
        xaxis_title="Distance from Center (pixels)",
        yaxis_title="Intensity (dB)",
        width=600,
        height=400,
        template='plotly',  # 使用默认模板，自动适配 Dark Mode
        plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
        # 不设置 font.color，让 Plotly 使用模板默认值（自动适配 Dark Mode）
        hovermode='closest',
    )
    
    return fig


def create_metric_cards(result: Dict[str, Any]) -> None:
    """使用 st.metric 显示关键指标。
    
    Args:
        result: analyze_single_image 返回的结果字典
    """
    if result.get('status') != 'success':
        return
    
    # 检查是否有单层厚度数据
    layer_thickness = result.get('layer_thickness_nm')
    
    if layer_thickness is not None:
        # 5 列布局（包含单层厚度）
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            periodicity = result.get('periodicity_nm', 0)
            st.metric(
                label="周期性层间距",
                value=f"{periodicity:.2f} nm",
                help="通过 FFT 分析得到的周期性层间距（黑+白总宽度）"
            )
        
        with col2:
            st.metric(
                label="单层黑色厚度",
                value=f"{layer_thickness:.2f} nm",
                help="通过灰度剖面分析得到的单层黑色条纹厚度"
            )
        
        with col3:
            scale_factor = result.get('scale_factor', 0)
            st.metric(
                label="比例尺因子",
                value=f"{scale_factor:.4f} nm/px",
                help="每像素对应的纳米数"
            )
        
        with col4:
            processing_time = result.get('processing_time', 0)
            st.metric(
                label="处理时间",
                value=f"{processing_time:.2f} s",
                help="图像分析总耗时"
            )
        
        with col5:
            fft_stats = result.get('fft_stats', {})
            peak_dist = fft_stats.get('peak_distance_pixels', 0)
            st.metric(
                label="峰值距离",
                value=f"{peak_dist:.2f} px",
                help="FFT 峰值距离中心的像素距离"
            )
    else:
        # 4 列布局（原有布局）
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            periodicity = result.get('periodicity_nm', 0)
            st.metric(
                label="周期性层间距",
                value=f"{periodicity:.2f} nm",
                help="通过 FFT 分析得到的周期性层间距（黑+白总宽度）"
            )
        
        with col2:
            scale_factor = result.get('scale_factor', 0)
            st.metric(
                label="比例尺因子",
                value=f"{scale_factor:.4f} nm/px",
                help="每像素对应的纳米数"
            )
        
        with col3:
            processing_time = result.get('processing_time', 0)
            st.metric(
                label="处理时间",
                value=f"{processing_time:.2f} s",
                help="图像分析总耗时"
            )
        
        with col4:
            fft_stats = result.get('fft_stats', {})
            peak_dist = fft_stats.get('peak_distance_pixels', 0)
            st.metric(
                label="峰值距离",
                value=f"{peak_dist:.2f} px",
                help="FFT 峰值距离中心的像素距离"
            )


def create_profile_plotly(
    profile_data: Dict[str, Any],
    peaks: Optional[list] = None,
    widths_nm: Optional[list] = None,
    mean_width_nm: Optional[float] = None,
) -> go.Figure:
    """创建灰度剖面曲线图，用于验证单层厚度测量。
    
    Args:
        profile_data: 包含 'profile' 和 'profile_smooth' 的字典
        peaks: 检测到的峰值位置索引列表
        widths_nm: 每个峰对应的宽度（纳米）
        mean_width_nm: 平均宽度（纳米）
        
    Returns:
        Plotly Figure 对象
    """
    if profile_data is None or not profile_data.get('profile'):
        # 返回空图表
        fig = go.Figure()
        fig.update_layout(
            title="灰度剖面曲线",
            xaxis_title="位置 (pixels)",
            yaxis_title="灰度值",
            width=600,
            height=400,
            template='plotly',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig
    
    # 从 profile_data 中提取数据
    profile = profile_data.get('profile', [])
    profile_smooth = profile_data.get('profile_smooth', profile)
    
    # 确保是列表类型（如果是 numpy 数组，转换为列表）
    if isinstance(profile, np.ndarray):
        profile = profile.tolist()
    if isinstance(profile_smooth, np.ndarray):
        profile_smooth = profile_smooth.tolist()
    
    # 如果没有提供 peaks 和 widths_nm，从 profile_data 中获取
    if peaks is None:
        peaks = profile_data.get('peaks', [])
    if widths_nm is None:
        widths_nm = profile_data.get('widths_nm', [])
    if mean_width_nm is None:
        mean_width_nm = profile_data.get('mean_width_nm', None)
    
    # 确保 peaks 和 widths_nm 是列表
    if isinstance(peaks, np.ndarray):
        peaks = peaks.tolist()
    if isinstance(widths_nm, np.ndarray):
        widths_nm = widths_nm.tolist()
    
    # 检查 profile 是否为空（使用 len 而不是直接布尔判断，避免 numpy 数组问题）
    if len(profile) == 0:
        # 返回空图表
        fig = go.Figure()
        fig.update_layout(
            title="灰度剖面曲线",
            xaxis_title="位置 (pixels)",
            yaxis_title="灰度值",
            width=600,
            height=400,
            template='plotly',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig
    
    x_axis = list(range(len(profile)))
    
    # 创建图表
    fig = go.Figure()
    
    # 绘制原始波形
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=profile,
        mode='lines',
        name='原始波形',
        line=dict(width=1, color='lightgray'),
        opacity=0.5,
        hovertemplate='位置: %{x} px<br>灰度: %{y:.1f}<extra></extra>',
    ))
    
    # 绘制平滑后的波形
    if len(profile_smooth) == len(profile):
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=profile_smooth,
            mode='lines',
            name='平滑波形',
            line=dict(width=2, color='blue'),
            hovertemplate='位置: %{x} px<br>灰度: %{y:.1f}<extra></extra>',
        ))
    
    # 标记检测到的峰值（对应黑色条纹）
    # 使用 len() 检查而不是直接布尔判断，避免 numpy 数组问题
    if len(peaks) > 0 and len(profile_smooth) > 0:
        peak_values = [profile_smooth[p] if p < len(profile_smooth) else 0 for p in peaks]
        fig.add_trace(go.Scatter(
            x=peaks,
            y=peak_values,
            mode='markers',
            name='检测到的黑色条纹',
            marker=dict(
                size=10,
                color='red',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            hovertemplate='位置: %{x} px<br>灰度: %{y:.1f}<extra></extra>',
        ))
        
        # 在峰值处绘制宽度标记（水平线段）
        # 使用 len() 检查而不是直接布尔判断，避免 numpy 数组问题
        if len(widths_nm) > 0 and len(widths_nm) == len(peaks) and len(profile_smooth) > 0:
            # 从 profile_data 获取 nm_per_pixel（如果可用）
            nm_per_pixel = profile_data.get('nm_per_pixel', None)
            if nm_per_pixel is None and mean_width_nm is not None and mean_width_nm > 0:
                # 如果没有提供，尝试从平均宽度估算
                widths_pixels = profile_data.get('widths_pixels', [])
                # 确保是列表类型
                if isinstance(widths_pixels, np.ndarray):
                    widths_pixels = widths_pixels.tolist()
                if len(widths_pixels) > 0:
                    avg_width_pixels = np.mean(widths_pixels)
                    nm_per_pixel = mean_width_nm / avg_width_pixels if avg_width_pixels > 0 else 1
                else:
                    nm_per_pixel = 1
            elif nm_per_pixel is None:
                nm_per_pixel = 1
            
            for peak_idx, width_nm in zip(peaks, widths_nm):
                if peak_idx < len(profile_smooth) and width_nm > 0 and nm_per_pixel > 0:
                    peak_value = profile_smooth[peak_idx]
                    # 计算半高宽位置（波谷的半高宽）
                    profile_min = min(profile_smooth)
                    half_max = profile_min + (peak_value - profile_min) * 0.5
                    
                    # 将纳米宽度转换为像素宽度
                    width_pixels = width_nm / nm_per_pixel
                    
                    # 绘制水平线段表示宽度
                    x_start = max(0, peak_idx - width_pixels / 2)
                    x_end = min(len(profile_smooth) - 1, peak_idx + width_pixels / 2)
                    
                    fig.add_shape(
                        type="line",
                        x0=x_start,
                        y0=half_max,
                        x1=x_end,
                        y1=half_max,
                        line=dict(color="red", width=3, dash="dash"),
                    )
    
    # 添加平均厚度标注
    if mean_width_nm is not None:
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"平均厚度: {mean_width_nm:.2f} nm",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=2,
        )
    
    # 更新布局
    fig.update_layout(
        title="灰度剖面曲线（单层厚度测量）",
        xaxis_title="位置 (pixels)",
        yaxis_title="灰度值",
        width=600,
        height=400,
        template='plotly',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return fig


def format_dataframe_with_styling(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """美化数据表格。
    
    Args:
        df: 原始 DataFrame
        
    Returns:
        (格式化后的 DataFrame, column_config 字典) 元组
    """
    # 创建列配置字典
    column_config = {}
    
    # 格式化数值列（保留2位小数）
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        column_config[col] = st.column_config.NumberColumn(
            col,
            format="%.2f"
        )
    
    # 为状态列添加颜色
    if '状态' in df.columns:
        column_config['状态'] = st.column_config.TextColumn(
            '状态',
            help="分析状态"
        )
    
    return df, column_config

