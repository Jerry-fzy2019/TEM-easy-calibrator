"""
膜厚校准页面 - Streamlit Dashboard 风格。
"""

import sys
import os
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.thickness.thickness import TEMCalibrationApp
from src.core.thickness.image_io import load_image
from src.ui_streamlit.utils import (
    draw_roi_box,
    create_fft_heatmap_plotly,
    create_fft_profile_plotly,
    create_profile_plotly,
    create_metric_cards,
    format_dataframe_with_styling,
)
from src.core.common.vision_utils import draw_debug_overlay
from src.ui_streamlit.scale_calibration_dialog import scale_calibration_dialog

st.set_page_config(page_title="膜厚校准", layout="wide")

st.title("📏 TEM 膜厚校准")

# ========== 侧边栏参数设置 ==========
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 图像预处理参数
    with st.expander("🖼️ 图像预处理", expanded=True):
        gaussian_blur = st.slider(
            "高斯模糊 (σ)",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=0.5,
            help="高斯模糊的标准差，0表示不模糊，值越大模糊程度越高"
        )
        contrast_alpha = st.slider(
            "对比度", 
            min_value=0.5, 
            max_value=3.0, 
            value=1.0,
            step=0.1,
            help="对比度增强系数，值越大对比度越高"
        )
        brightness_beta = st.slider(
            "亮度", 
            min_value=-100, 
            max_value=100, 
            value=0, 
            step=5,
            help="亮度调整值，正数变亮，负数变暗"
        )
    
    # 显微镜检定参数：放大倍率 & 标准条纹宽度
    with st.expander("🔍 显微镜检定参数", expanded=False):
        nominal_mag = st.number_input(
            "标称放大倍率",
            min_value=100.0,
            max_value=2_000_000.0,
            value=st.session_state.get("thickness_nominal_mag", 100000.0),
            step=100.0,
            help="显微镜设置的标称放大倍率（例如 100000 表示 100k×）",
            key="thickness_nominal_mag_input",
        )
        st.session_state["thickness_nominal_mag"] = nominal_mag
        
        standard_stripe_width = st.number_input(
            "标准条纹宽度 (nm)",
            min_value=0.01,
            max_value=100.0,
            value=st.session_state.get("thickness_standard_stripe_width", 0.34),
            step=0.01,
            help="标准样品给出的单层条纹真实厚度（例如 Si 晶面 0.34 nm）",
            key="thickness_standard_stripe_width_input",
        )
        st.session_state["thickness_standard_stripe_width"] = standard_stripe_width

    # 膜阈值设定（用于判断黑条纹/膜的波谷深度）
    with st.expander("📊 膜阈值设定", expanded=True):
        peak_height_min = st.slider(
            "峰谷检测阈值",
            min_value=0.05,
            max_value=0.95,
            value=st.session_state.get("thickness_peak_height_min", 0.2),
            step=0.05,
            help="归一化后用于检测黑条纹波谷的最小深度，越大则只认更深的谷（更黑的条纹）",
            key="thickness_peak_height_slider",
        )
        st.session_state["thickness_peak_height_min"] = peak_height_min
        # 左侧灰度直方图：若有最近一次剖面数据则显示，便于判断阈值
        if st.session_state.get("thickness_last_profile_raw"):
            prof = np.array(st.session_state["thickness_last_profile_raw"])
            if len(prof) > 0:
                st.caption("最近划线剖面灰度分布（用于判断阈值）")
                try:
                    import plotly.graph_objects as go
                    hist, bins = np.histogram(prof, bins=min(50, max(20, len(prof) // 10)), range=(prof.min(), prof.max()))
                    fig_hist = go.Figure(data=[go.Bar(x=(bins[:-1] + bins[1:]) / 2, y=hist, name="灰度")])
                    fig_hist.update_layout(
                        height=180,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="灰度值",
                        yaxis_title="频数",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception:
                    pass

# ========== 主工作区 ==========
st.header("📋 操作步骤")
thickness_step = st.session_state.get("thickness_step", 0)
col_step1, col_step2, col_step3, col_step4 = st.columns(4)
with col_step1:
    if thickness_step >= 1:
        st.success("✅ 步骤1: 上传图像")
    else:
        st.info("步骤1: 上传图像")
with col_step2:
    if thickness_step >= 2:
        st.success("✅ 步骤2: 换算像素比例")
    else:
        st.info("步骤2: 换算像素比例")
with col_step3:
    if thickness_step >= 3:
        st.success("✅ 步骤3: 在条纹上划线")
    else:
        st.info("步骤3: 在条纹上划线")
with col_step4:
    if thickness_step >= 4:
        st.success("✅ 步骤4: 查看结果")
    else:
        st.info("步骤4: 查看结果")
st.divider()

st.subheader("📤 步骤1: 上传图像")
uploaded_file = st.file_uploader(
    "上传 TEM 图像",
    type=['tif', 'tiff', 'png', 'jpg', 'jpeg', 'dm3'],
    help="支持 TIF, PNG, JPG, DM3 格式",
    key="thickness_uploaded_file",
)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.thickness_step = 1
elif 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if uploaded_file is None:
    st.session_state.thickness_step = 0
    st.info("👆 请上传 TEM 图像以开始测量")
    st.markdown("""
    **使用说明：** 上传图像后，在右侧完成像素比例换算，在预处理图上于条纹处划线测量单层厚度。
    """)
else:
        # 保存上传的文件到临时位置
        suffix = Path(uploaded_file.name).suffix
        if not suffix:
            suffix = '.png'
        
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        # 初始化分析器
        if 'app' not in st.session_state:
            st.session_state.app = TEMCalibrationApp()
        
        app = st.session_state.app
        
        # 加载图像到分析器（如果还没有加载或文件路径改变）
        current_file_path = st.session_state.get('current_file_path', None)
        if app.current_image is None or current_file_path != tmp_path:
            app.current_image = load_image(Path(tmp_path))
            if app.current_image is not None:
                h, w = app.current_image.shape[:2]
                app._add_log(f"图像加载成功，尺寸: {w}×{h}", "SUCCESS")
            st.session_state.current_file_path = tmp_path
            app.processed_image = None

        if app.current_image is None:
            st.error("❌ 图像加载失败，请检查文件格式（支持 TIF, PNG, JPG, DM3）")
        else:
            # ========== 图像预处理 ==========
            from src.core.thickness.image_preprocess import preprocess_image

            current_params = st.session_state.get('preprocess_params', {})
            params_changed = (
                current_params.get('gaussian_blur', 0.0) != gaussian_blur or
                current_params.get('contrast_alpha', 1.0) != contrast_alpha or
                current_params.get('brightness_beta', 0) != brightness_beta
            )

            if app.processed_image is None or params_changed:
                app.processed_image = preprocess_image(
                    app.current_image,
                    noise_kernel=1,
                    contrast_alpha=contrast_alpha,
                    brightness_beta=brightness_beta,
                    gaussian_blur=gaussian_blur,
                )
                st.session_state.preprocess_params = {
                    'gaussian_blur': gaussian_blur,
                    'contrast_alpha': contrast_alpha,
                    'brightness_beta': brightness_beta,
                }
                app.fft_analyzer.original_image = app.current_image.copy()
                app.fft_analyzer.gray_image = app.processed_image.copy()

            # 预处理后的图像（仅图像，不挤比例尺）
            st.subheader("🖼️ 预处理后的图像")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("**原始图像**")
                st.image(app.current_image, caption=f"尺寸: {app.current_image.shape[1]}×{app.current_image.shape[0]} 像素")
            with col_img2:
                st.markdown("**预处理后图像**")
                st.image(app.processed_image, caption=f"高斯模糊: {gaussian_blur:.1f}, 对比度: {contrast_alpha:.1f}, 亮度: {brightness_beta}")

            # 步骤2：像素比例换算单开一栏（整行，不挤在右侧）
            st.subheader("📏 步骤2: 像素比例换算")
            current_nm_per_pixel = app.fft_analyzer.nm_per_pixel if app.fft_analyzer.nm_per_pixel > 0 else None
            uploaded_file_path = st.session_state.get('current_file_path', None)

            def on_calibrated_callback(val: float):
                app.fft_analyzer.nm_per_pixel = val

            nm_per_pixel, scale_info = scale_calibration_dialog(
                image=app.processed_image if app.processed_image is not None else app.current_image,
                image_file_path=uploaded_file_path,
                current_nm_per_pixel=current_nm_per_pixel,
                session_state_key="thickness_scale_calibration",
                on_calibrated=on_calibrated_callback,
            )
            if nm_per_pixel and nm_per_pixel > 0:
                app.fft_analyzer.nm_per_pixel = nm_per_pixel
                st.session_state.thickness_step = 2

        st.session_state.current_file_path = tmp_path

        # ========== 工具栏与划线 ==========
        if app.current_image is None or app.processed_image is None:
            st.info("请先上传并加载成功图像后再进行划线测量。")
        else:
            tool_mode = st.radio(
                "选择工具",
                ["线剖面分析"],
                horizontal=True,
                key="tool_mode",
                help="选择测量工具：线剖面分析"
            )

            # ========== 交互式测量区域 ==========
            try:
                image_for_display = app.processed_image.copy()
                if image_for_display.dtype != np.uint8:
                    if image_for_display.max() <= 1.0:
                        image_for_display = (image_for_display * 255).astype(np.uint8)
                    else:
                        image_for_display = image_for_display.astype(np.uint8)

                if len(image_for_display.shape) == 2:
                    image_pil = Image.fromarray(image_for_display, mode='L').convert('RGB')
                elif len(image_for_display.shape) == 3:
                    if image_for_display.shape[2] == 3:
                        image_pil = Image.fromarray(image_for_display, mode='RGB')
                    elif image_for_display.shape[2] == 4:
                        image_pil = Image.fromarray(image_for_display, mode='RGBA').convert('RGB')
                    else:
                        image_pil = Image.fromarray(image_for_display[:, :, 0], mode='L').convert('RGB')
                else:
                    st.error(f"❌ 不支持的图像维度: {image_for_display.shape}")
                    st.stop()

                img_width, img_height = image_pil.size
                h_np, w_np = app.processed_image.shape[:2]

                if img_width == 0 or img_height == 0:
                    st.error("❌ 图像尺寸无效")
                    st.stop()

                if img_width != w_np or img_height != h_np:
                    st.warning(f"⚠️ 尺寸不一致：PIL ({img_width}, {img_height}) vs numpy ({w_np}, {h_np})")

                max_canvas_size = 1200
                if max(img_width, img_height) <= max_canvas_size:
                    canvas_width = img_width
                    canvas_height = img_height
                else:
                    max_display_size = 800
                    scale = min(1.0, max_display_size / max(img_width, img_height))
                    canvas_width = int(img_width * scale)
                    canvas_height = int(img_height * scale)
                    if canvas_width < 100:
                        canvas_width = 100
                        canvas_height = int(img_height * (canvas_width / img_width))
                        if canvas_height < 100:
                            canvas_height = 100
                            canvas_width = int(img_width * (canvas_height / img_height))

                if tool_mode == "线剖面分析":
                    drawing_mode = "line"
                    title = "📏 线剖面分析"
                    help_text = "在条纹上画一条线（建议垂直于条纹方向）进行单层厚度测量"
                if st.session_state.get("thickness_step", 0) >= 2:
                    st.session_state.thickness_step = 3

                with st.expander(title, expanded=True):
                    # 创建 Canvas（使用动态 key 以便切换工具时清空）
                    canvas_key = f"canvas_{tool_mode.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=3,
                        stroke_color="#FF0000",
                        background_image=image_pil,
                        update_streamlit=True,
                        drawing_mode=drawing_mode,
                        point_display_radius=0,
                        key=canvas_key,
                        width=canvas_width,
                        height=canvas_height,
                    )

                    # 处理 Canvas 事件
                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data.get("objects", [])
                        if objects:
                            last_obj = objects[-1]
                            obj_type = last_obj.get("type")

                            h_np, w_np = app.processed_image.shape[:2]
                            scale_x = w_np / canvas_width
                            scale_y = h_np / canvas_height

                            if tool_mode == "线剖面分析" and obj_type == "line":
                                canvas_x1 = last_obj.get("x1", 0)
                                canvas_y1 = last_obj.get("y1", 0)
                                canvas_x2 = last_obj.get("x2", 0)
                                canvas_y2 = last_obj.get("y2", 0)

                                if canvas_x1 < 0 or canvas_y1 < 0 or canvas_x2 < 0 or canvas_y2 < 0:
                                    canvas_x1 = canvas_x1 + canvas_width / 2
                                    canvas_y1 = canvas_y1 + canvas_height / 2
                                    canvas_x2 = canvas_x2 + canvas_width / 2
                                    canvas_y2 = canvas_y2 + canvas_height / 2

                                canvas_x1 = max(0, min(canvas_x1, canvas_width - 1))
                                canvas_y1 = max(0, min(canvas_y1, canvas_height - 1))
                                canvas_x2 = max(0, min(canvas_x2, canvas_width - 1))
                                canvas_y2 = max(0, min(canvas_y2, canvas_height - 1))

                                x1_float = canvas_x1 * scale_x
                                y1_float = canvas_y1 * scale_y
                                x2_float = canvas_x2 * scale_x
                                y2_float = canvas_y2 * scale_y

                                x1 = int(round(x1_float))
                                y1 = int(round(y1_float))
                                x2 = int(round(x2_float))
                                y2 = int(round(y2_float))

                                x1 = max(0, min(x1, w_np - 1))
                                y1 = max(0, min(y1, h_np - 1))
                                x2 = max(0, min(x2, w_np - 1))
                                y2 = max(0, min(y2, h_np - 1))

                                point1 = (x1, y1)
                                point2 = (x2, y2)

                                peak_height_min = st.session_state.get("thickness_peak_height_min", 0.2)
                                result_line = app.analyze_manual_line(point1, point2, peak_height_min=peak_height_min)
                                if result_line.get("profile_raw"):
                                    st.session_state["thickness_last_profile_raw"] = result_line["profile_raw"]

                                if result_line.get('status') == 'success':
                                    st.session_state.thickness_step = 4
                                    thickness = result_line.get('layer_thickness_nm', 0)
                                    std_thick = result_line.get('std_thickness_nm', 0)

                                    st.markdown("### 📊 测量数据")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("单层厚度", f"{thickness:.3f} nm")
                                    with col2:
                                        st.metric("标准差", f"{std_thick:.3f} nm")

                                    nominal_mag = st.session_state.get("thickness_nominal_mag", None)
                                    standard_stripe_width = st.session_state.get("thickness_standard_stripe_width", None)

                                    if standard_stripe_width and standard_stripe_width > 0:
                                        st.divider()
                                        st.markdown("### 📐 误差计算")
                                        # 实际放大倍率 = 实际膜厚 / 标称膜厚（商）
                                        actual_ratio = thickness / standard_stripe_width
                                        # 误差 = (测量厚度 - 标称膜厚) / 标称膜厚 × 100%
                                        thickness_error_percent = (thickness - standard_stripe_width) / standard_stripe_width * 100.0
                                        st.markdown("**计算公式：**")
                                        st.latex(r"""
                                        \begin{align}
                                        \text{实际放大倍率} &= \frac{\text{实际膜厚}}{\text{标称膜厚}} = \frac{t_{\text{测量}}}{t_{\text{标称}}} \\
                                        \text{厚度误差} (\%) &= \frac{t_{\text{测量}} - t_{\text{标称}}}{t_{\text{标称}}} \times 100\%
                                        \end{align}
                                        """)
                                        st.markdown("**计算过程：**")
                                        st.markdown(f"""
                                        - 实际放大倍率 = 实际膜厚 / 标称膜厚 = {thickness:.3f} / {standard_stripe_width:.3f} = **{actual_ratio:.6f}**
                                        - 厚度误差 = ({thickness:.3f} − {standard_stripe_width:.3f}) / {standard_stripe_width:.3f} × 100% = **{thickness_error_percent:+.3f}%**
                                        """)
                                        st.divider()
                                        st.markdown("### ✅ 检定结果")
                                        error_threshold = 5.0
                                        is_ok = abs(thickness_error_percent) <= error_threshold
                                        col_m1, col_m2, col_m3 = st.columns(3)
                                        with col_m1:
                                            st.metric("标称膜厚 (nm)", f"{standard_stripe_width:.3f}")
                                        with col_m2:
                                            st.metric("实际放大倍率", f"{actual_ratio:.6f}")
                                        with col_m3:
                                            st.metric("厚度误差 (%)", f"{thickness_error_percent:+.3f} %")
                                        if nominal_mag and nominal_mag > 0:
                                            st.caption(f"标称放大倍率参考: {nominal_mag:,.0f} ×")
                                        if is_ok:
                                            st.success(f"✅ 合格（|厚度误差| ≤ {error_threshold:.1f}%）")
                                        else:
                                            st.error(f"❌ 不合格（|厚度误差| ≤ {error_threshold:.1f}%）")

                                    st.divider()
                                    st.markdown("### 📈 灰度剖面曲线分析")
                                    profile_raw = result_line.get('profile_raw')
                                    profile_data = result_line.get('profile_data')

                                    if profile_raw:
                                        st.subheader("📈 灰度剖面曲线（原始图像灰度）")
                                        import plotly.graph_objects as go
                                        x_axis = list(range(len(profile_raw)))
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=x_axis,
                                            y=profile_raw,
                                            mode='lines',
                                            name='图像灰度值',
                                            line=dict(width=2, color='blue'),
                                            hovertemplate='位置: %{x} px<br>灰度值: %{y:.1f}<extra></extra>',
                                        ))
                                        if profile_data and profile_data.get('peaks'):
                                            peaks = profile_data.get('peaks', [])
                                            left_ips = profile_data.get('left_ips', [])
                                            right_ips = profile_data.get('right_ips', [])
                                            widths_nm = profile_data.get('widths_nm', [])
                                            for i, peak_idx in enumerate(peaks):
                                                if peak_idx < len(profile_raw):
                                                    peak_val = profile_raw[int(peak_idx)]
                                                    fig.add_trace(go.Scatter(
                                                        x=[peak_idx],
                                                        y=[peak_val],
                                                        mode='markers',
                                                        name=f'黑色条纹 {i+1}' if i == 0 else '',
                                                        marker=dict(size=10, color='red', symbol='circle'),
                                                        showlegend=(i == 0),
                                                    ))
                                            for i, (left_ip, right_ip, width_nm) in enumerate(zip(left_ips, right_ips, widths_nm)):
                                                left_idx = int(round(left_ip))
                                                right_idx = int(round(right_ip))
                                                if 0 <= left_idx < len(profile_raw) and 0 <= right_idx < len(profile_raw):
                                                    peak_idx = int(round(peaks[i])) if i < len(peaks) else int(round((left_ip + right_ip) / 2))
                                                    if 0 <= peak_idx < len(profile_raw):
                                                        peak_val = profile_raw[peak_idx]
                                                        left_val = profile_raw[left_idx]
                                                        right_val = profile_raw[right_idx]
                                                        avg_boundary = (left_val + right_val) / 2
                                                        fwhm_height = peak_val + (avg_boundary - peak_val) * 0.5
                                                        fig.add_trace(go.Scatter(
                                                            x=[left_ip, right_ip],
                                                            y=[fwhm_height, fwhm_height],
                                                            mode='lines',
                                                            name=f'半高宽 {i+1}' if i == 0 else '',
                                                            line=dict(width=2, color='red', dash='dash'),
                                                            showlegend=(i == 0),
                                                        ))
                                        fig.update_layout(
                                            title="图像灰度剖面曲线",
                                            xaxis_title="位置 (像素)",
                                            yaxis_title="灰度值",
                                            width=800,
                                            height=400,
                                            template='plotly_white',
                                            hovermode='x unified',
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    elif profile_data:
                                        st.subheader("📈 灰度剖面曲线")
                                        fig_profile = create_profile_plotly(profile_data)
                                        st.plotly_chart(fig_profile, use_container_width=True)
                                else:
                                    st.error(f"❌ {result_line.get('error_message', '线剖面分析失败')}")
                        else:
                            if tool_mode == "线剖面分析":
                                st.info("👆 请在条纹上画一条线（建议垂直于条纹方向）")
            except Exception as e:
                st.error(f"❌ 图像处理失败: {str(e)}")
                import traceback
                with st.expander("错误详情"):
                    st.code(traceback.format_exc())
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    pass
