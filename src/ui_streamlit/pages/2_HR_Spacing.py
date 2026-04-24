"""
高分辨晶面间距校准页面。
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.hr_spacing.hr_spacing import HRSpacingAnalyzer
from src.core.thickness.image_io import load_image
from src.core.common.scale_calibration import convert_nm_per_pixel_to_saed_factor
from src.ui_streamlit.scale_calibration_dialog import scale_calibration_dialog

st.set_page_config(page_title="晶面间距", layout="wide")

st.title("🔬 高分辨晶面间距校准")

# ========== 侧边栏参数设置 ==========
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # Friedel 过滤参数
    with st.expander("🔄 Friedel 过滤", expanded=False):
        apply_friedel_filter = st.checkbox(
            "应用 Friedel 过滤",
            value=True,
            help="去除对称的峰对，只保留独立的晶面族"
        )
        
        d_tolerance = st.number_input(
            "d-spacing 容差 (nm)",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="判断两个峰是否为同一晶面族的 d-spacing 容差"
        )
        
        angle_tolerance_deg = st.slider(
            "角度容差 (度)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="判断两个峰是否为对称峰的角度容差"
        )

# ========== 主工作区 ==========
st.header("📋 操作步骤")

step = st.session_state.get("hr_step", 1)
col_step1, col_step2, col_step3, col_step4 = st.columns(4)
with col_step1:
    if step >= 1:
        st.success("✅ 步骤1: 上传图像并换算像素比例")
    else:
        st.info("步骤1: 上传图像并换算像素比例")
with col_step2:
    if step >= 2:
        st.success("✅ 步骤2: 选择中心")
    else:
        st.info("步骤2: 选择中心")
with col_step3:
    if step >= 3:
        st.success("✅ 步骤3: 选择斑点")
    else:
        st.info("步骤3: 选择斑点")
with col_step4:
    if step >= 4:
        st.success("✅ 步骤4: 查看结果")
    else:
        st.info("步骤4: 查看结果")

st.divider()

# ========== 步骤1: 上传图像和像素比例换算 ==========
st.subheader("📤 步骤1: 上传图像并换算像素比例")

uploaded_file = st.file_uploader(
    "上传 HRTEM 图像",
    type=['tif', 'tiff', 'png', 'jpg', 'jpeg', 'dm3'],
    help="支持 TIF, PNG, JPG, DM3 格式",
    key="hr_uploaded_file"
)

if uploaded_file is None:
    st.info("👆 请先上传 HRTEM 图像")
    st.markdown("""
                    ### 📖 使用说明
                    
                    本模块用于测量高分辨 TEM 图像的晶面间距，操作流程如下：
                    
                    1. **上传图像**: 上传包含清晰晶格条纹的 HRTEM 图像
                    2. **换算像素比例**: 在右侧完成像素比例换算（如果图像有标尺）
                    3. **选择中心**: 在 FFT 图上点击选择中心位置（零频位置）
                    4. **选择斑点**: 在 FFT 图上点击选择一个衍射斑点
                    5. **IFFT重构**: 系统自动使用中心+斑点+对称点进行IFFT重构
                    6. **划线测量**: 在重构图像上划线测量晶面间距
                    
                    ### ⚠️ 注意事项
                    
                    - 图像下方1/10区域（通常包含比例尺）会自动排除，不参与分析
                    - 需要先选择中心位置，再选择一个衍射斑点
                    - 系统会自动使用该斑点及其对称点进行IFFT重构
                    """)
    if 'hr_step' not in st.session_state:
        st.session_state.hr_step = 0
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
    
    try:
        # 初始化分析器
        if 'hr_analyzer' not in st.session_state:
            st.session_state.hr_analyzer = HRSpacingAnalyzer()
        
        analyzer = st.session_state.hr_analyzer
        
        # 加载图像（使用缓存避免重复加载）
        if 'hr_current_image' in st.session_state and st.session_state.get('hr_current_file_path') == tmp_path:
            image = st.session_state.hr_current_image
        else:
            image = load_image(Path(tmp_path))
            if image is None:
                st.error("❌ 图像加载失败，请检查文件格式")
                st.stop()
            # 保存图像和路径到session_state
            st.session_state.hr_current_image = image
            st.session_state.hr_current_file_path = tmp_path
        
        # 显示原始图像（限制显示尺寸，避免大图无法显示）
        st.subheader("🖼️ 原始图像")
        img_h, img_w = image.shape[:2]
        max_show = 1200
        if max(img_w, img_h) > max_show:
            r = max_show / max(img_w, img_h)
            show_w, show_h = int(img_w * r), int(img_h * r)
            if len(image.shape) == 2:
                img_pil = Image.fromarray(image, mode='L').convert('RGB')
            else:
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_pil = img_pil.resize((show_w, show_h), Image.Resampling.LANCZOS)
            st.image(np.array(img_pil), caption=f"图像尺寸: {img_w}×{img_h} 像素（已缩小显示）", use_column_width=True)
        else:
            if len(image.shape) == 2:
                st.image(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), caption=f"图像尺寸: {img_w}×{img_h} 像素", use_column_width=True)
            else:
                st.image(image, caption=f"图像尺寸: {img_w}×{img_h} 像素", use_column_width=True)

        # 像素比例换算：单独一栏，宽度与漂移/污染/膜厚模块一致
        st.subheader("📏 步骤1: 像素比例换算")
        from src.ui_streamlit.scale_calibration_dialog import scale_calibration_dialog
        current_nm_per_pixel = st.session_state.get("hr_nm_per_pixel", None)
        uploaded_file_path = st.session_state.get('hr_current_file_path', None)

        # 顶部显示当前像素比例，风格对齐膜厚模块
        col_scale1, col_scale2 = st.columns([2, 3])
        with col_scale1:
            if current_nm_per_pixel and current_nm_per_pixel > 0:
                st.info(f"当前像素比例: **{current_nm_per_pixel:.6f} nm/px**")
            else:
                st.info("当前像素比例: **未设置**")

        def on_calibrated_callback(val: float):
            """校准成功后的回调函数"""
            st.session_state.hr_nm_per_pixel = val

        nm_per_pixel, scale_info = scale_calibration_dialog(
            image=st.session_state.get('hr_current_image', None),
            image_file_path=uploaded_file_path,
            current_nm_per_pixel=current_nm_per_pixel,
            session_state_key="hr_scale_calibration",
            on_calibrated=on_calibrated_callback,
        )

        if nm_per_pixel and nm_per_pixel > 0:
            st.session_state.hr_nm_per_pixel = nm_per_pixel
            st.success(f"✅ 像素比例已设置为 {nm_per_pixel:.6f} nm/px")
        else:
            st.warning("⚠️ 尚未完成像素比例换算，请根据标尺或已知距离完成换算后再继续后续步骤。")
        
        st.session_state.hr_step = 1

        st.divider()

        # ========== 步骤1.5: FFT预处理参数 ==========
        st.subheader("⚙️ FFT预处理参数")
        
        col_pre1, col_pre2, col_pre3 = st.columns(3)
        
        with col_pre1:
            # 对数缩放参数
            use_log_scale = st.checkbox("使用对数缩放", value=True, help="对FFT幅度谱应用对数变换以增强对比度")
            log_scale_factor = st.slider(
                "对数缩放因子",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="对数变换的缩放因子，值越大对比度越高"
            )
        
        with col_pre2:
            # 归一化参数
            normalize_min = st.slider(
                "归一化最小值 (%)",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
                help="归一化时的最小百分比，用于裁剪低值"
            )
            normalize_max = st.slider(
                "归一化最大值 (%)",
                min_value=50,
                max_value=100,
                value=100,
                step=1,
                help="归一化时的最大百分比，用于裁剪高值"
            )
        
        with col_pre3:
            # 显示增强参数
            gamma_correction = st.slider(
                "Gamma校正",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Gamma校正值，用于调整图像亮度"
            )
            contrast_enhance = st.slider(
                "对比度增强",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="对比度增强倍数"
            )
        
        st.divider()

        # ========== 步骤2: 选择中心 ==========
        st.subheader("🎯 步骤2: 选择中心")

        # 计算FFT
        from src.core.thickness.fft_core import FFTPeriodicityAnalyzer
        fft_analyzer = FFTPeriodicityAnalyzer()
        fft_analyzer.original_image = image.copy()
        fft_analyzer.gray_image = image.copy()
        fft_analyzer.set_roi(0, 0, image.shape[1], image.shape[0])

        # 设置像素比例
        nm_per_pixel = st.session_state.get("hr_nm_per_pixel", None)
        if nm_per_pixel and nm_per_pixel > 0:
            fft_analyzer.nm_per_pixel = nm_per_pixel

        # 计算FFT（增加异常捕获，避免整页空白）
        try:
            fft_shifted, magnitude_log = fft_analyzer.compute_fft()
        except Exception as e:
            st.error(f"❌ FFT 计算失败，请检查图像是否有效或尝试重新上传: {e}")
            import traceback
            with st.expander("FFT 错误详情"):
                st.code(traceback.format_exc())
            st.stop()
        
        # 保存到session_state以便后续使用
        st.session_state.hr_fft_shifted = fft_shifted
        st.session_state.hr_magnitude_log = magnitude_log
        st.session_state.hr_fft_analyzer = fft_analyzer

        # 应用预处理参数
        magnitude_processed = magnitude_log.copy()
        
        # 对数缩放
        if use_log_scale:
            magnitude_processed = np.log1p(magnitude_processed * log_scale_factor)
        
        # 归一化（使用百分比裁剪）
        min_val = np.percentile(magnitude_processed, normalize_min)
        max_val = np.percentile(magnitude_processed, normalize_max)
        if max_val > min_val:
            magnitude_processed = np.clip((magnitude_processed - min_val) / (max_val - min_val), 0, 1)
        else:
            magnitude_processed = (magnitude_processed - magnitude_processed.min()) / (magnitude_processed.max() - magnitude_processed.min() + 1e-10)
        
        # Gamma校正
        if gamma_correction != 1.0:
            magnitude_processed = np.power(magnitude_processed, 1.0 / gamma_correction)
        
        # 对比度增强
        magnitude_processed = magnitude_processed * contrast_enhance
        magnitude_processed = np.clip(magnitude_processed, 0, 1)
        
        # 转换为8位图像
        magnitude_normalized = (magnitude_processed * 255).astype(np.uint8)
        if len(magnitude_normalized.shape) == 2:
            fft_pil_full = Image.fromarray(magnitude_normalized, mode='L').convert('RGB')
        else:
            fft_pil_full = Image.fromarray(magnitude_normalized, mode='RGB')

        fft_h, fft_w = magnitude_log.shape

        # 默认自动中心：使用 FFT 幅度图中最亮点，用户仍可手动点击覆盖
        if 'hr_center' not in st.session_state:
            try:
                max_idx = int(np.argmax(magnitude_log))
                cy, cx = divmod(max_idx, fft_w)
                st.session_state.hr_center = {'x': float(cx), 'y': float(cy)}
                st.session_state.hr_step = max(st.session_state.get('hr_step', 1), 2)
            except Exception:
                # 如果自动检测失败则保持为空，依赖用户手动选择
                pass

        # 计算canvas尺寸（限制最大尺寸，避免大图无法显示，适当放大便于标注）
        # 将 FFT 交互图尽量放大一些，便于选中心/斑点
        max_display_width = 1600
        if fft_w <= max_display_width and fft_h <= max_display_width:
            canvas_width = fft_w
            canvas_height = fft_h
            fft_pil = fft_pil_full
        else:
            scale = min(max_display_width / fft_w, max_display_width / fft_h)
            canvas_width = int(fft_w * scale)
            canvas_height = int(fft_h * scale)
            fft_pil = fft_pil_full.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        # 确保最小尺寸，避免过小
        min_size = 400
        if canvas_width < min_size or canvas_height < min_size:
            scale2 = max(min_size / canvas_width, min_size / canvas_height)
            canvas_width = int(canvas_width * scale2)
            canvas_height = int(canvas_height * scale2)
            fft_pil = fft_pil_full.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        def _canvas_point_to_center(pt, radius=5):
            """将 canvas point 对象转为圆心坐标（Fabric 可能返回 left/top 左上角）"""
            x, y = pt.get("x"), pt.get("y")
            if x is not None and y is not None:
                return float(x), float(y)
            left, top = pt.get("left"), pt.get("top")
            if left is not None and top is not None:
                w = float(pt.get("width", radius * 2))
                h = float(pt.get("height", radius * 2))
                return left + w / 2, top + h / 2
            return float(pt.get("x1", 0)), float(pt.get("y1", 0))

        # 显示FFT图（选择中心）——使用缩放图避免无法显示
        st.markdown("**FFT 图 - 请在下方图上点击选择中心位置（零频位置，通常在图像中央亮斑处）**")
        st.info("💡 提示：FFT图像的中心通常是最亮的区域，对应原始图像的低频分量（直流分量）")

        try:
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=fft_pil,
                update_streamlit=True,
                drawing_mode="point",
                point_display_radius=5,
                key="hr_center_canvas",
                width=canvas_width,
                height=canvas_height,
            )
        except Exception as e:
            st.error(f"中心选择Canvas显示错误: {e}")
            import traceback
            with st.expander("错误详情"):
                st.code(traceback.format_exc())
            canvas_result = None

        # 处理中心选择
        if canvas_result is not None and canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                last_point = objects[-1]
                canvas_x_raw, canvas_y_raw = _canvas_point_to_center(last_point, radius=5)

                canvas_x = canvas_x_raw
                canvas_y = canvas_y_raw
                if canvas_x < 0 or canvas_y < 0:
                    canvas_x = canvas_x + canvas_width / 2
                    canvas_y = canvas_y + canvas_height / 2
                canvas_x = max(0, min(canvas_x, canvas_width - 1))
                canvas_y = max(0, min(canvas_y, canvas_height - 1))

                scale_x = fft_w / canvas_width
                scale_y = fft_h / canvas_height
                x = canvas_x * scale_x
                y = canvas_y * scale_y

                # 确保坐标在有效范围内
                x = max(0, min(x, fft_w - 1))
                y = max(0, min(y, fft_h - 1))

                # 只在坐标真正改变时才更新
                current_center = st.session_state.get('hr_center')
                if current_center is None or abs(current_center.get('x', 0) - x) > 0.1 or abs(current_center.get('y', 0) - y) > 0.1:
                    st.session_state.hr_center = {'x': float(x), 'y': float(y)}
                    st.success(f"✅ 已选择中心位置: ({x:.1f}, {y:.1f})")
                    st.session_state.hr_step = 2
                    st.rerun()

        # 显示已选择的信息
        if st.session_state.get('hr_center') is not None:
            center = st.session_state.hr_center
            st.info(f"**中心位置**: ({center['x']:.1f}, {center['y']:.1f})（默认自动检测，点击可手动修改）")
        else:
            st.warning("⚠️ 请先点击选择中心位置")

        st.divider()

        # ========== 步骤3: 选择斑点 ==========
        if st.session_state.get('hr_center') is not None:
            st.subheader("🔬 步骤3: 选择衍射斑点")

            # 自动修正选项（是否自动吸附到局部最亮点）
            st.session_state.setdefault("hr_auto_refine_spot", True)
            auto_refine = st.checkbox(
                "启用自动修正到局部最亮点（推荐）",
                value=st.session_state["hr_auto_refine_spot"],
                key="hr_auto_refine_spot_checkbox",
                help="在您点击附近的小范围内搜索最亮的衍射斑点并自动对齐。",
            )
            st.session_state["hr_auto_refine_spot"] = auto_refine

            # 初始化斑点列表
            if 'hr_selected_spots' not in st.session_state:
                st.session_state.hr_selected_spots = []

            # 先在显示尺寸上做图，十字位置与用户点击一致（按显示坐标画十字）
            if len(magnitude_normalized.shape) == 2:
                fft_display = cv2.cvtColor(
                    cv2.resize(magnitude_normalized, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_GRAY2RGB
                )
            else:
                fft_display = cv2.resize(magnitude_normalized, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)
            cw, ch = canvas_width, canvas_height
            center = st.session_state.hr_center
            # 中心在显示图上的坐标（与步骤2 的 canvas 一致）
            cx_disp = int(round(center['x'] * cw / fft_w))
            cy_disp = int(round(center['y'] * ch / fft_h))
            cross_len = 12
            if 0 <= cx_disp < cw and 0 <= cy_disp < ch:
                cv2.line(fft_display, (max(0, cx_disp - cross_len), cy_disp),
                         (min(cw - 1, cx_disp + cross_len), cy_disp), (255, 0, 0), 2)
                cv2.line(fft_display, (cx_disp, max(0, cy_disp - cross_len)),
                         (cx_disp, min(ch - 1, cy_disp + cross_len)), (255, 0, 0), 2)
            for spot in st.session_state.hr_selected_spots:
                # 选中斑点（白色十字）
                sx = float(spot['x'])
                sy = float(spot['y'])
                sx_disp = int(round(sx * cw / fft_w))
                sy_disp = int(round(sy * ch / fft_h))
                if 0 <= sx_disp < cw and 0 <= sy_disp < ch:
                    x1 = max(0, sx_disp - cross_len)
                    x2 = min(cw - 1, sx_disp + cross_len)
                    y1 = max(0, sy_disp - cross_len)
                    y2 = min(ch - 1, sy_disp + cross_len)
                    cv2.line(fft_display, (x1, sy_disp), (x2, sy_disp), (255, 255, 255), 2)
                    cv2.line(fft_display, (sx_disp, y1), (sx_disp, y2), (255, 255, 255), 2)

                    # 关于中心的对称点（黄色十字），IFFT 时也会自动使用这个对称点
                    sym_x = 2.0 * center['x'] - sx
                    sym_y = 2.0 * center['y'] - sy
                    sym_disp_x = int(round(sym_x * cw / fft_w))
                    sym_disp_y = int(round(sym_y * ch / fft_h))
                    if 0 <= sym_disp_x < cw and 0 <= sym_disp_y < ch:
                        x1s = max(0, sym_disp_x - cross_len)
                        x2s = min(cw - 1, sym_disp_x + cross_len)
                        y1s = max(0, sym_disp_y - cross_len)
                        y2s = min(ch - 1, sym_disp_y + cross_len)
                        cv2.line(fft_display, (x1s, sym_disp_y), (x2s, sym_disp_y), (0, 255, 255), 2)
                        cv2.line(fft_display, (sym_disp_x, y1s), (sym_disp_x, y2s), (0, 255, 255), 2)

            fft_with_center_pil = Image.fromarray(fft_display, mode='RGB')

            # 显示FFT图（选择斑点）——白色十字为已选斑点，黄色十字为对称点
            st.markdown("**FFT 图 - 请在下方图上点击选择一个衍射斑点**")
            st.markdown("*红色十字=中心，白色十字=已选斑点，黄色十字=关于中心的对称点*")
            st.info("💡 提示：系统会使用“中心 + 选中斑点 + 对称斑点”三个点进行 IFFT 重构。")

            try:
                spot_canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.5)",
                    stroke_width=2,
                    stroke_color="#FFFFFF",
                    background_image=fft_with_center_pil,
                    update_streamlit=True,
                    drawing_mode="point",
                    point_display_radius=5,
                    key="hr_spot_canvas",
                    width=canvas_width,
                    height=canvas_height,
                )
            except Exception as e:
                st.error(f"Canvas显示错误: {e}")
                spot_canvas_result = None

            # 处理斑点选择（同样用圆心坐标）
            if spot_canvas_result is not None and spot_canvas_result.json_data is not None:
                objects = spot_canvas_result.json_data.get("objects", [])
                if objects:
                    last_point = objects[-1]
                    canvas_x_raw, canvas_y_raw = _canvas_point_to_center(last_point, radius=5)
                    canvas_x = canvas_x_raw
                    canvas_y = canvas_y_raw
                    if canvas_x < 0 or canvas_y < 0:
                        canvas_x = canvas_x + canvas_width / 2
                        canvas_y = canvas_y + canvas_height / 2
                    canvas_x = max(0, min(canvas_x, canvas_width - 1))
                    canvas_y = max(0, min(canvas_y, canvas_height - 1))
                    scale_x = fft_w / canvas_width
                    scale_y = fft_h / canvas_height
                    x_raw = canvas_x * scale_x
                    y_raw = canvas_y * scale_y

                    # 确保坐标在有效范围内
                    x_raw = max(0, min(x_raw, fft_w - 1))
                    y_raw = max(0, min(y_raw, fft_h - 1))

                    x = x_raw
                    y = y_raw

                    # 在局部窗口内自动修正到最亮点
                    def _refine_peak_local_max(mag, x0, y0, radius=3):
                        h, w = mag.shape
                        xi = int(round(x0))
                        yi = int(round(y0))
                        x_min = max(0, xi - radius)
                        x_max = min(w - 1, xi + radius)
                        y_min = max(0, yi - radius)
                        y_max = min(h - 1, yi + radius)
                        patch = mag[y_min:y_max + 1, x_min:x_max + 1]
                        if patch.size == 0:
                            return x0, y0
                        dy, dx = np.unravel_index(int(np.argmax(patch)), patch.shape)
                        return float(x_min + dx), float(y_min + dy)

                    if st.session_state.get("hr_auto_refine_spot", True):
                        try:
                            x_ref, y_ref = _refine_peak_local_max(magnitude_normalized, x, y, radius=3)
                            x_ref = float(np.clip(x_ref, 0, fft_w - 1))
                            y_ref = float(np.clip(y_ref, 0, fft_h - 1))
                            if abs(x_ref - x_raw) > 0.5 or abs(y_ref - y_raw) > 0.5:
                                st.info(
                                    f"已自动修正到局部最亮点："
                                    f"原始点击 ({x_raw:.1f}, {y_raw:.1f}) → "
                                    f"修正后 ({x_ref:.1f}, {y_ref:.1f})"
                                )
                            x, y = x_ref, y_ref
                        except Exception:
                            x, y = x_raw, y_raw

                    # 只允许选择一个斑点，如果已选择则替换
                    if len(st.session_state.hr_selected_spots) > 0:
                        # 检查是否点击的是同一个位置（容差5像素）
                        existing_spot = st.session_state.hr_selected_spots[0]
                        if abs(existing_spot['x'] - x) < 5 and abs(existing_spot['y'] - y) < 5:
                            st.info("已选择该斑点，如需更换请先清空")
                        else:
                            # 替换为新的斑点
                            st.session_state.hr_selected_spots[0] = {
                                'x': float(x),
                                'y': float(y),
                                'raw_x': float(x_raw),
                                'raw_y': float(y_raw),
                            }
                            st.success(f"✅ 已选择斑点: ({x:.1f}, {y:.1f})")
                            st.session_state.hr_step = 3
                            st.rerun()
                    else:
                        # 添加第一个斑点
                        st.session_state.hr_selected_spots.append({
                            'x': float(x),
                            'y': float(y),
                            'raw_x': float(x_raw),
                            'raw_y': float(y_raw),
                        })
                        st.success(f"✅ 已选择斑点: ({x:.1f}, {y:.1f})")
                        st.session_state.hr_step = 3
                        st.rerun()

            # 显示已选择的斑点
            if st.session_state.hr_selected_spots:
                spot = st.session_state.hr_selected_spots[0]
                st.info(f"**已选择的斑点**: ({spot['x']:.1f}, {spot['y']:.1f})")
                
                if st.button("清空斑点", key="hr_clear_spot", type="secondary", use_container_width=True):
                    st.session_state.hr_selected_spots = []
                    st.session_state.hr_step = 3
                    st.rerun()
            else:
                st.info("👆 请点击选择一个衍射斑点")

        st.divider()

        # ========== 步骤4: IFFT重构和划线分析 ==========
        if st.session_state.get('hr_center') is not None and st.session_state.get('hr_selected_spots'):
            st.subheader("📊 步骤4: IFFT重构和晶面间距测量")

            # 获取像素比例
            nm_per_pixel = st.session_state.get("hr_nm_per_pixel", None)
            if nm_per_pixel and nm_per_pixel > 0:
                # 从session_state读取坐标
                center = st.session_state.get('hr_center')
                spots = st.session_state.get('hr_selected_spots', [])
                
                if center is None or not spots or len(spots) == 0:
                    st.error("❌ 请选择中心和一个斑点")
                else:
                    center_x = float(center.get('x', 0))
                    center_y = float(center.get('y', 0))
                    spot = spots[0]  # 只使用第一个斑点
                    
                    # 使用中心+斑点+对称点进行IFFT重构
                    from src.core.hr_spacing.hr_spacing import HRSpacingAnalyzer
                    analyzer = HRSpacingAnalyzer()
                    
                    # 从session_state获取FFT数据
                    fft_shifted = st.session_state.get('hr_fft_shifted')
                    magnitude_log = st.session_state.get('hr_magnitude_log')
                    
                    if fft_shifted is None or magnitude_log is None:
                        st.error("❌ FFT数据未找到，请重新上传图像")
                    else:
                        # 执行IFFT重构
                        reconstructed_image = analyzer._reconstruct_from_single_peak(
                            fft_shifted=fft_shifted,
                            center=(center_x, center_y),
                            peak=spot,
                            fft_shape=magnitude_log.shape
                        )
                        
                        if reconstructed_image is not None:
                            # 缓存重构图像
                            if 'hr_reconstructed_image' not in st.session_state or not np.array_equal(
                                st.session_state.get('hr_reconstructed_image'), reconstructed_image
                            ):
                                st.session_state.hr_reconstructed_image = reconstructed_image.copy()
                            
                            # 在重构图像上划线测量晶面间距（直接在大图上标记）
                            st.markdown("**📏 在下方 IFFT 重构大图上划线测量晶面间距**")
                            st.info("💡 请在重构图像的条纹上画一条线，并在下方输入跨越的晶格层数，系统将根据线长和层数计算晶面间距，同时给出 FFT 距离计算得到的晶面间距以对照")
                            
                            recon_h, recon_w = reconstructed_image.shape[:2]
                            # 限制canvas尺寸，避免大图无法显示（适当放大便于划线）
                            max_canvas_size = 1600
                            if max(recon_w, recon_h) <= max_canvas_size:
                                line_canvas_w = max(recon_w, 600)
                                line_canvas_h = max(recon_h, 600)
                                scale = min(line_canvas_w / recon_w, line_canvas_h / recon_h)
                                if scale != 1.0:
                                    new_w = int(recon_w * scale)
                                    new_h = int(recon_h * scale)
                                else:
                                    new_w, new_h = recon_w, recon_h
                                if len(reconstructed_image.shape) == 2:
                                    recon_pil = Image.fromarray(reconstructed_image, mode='L').convert('RGB')
                                else:
                                    recon_pil = Image.fromarray(reconstructed_image, mode='RGB')
                                if (new_w, new_h) != (recon_w, recon_h):
                                    recon_pil = recon_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                                line_canvas_w, line_canvas_h = new_w, new_h
                            else:
                                scale = max_canvas_size / max(recon_w, recon_h)
                                line_canvas_w = int(recon_w * scale)
                                line_canvas_h = int(recon_h * scale)
                                if len(reconstructed_image.shape) == 2:
                                    tmp = Image.fromarray(reconstructed_image, mode='L').convert('RGB')
                                else:
                                    tmp = Image.fromarray(reconstructed_image, mode='RGB')
                                recon_pil = tmp.resize((line_canvas_w, line_canvas_h), Image.Resampling.LANCZOS)
                            
                            # 创建划线canvas（缩放图避免无法显示）
                            try:
                                line_canvas_result = st_canvas(
                                    fill_color="rgba(255, 0, 0, 0.3)",
                                    stroke_width=2,
                                    stroke_color="#FF0000",
                                    background_image=recon_pil,
                                    update_streamlit=True,
                                    drawing_mode="line",
                                    point_display_radius=0,
                                    key="hr_line_canvas",
                                    width=line_canvas_w,
                                    height=line_canvas_h,
                                )
                            except Exception as e:
                                st.error(f"IFFT 划线区域显示错误: {e}")
                                line_canvas_result = None
                            
                            # 处理划线结果（canvas 坐标为缩放图，需换算回原图像素）
                            if line_canvas_result is not None and line_canvas_result.json_data is not None:
                                objects = line_canvas_result.json_data.get("objects", [])
                                if objects:
                                    last_obj = objects[-1]
                                    if last_obj.get("type") == "line":
                                        canvas_x1 = float(last_obj.get("x1", 0))
                                        canvas_y1 = float(last_obj.get("y1", 0))
                                        canvas_x2 = float(last_obj.get("x2", 0))
                                        canvas_y2 = float(last_obj.get("y2", 0))
                                        scale_x = recon_w / line_canvas_w
                                        scale_y = recon_h / line_canvas_h
                                        x1 = int(canvas_x1 * scale_x)
                                        y1 = int(canvas_y1 * scale_y)
                                        x2 = int(canvas_x2 * scale_x)
                                        y2 = int(canvas_y2 * scale_y)
                                        
                                        point1 = (x1, y1)
                                        point2 = (x2, y2)
                                        
                                        # 使用膜厚模块的划线分析方法
                                        from src.core.thickness.profile_analysis import ProfileThicknessAnalyzer
                                        profile_analyzer = ProfileThicknessAnalyzer(nm_per_pixel)
                                        
                                        # 提取线剖面（使用Bresenham算法）
                                        def extract_line_profile(img, pt1, pt2):
                                            """从图像中提取两点间直线的灰度剖面"""
                                            x1, y1 = pt1
                                            x2, y2 = pt2
                                            
                                            # 计算直线长度
                                            length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
                                            if length == 0:
                                                return np.array([])
                                            
                                            # 生成采样点
                                            t = np.linspace(0, 1, length)
                                            x = (x1 + (x2 - x1) * t).astype(int)
                                            y = (y1 + (y2 - y1) * t).astype(int)
                                            
                                            # 确保坐标在范围内
                                            h, w = img.shape[:2]
                                            x = np.clip(x, 0, w - 1)
                                            y = np.clip(y, 0, h - 1)
                                            
                                            # 提取灰度值
                                            if len(img.shape) == 2:
                                                profile = img[y, x]
                                            else:
                                                profile = np.mean(img[y, x], axis=1)
                                            
                                            return profile
                                        
                                        profile_raw = extract_line_profile(reconstructed_image, point1, point2)

                                        if len(profile_raw) >= 10:
                                            # 基于 IFFT 图像：用线段长度和层数计算晶面间距
                                            line_length_px = float(np.hypot(x2 - x1, y2 - y1))
                                            layers_count = st.number_input(
                                                "沿线跨越的晶格层数（条纹数）",
                                                min_value=1,
                                                value=1,
                                                step=1,
                                                key="hr_layers_count",
                                                help="例如跨越 5 条晶格条纹则填 5，系统将用线段物理长度除以层数得到单层晶面间距",
                                            )
                                            line_length_nm = line_length_px * nm_per_pixel
                                            d_manual_nm = line_length_nm / layers_count if layers_count > 0 else None
                                            st.markdown("### 基于 IFFT 图像的晶面间距（手动层数）")
                                            st.latex(r"""
                                            d_{\text{real}} = \frac{L_{\text{线段}}}{N_{\text{层}}}
                                            """)
                                            st.markdown(
                                                f"- 线段长度: {line_length_px:.2f} px × {nm_per_pixel:.6f} nm/px = **{line_length_nm:.4f} nm**  \n"
                                                f"- 层数: **{layers_count}**  \n"
                                                f"- 晶面间距 d_real = {line_length_nm:.4f} nm / {layers_count} = **{d_manual_nm:.4f} nm**"
                                            )

                                            # 基于 FFT 距离的理论晶面间距（与 SAED 的思路一致）
                                            st.markdown("### 基于 FFT 距离的理论晶面间距")
                                            # FFT 像素坐标中，中心到峰的距离 r_fft，图像尺寸 N，像素尺寸 s=nm_per_pixel
                                            # 单位倒易矢量步长 Δk = 1 / (N · s)，所以 d_fft = 1 / (r_fft · Δk) = N · s / r_fft
                                            fft_shifted = st.session_state.get('hr_fft_shifted')
                                            if fft_shifted is not None:
                                                fft_h, fft_w = fft_shifted.shape[:2]
                                                N_fft = float(max(fft_h, fft_w))
                                                dx_fft = float(spot['x'] - center_x)
                                                dy_fft = float(spot['y'] - center_y)
                                                r_fft = float(np.hypot(dx_fft, dy_fft))
                                                if r_fft > 1e-6:
                                                    d_fft_nm = N_fft * nm_per_pixel / r_fft
                                                    st.latex(r"""
                                                    d_{\text{FFT}} = \frac{N \cdot s}{r_{\text{FFT}}}
                                                    """)
                                                    st.markdown(
                                                        f"- 图像尺寸 N 取 max(H,W) = {N_fft:.0f} 像素  \n"
                                                        f"- 像素尺寸 s = {nm_per_pixel:.6f} nm/px  \n"
                                                        f"- FFT 中心到选取峰的像素距离 r_FFT = {r_fft:.2f} px  \n"
                                                        f"- 晶面间距 d_FFT = N·s / r_FFT"
                                                        f" = {N_fft:.0f} × {nm_per_pixel:.6f} / {r_fft:.2f}"
                                                        f" = **{d_fft_nm:.4f} nm**"
                                                    )
                                                else:
                                                    st.warning("⚠️ FFT 峰距过小，无法计算基于 FFT 的晶面间距")
                                        else:
                                            st.warning("⚠️ 剖面长度太短，请画一条更长的线")
                        else:
                            st.error("❌ IFFT重构失败")
            else:
                st.warning("⚠️ 请先完成像素比例换算")
    
    except Exception as e:
        st.error(f"❌ 处理失败: {str(e)}")
        import traceback
        with st.expander("错误详情"):
            st.code(traceback.format_exc())
    finally:
        # 清理临时文件
        pass
